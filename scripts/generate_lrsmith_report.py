from __future__ import annotations

import csv
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix

import mor  # noqa: F401
from mor.algorithm.auto.lyapunov import selectLyapunovAlgorithm
from mor.algorithm.registry import algorithmRegistry
from mor.operators import matrixOperator
from report_svg_utils import write_bar_chart_svg, write_line_chart_svg, write_success_matrix_svg
from report_run_utils import reset_report_dir, write_case_manifest
from standard_benchmark_cases import STANDARD_BENCHMARK_SOURCES, matrix_depot_poisson_1d


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "lrsmith"
JSON_PATH = REPORT_DIR / "lrsmith_results.json"
CSV_PATH = REPORT_DIR / "lrsmith_results.csv"
MD_PATH = REPORT_DIR / "LRSMITH_REPORT.md"
ANALYSIS_CSV_PATH = REPORT_DIR / "lrsmith_strategy_analysis.csv"
ANALYSIS_MD_PATH = REPORT_DIR / "LRSMITH_STRATEGY_ANALYSIS.md"
RESIDUAL_SVG_PATH = REPORT_DIR / "lrsmith_residuals.svg"
RUNTIME_SVG_PATH = REPORT_DIR / "lrsmith_runtime.svg"
DETAILED_RUNTIME_SVG_PATH = REPORT_DIR / "lrsmith_detailed_runtime.svg"
DETAILED_RESIDUAL_SVG_PATH = REPORT_DIR / "lrsmith_detailed_residual.svg"
SUCCESS_SVG_PATH = REPORT_DIR / "lrsmith_success_matrix.svg"


@dataclass
class DiscreteCase:
    name: str
    family: str
    problem_size: int
    A: np.ndarray
    B: np.ndarray
    E: np.ndarray | None = None
    max_iter: int = 200
    tol: float = 1e-12
    reference_supported: bool = True


def near_unit_discrete_poisson_1d(n: int) -> np.ndarray:
    p = matrix_depot_poisson_1d(n).toarray()
    return np.eye(n) - 0.2 * p


def contractive_discrete_poisson_1d(n: int):
    return 0.2 * matrix_depot_poisson_1d(n)


def two_input_rhs(n: int) -> np.ndarray:
    return np.column_stack([np.ones(n), np.linspace(0.5, 1.5, n)])


def duplicated_two_input_rhs(n: int) -> np.ndarray:
    return np.column_stack([np.ones(n), np.ones(n)])


def build_cases() -> list[DiscreteCase]:
    return [
        DiscreteCase("diagonal_contracting_n40", "diagonal_contracting", 40, np.diag(np.linspace(0.1, 0.6, 40)), np.ones((40, 1)), max_iter=120),
        DiscreteCase("diagonal_near_unit_n100", "diagonal_near_unit", 100, np.diag(np.linspace(0.1, 0.95, 100)), np.ones((100, 1)), max_iter=400),
        DiscreteCase("diagonal_extreme_near_unit_n80", "diagonal_near_unit", 80, np.diag(np.linspace(0.1, 0.995, 80)), np.ones((80, 1)), max_iter=320),
        DiscreteCase("diagonal_unstable_n40", "diagonal_unstable", 40, np.diag(np.linspace(0.8, 1.05, 40)), np.ones((40, 1)), max_iter=80, reference_supported=False),
        DiscreteCase("diagonal_zero_rhs_n40", "zero_rhs", 40, np.diag(np.linspace(0.1, 0.6, 40)), np.zeros((40, 1)), max_iter=40),
        DiscreteCase("descriptor_diagonal_contracting_n40", "descriptor_contracting", 40, np.diag(np.linspace(0.2, 0.5, 40)), np.ones((40, 1)), E=np.diag(np.linspace(1.0, 3.0, 40)), max_iter=120),
        DiscreteCase("matrix_depot_poisson_1d_100_contracting", "poisson_1d_contracting", 100, contractive_discrete_poisson_1d(100), np.ones((100, 1)), max_iter=160),
        DiscreteCase("matrix_depot_poisson_1d_100_contracting_two_inputs", "poisson_1d_contracting_multi_input", 100, contractive_discrete_poisson_1d(100), two_input_rhs(100), max_iter=160),
        DiscreteCase("matrix_depot_poisson_1d_100_contracting_duplicated_inputs", "poisson_1d_contracting_rank_deficient_input", 100, contractive_discrete_poisson_1d(100), duplicated_two_input_rhs(100), max_iter=160),
        DiscreteCase("matrix_depot_poisson_1d_100_near_unit", "poisson_1d_near_unit", 100, near_unit_discrete_poisson_1d(100), np.ones((100, 1)), max_iter=400),
        DiscreteCase("matrix_depot_poisson_1d_1000_contracting_two_inputs", "poisson_1d_contracting_multi_input", 1000, contractive_discrete_poisson_1d(1000), two_input_rhs(1000), max_iter=220, reference_supported=False),
        DiscreteCase("matrix_depot_poisson_1d_10000_contracting_two_inputs", "poisson_1d_contracting_multi_input", 10000, contractive_discrete_poisson_1d(10000), two_input_rhs(10000), max_iter=260, reference_supported=False),
    ]


def discrete_relative_residual(a_data, x_data, q_data) -> float:
    residual = a_data @ x_data @ a_data.T - x_data + q_data
    return float(np.linalg.norm(residual) / max(np.linalg.norm(q_data), 1e-16))


def lrsmith_relative_residual(smith, A: matrixOperator, E: matrixOperator | None, Z, B: matrixOperator) -> float:
    z_data = Z.left
    b_data = B.toBackendData()
    az_data = smith._advance(A, E, z_data)
    residual = smith.discreteLyapunovResidualNorm(az_data, z_data, b_data)
    q_norm = np.linalg.norm(b_data.T @ b_data)
    return float(residual / max(q_norm, 1e-16))


def spectral_radius(a_data) -> float | None:
    if hasattr(a_data, "toarray"):
        a_data = a_data.toarray()
    if a_data.shape[0] > 300:
        return None
    return float(max(abs(np.linalg.eigvals(a_data))))


def evaluate_case(case: DiscreteCase) -> dict:
    a_sparse = csr_matrix(case.A)
    A = matrixOperator(a_sparse, backendName="scipy")
    E = matrixOperator(case.E, backendName="scipy") if case.E is not None else None
    B = matrixOperator(case.B, backendName="scipy")
    a_dense = case.A.toarray() if hasattr(case.A, "toarray") else case.A
    effective_dense = np.linalg.solve(case.E, a_dense) if case.E is not None else a_dense
    q_data = case.B @ case.B.T if case.reference_supported else None
    auto_variant = selectLyapunovAlgorithm(variant="auto", A=A, E=E, backendName="scipy", isContinuous=False)

    smith = algorithmRegistry.get(
        category="lyapunov",
        variant="lrsmith",
        backendName="scipy",
        maxIter=case.max_iter,
        tol=case.tol,
        checkStability=False,
    )
    start = time.perf_counter()
    Z = smith.solve(A, E, B)
    smith_elapsed_ms = (time.perf_counter() - start) * 1000.0
    smith_residual = lrsmith_relative_residual(smith, A, E, Z, B)

    rho = spectral_radius(effective_dense)

    if case.reference_supported:
        start = time.perf_counter()
        x_ref = linalg.solve_discrete_lyapunov(effective_dense, q_data)
        dense_elapsed_ms = (time.perf_counter() - start) * 1000.0
        x_smith = Z.toBackendData()
        smith_residual = discrete_relative_residual(effective_dense, x_smith, q_data)
        reference_error = float(np.linalg.norm(x_smith - x_ref) / max(np.linalg.norm(x_ref), 1e-16))
        dense_residual = discrete_relative_residual(effective_dense, x_ref, q_data)
    else:
        x_ref, dense_elapsed_ms, reference_error, dense_residual = None, None, None, None

    results = [
        {
            "variant": "lrsmith",
            "elapsed_ms": round(smith_elapsed_ms, 3),
            "rank": int(Z.left.shape[1]),
            "iterations": int(smith.lastIterations),
            "relative_residual": smith_residual,
            "relative_reference_error": reference_error,
        },
        {
            "variant": "dense_reference",
            "elapsed_ms": round(dense_elapsed_ms, 3) if dense_elapsed_ms is not None else None,
            "rank": int(np.linalg.matrix_rank(x_ref)) if x_ref is not None else None,
            "iterations": 1 if x_ref is not None else None,
            "relative_residual": dense_residual,
            "relative_reference_error": 0.0 if x_ref is not None else None,
            "supported": x_ref is not None,
        },
    ]

    return {
        "case": case.name,
        "family": case.family,
        "problem_size": case.problem_size,
        "backend": "scipy",
        "operator_kind": "sparse_matrixOperator",
        "data_regime": "standard_sparse_discrete_stable",
        "shape": list(case.A.shape),
        "descriptor": case.E is not None,
        "inputs": int(case.B.shape[1]),
        "auto_variant": auto_variant,
        "spectral_radius": rho,
        "max_iter": case.max_iter,
        "results": results,
    }


def flatten_rows(cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        for result in case["results"]:
            rows.append(
                {
                    "case": case["case"],
                    "family": case["family"],
                    "problem_size": case["problem_size"],
                    "backend": case["backend"],
                    "operator_kind": case["operator_kind"],
                    "data_regime": case["data_regime"],
                    "shape": "x".join(str(x) for x in case["shape"]),
                    "descriptor": case["descriptor"],
                    "inputs": case["inputs"],
                    "auto_variant": case["auto_variant"],
                    "spectral_radius": case["spectral_radius"],
                    "max_iter": case["max_iter"],
                    "variant": result["variant"],
                    "elapsed_ms": result["elapsed_ms"],
                    "rank": result["rank"],
                    "iterations": result["iterations"],
                    "relative_residual": result["relative_residual"],
                    "relative_reference_error": result["relative_reference_error"],
                    "success": result["relative_residual"] < 1e-8 if result["variant"] == "lrsmith" else (result["relative_residual"] is not None and result["relative_residual"] < 1e-10),
                }
            )
    return rows


def build_strategy_analysis(rows: list[dict]) -> list[dict]:
    analysis = []
    for row in rows:
        if row["variant"] != "lrsmith":
            continue
        residual = float(row["relative_residual"])
        iterations = int(row["iterations"])
        max_iter = int(row["max_iter"])
        rho = None if row["spectral_radius"] in (None, "") else float(row["spectral_radius"])
        if rho is not None and rho > 1.0:
            recommendation = "reject_unstable_discrete_operator"
        elif residual >= 1e-8 and rho is not None and rho > 0.9:
            recommendation = "near_unit_slow_convergence"
        elif 1e-8 <= residual < 1e-7:
            recommendation = "monitor_residual_estimator_floor"
        elif residual >= 1e-8:
            recommendation = "fix_algorithm_or_budget"
        elif iterations >= 0.9 * max_iter:
            recommendation = "monitor_iteration_budget"
        else:
            recommendation = "keep"
        analysis.append(
            {
                "case": row["case"],
                "family": row["family"],
                "problem_size": row["problem_size"],
                "inputs": row["inputs"],
                "descriptor": row["descriptor"],
                "spectral_radius": row["spectral_radius"],
                "max_iter": row["max_iter"],
                "elapsed_ms": row["elapsed_ms"],
                "rank": row["rank"],
                "iterations": row["iterations"],
                "relative_residual": row["relative_residual"],
                "relative_reference_error": row["relative_reference_error"],
                "success": row["success"],
                "recommendation": recommendation,
            }
        )
    return analysis


def write_strategy_analysis(analysis: list[dict]) -> None:
    if not analysis:
        return
    with ANALYSIS_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(analysis[0].keys()))
        writer.writeheader()
        writer.writerows(analysis)
    family_lines = []
    for family in sorted({row["family"] for row in analysis}):
        family_rows = [row for row in analysis if row["family"] == family]
        failures = sum(row["success"] != "True" and row["success"] is not True for row in family_rows)
        max_residual = max(float(row["relative_residual"]) for row in family_rows)
        max_rank = max(int(row["rank"]) for row in family_rows)
        family_lines.append(
            f"- `{family}`: 案例数={len(family_rows)}, 失败数={failures}, "
            f"最大残差={max_residual:.3e}, 最大秩={max_rank}"
        )
    review_rows = [row for row in analysis if row["recommendation"] != "keep"]
    review_lines = [
        f"- `{row['case']}`: rho={row['spectral_radius']}, 残差={float(row['relative_residual']):.3e}, "
        f"迭代={row['iterations']}, 秩={row['rank']}, 建议=`{row['recommendation']}`"
        for row in review_rows
    ]
    ANALYSIS_MD_PATH.write_text(
        "\n".join(
            [
                "# LRSmith 策略分析",
                "",
                "本文区分 LRSmith 正确性问题和接近单位圆时的慢收敛问题。",
                "",
                "## 案例族汇总",
                "",
                *family_lines,
                "",
                "## 需要复核的案例",
                "",
                *(review_lines or ["- 无"]),
                "",
                f"完整逐案例策略数据见：`{ANALYSIS_CSV_PATH.name}`",
                "",
            ]
        ),
        encoding="utf-8-sig",
    )


def write_csv(rows: list[dict]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _svg_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def write_bar_svg(rows: list[dict], path: Path, title: str, value_key: str, log10: bool = False) -> None:
    smith_rows = [row for row in rows if row["variant"] == "lrsmith"]
    labeled_rows = [{**row, "display_label": f"{row['problem_size']}|{row['inputs']}"} for row in smith_rows]
    write_bar_chart_svg(labeled_rows, path, title, value_key, "summary bars", "display_label", log10=log10, color="#7b5d73")


def write_markdown(payload: dict) -> None:
    lines = []
    for case in payload["cases"]:
        smith = next(item for item in case["results"] if item["variant"] == "lrsmith")
        dense = next(item for item in case["results"] if item["variant"] == "dense_reference")
        reference_error = (
            f"{smith['relative_reference_error']:.3e}"
            if smith["relative_reference_error"] is not None
            else "n/a"
        )
        dense_timing = f"{dense['elapsed_ms']} ms" if dense["elapsed_ms"] is not None else "n/a"
        lines.append(
            f"- `{case['case']}`: auto -> `{case['auto_variant']}`, "
            f"LRSmith 残差 -> {smith['relative_residual']:.3e}, "
            f"参考误差 -> {reference_error}, "
            f"迭代 -> {smith['iterations']}, "
            f"LRSmith/dense 耗时 -> {smith['elapsed_ms']} ms / {dense_timing}"
        )
    MD_PATH.write_text(
        f"""# LRSmith 报告

生成时间：`{payload['generated_at']}`

环境：
- Python: `{payload['environment']['python']}`
- Platform: `{payload['environment']['platform']}`
- NumPy: `{payload['environment']['numpy']}`

## 范围

本文覆盖稀疏离散 Lyapunov 在 `lrsmith` 路径上的行为。

## 案例摘要

{chr(10).join(lines)}

## 可视化

- `{RESIDUAL_SVG_PATH.name}`
- `{RUNTIME_SVG_PATH.name}`
- `{DETAILED_RESIDUAL_SVG_PATH.name}`
- `{DETAILED_RUNTIME_SVG_PATH.name}`
- `{SUCCESS_SVG_PATH.name}`

## 说明

- dense reference solve 仅在规模允许时用于精度检查，不作为稀疏默认路径。
- `auto` 对稀疏离散案例选择 `lrsmith`。
- 策略分析见 `{ANALYSIS_MD_PATH.name}`。
""",
        encoding="utf-8-sig",
    )


def main() -> int:
    reset_report_dir(REPORT_DIR)
    cases = [evaluate_case(case) for case in build_cases()]
    manifest_cases = [
        {
            "case": case.name,
            "family": case.family,
            "problem_size": case.problem_size,
            "inputs": int(case.B.shape[1]),
            "descriptor": case.E is not None,
            "reference_supported": case.reference_supported,
            "source_name": "Julia MatrixDepot" if "poisson" in case.family else "local-generated",
            "source_case": "poisson_1d" if "poisson" in case.family else case.family,
            "source_url": STANDARD_BENCHMARK_SOURCES["julia_matrix_depot"]["url"] if "poisson" in case.family else "local-generated",
        }
        for case in build_cases()
    ]
    rows = flatten_rows(cases)
    analysis = build_strategy_analysis(rows)
    payload = {
        "phase": "lrsmith",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "visualizations": [
            RESIDUAL_SVG_PATH.name,
            RUNTIME_SVG_PATH.name,
            DETAILED_RESIDUAL_SVG_PATH.name,
            DETAILED_RUNTIME_SVG_PATH.name,
            SUCCESS_SVG_PATH.name,
        ],
        "cases": cases,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(rows)
    write_strategy_analysis(analysis)
    manifest_json, manifest_csv = write_case_manifest(REPORT_DIR, "lrsmith", manifest_cases)
    write_bar_svg(rows, RESIDUAL_SVG_PATH, "LRSmith Relative Residuals", "relative_residual", log10=True)
    write_bar_svg(rows, RUNTIME_SVG_PATH, "LRSmith Runtime", "elapsed_ms")
    series_runtime = []
    series_residual = []
    for variant in ("lrsmith", "dense_reference"):
        variant_rows = sorted([row for row in rows if row["variant"] == variant and row["elapsed_ms"] is not None], key=lambda item: item["problem_size"])
        series_runtime.append(
            {
                "name": variant,
                "points": [{"x": row["problem_size"], "y": row["elapsed_ms"], "label": str(row["problem_size"])} for row in variant_rows],
            }
        )
        if variant_rows:
            series_residual.append(
                {
                    "name": variant,
                    "points": [{"x": row["problem_size"], "y": row["relative_residual"], "label": str(row["problem_size"])} for row in variant_rows],
                }
            )
    write_line_chart_svg(series_runtime, DETAILED_RUNTIME_SVG_PATH, "LRSmith Detailed Runtime Sweep", "problem size", "elapsed ms")
    write_line_chart_svg(series_residual, DETAILED_RESIDUAL_SVG_PATH, "LRSmith Detailed Residual Sweep", "problem size", "relative residual", log10=True)
    success_cases = [row["case"] for row in rows if row["variant"] == "lrsmith"]
    success_map = {(row["variant"], row["case"]): bool(row["success"]) for row in rows}
    write_success_matrix_svg(["lrsmith", "dense_reference"], success_cases, success_map, SUCCESS_SVG_PATH, "LRSmith Success Matrix")
    write_markdown(payload)
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {MD_PATH}")
    print(f"Wrote {ANALYSIS_MD_PATH}")
    print(f"Wrote {ANALYSIS_CSV_PATH}")
    print(f"Wrote {RESIDUAL_SVG_PATH}")
    print(f"Wrote {RUNTIME_SVG_PATH}")
    print(f"Wrote {DETAILED_RESIDUAL_SVG_PATH}")
    print(f"Wrote {DETAILED_RUNTIME_SVG_PATH}")
    print(f"Wrote {SUCCESS_SVG_PATH}")
    print(f"Wrote {manifest_json}")
    print(f"Wrote {manifest_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
