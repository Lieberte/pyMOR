from __future__ import annotations

import csv
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import linalg

import mor  # noqa: F401
from mor.algorithm.auto.lyapunov import recommendLyapunovOptions, selectLyapunovAlgorithm
from mor.algorithm.registry import algorithmRegistry
from mor.operators import matrixOperator
from report_svg_utils import write_bar_chart_svg, write_line_chart_svg, write_success_matrix_svg
from report_run_utils import reset_report_dir, write_case_manifest
from standard_benchmark_cases import matrix_depot_poisson_1d
from standard_benchmark_cases import STANDARD_BENCHMARK_SOURCES


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "lradi"
JSON_PATH = REPORT_DIR / "lradi_results.json"
CSV_PATH = REPORT_DIR / "lradi_results.csv"
MD_PATH = REPORT_DIR / "LRADI_REPORT.md"
ANALYSIS_CSV_PATH = REPORT_DIR / "lradi_strategy_analysis.csv"
ANALYSIS_MD_PATH = REPORT_DIR / "LRADI_STRATEGY_ANALYSIS.md"
RESIDUAL_SVG_PATH = REPORT_DIR / "lradi_residuals.svg"
RUNTIME_SVG_PATH = REPORT_DIR / "lradi_runtime.svg"
DETAILED_RUNTIME_SVG_PATH = REPORT_DIR / "lradi_detailed_runtime.svg"
DETAILED_RESIDUAL_SVG_PATH = REPORT_DIR / "lradi_detailed_residual.svg"
SUCCESS_SVG_PATH = REPORT_DIR / "lradi_success_matrix.svg"


@dataclass
class LyapunovCase:
    name: str
    family: str
    problem_size: int
    A: Any
    B: np.ndarray
    strategy: str = "baseline"
    tol: float = 1e-10
    max_iter: int = 160
    compression_columns: int = 16
    max_shifts: int = 8
    reference_supported: bool = True


def build_cases() -> list[LyapunovCase]:
    rng = np.random.default_rng(13)
    cases: list[LyapunovCase] = []
    for n in (10, 100):
        cases.append(
            LyapunovCase(
                name=f"matrix_depot_poisson_1d_{n}_single_input_baseline",
                family="poisson_1d_single_input",
                problem_size=n,
                A=-matrix_depot_poisson_1d(n),
                B=np.ones((n, 1)),
            )
        )
    cases.extend(
        [
            LyapunovCase(
                name="matrix_depot_poisson_1d_1000_single_input_baseline",
                family="poisson_1d_single_input",
                problem_size=1000,
                A=-matrix_depot_poisson_1d(1000),
                B=np.ones((1000, 1)),
                compression_columns=32,
                max_shifts=10,
                reference_supported=False,
            ),
            LyapunovCase(
                name="matrix_depot_poisson_1d_1000_two_inputs_baseline",
                family="poisson_1d_two_inputs",
                problem_size=1000,
                A=-matrix_depot_poisson_1d(1000),
                B=rng.standard_normal((1000, 2)),
                compression_columns=32,
                max_shifts=10,
                reference_supported=False,
            ),
            LyapunovCase(
                name="matrix_depot_poisson_1d_1000_four_inputs_baseline",
                family="poisson_1d_four_inputs",
                problem_size=1000,
                A=-matrix_depot_poisson_1d(1000),
                B=rng.standard_normal((1000, 4)),
                compression_columns=48,
                max_shifts=10,
                reference_supported=False,
            ),
            LyapunovCase(
                name="matrix_depot_poisson_1d_1000_two_inputs_slow_scale",
                family="poisson_1d_scaled_slow",
                problem_size=1000,
                A=-0.01 * matrix_depot_poisson_1d(1000),
                B=rng.standard_normal((1000, 2)),
                compression_columns=32,
                max_shifts=10,
                reference_supported=False,
            ),
            LyapunovCase(
                name="matrix_depot_poisson_1d_1000_two_inputs_fast_scale",
                family="poisson_1d_scaled_fast",
                problem_size=1000,
                A=-100.0 * matrix_depot_poisson_1d(1000),
                B=rng.standard_normal((1000, 2)),
                compression_columns=32,
                max_shifts=10,
                reference_supported=False,
            ),
            LyapunovCase(
                name="matrix_depot_poisson_1d_10000_two_inputs_baseline",
                family="poisson_1d_two_inputs",
                problem_size=10000,
                A=-matrix_depot_poisson_1d(10000),
                B=rng.standard_normal((10000, 2)),
                strategy="baseline",
                compression_columns=40,
                max_shifts=12,
                reference_supported=False,
            ),
            LyapunovCase(
                name="matrix_depot_poisson_1d_10000_two_inputs_tuned",
                family="poisson_1d_two_inputs",
                problem_size=10000,
                A=-matrix_depot_poisson_1d(10000),
                B=rng.standard_normal((10000, 2)),
                strategy="auto_tuned",
                reference_supported=False,
            ),
        ]
    )
    return cases


def solve_dense_reference(a_dense: np.ndarray, b_data: np.ndarray) -> tuple[np.ndarray, float]:
    q_data = b_data @ b_data.T
    start = time.perf_counter()
    x_ref = linalg.solve_continuous_lyapunov(a_dense, -q_data)
    return x_ref, (time.perf_counter() - start) * 1000.0


def relative_residual(a_data: Any, x_data: np.ndarray, q_data: np.ndarray) -> float:
    residual = a_data @ x_data + x_data @ a_data.T + q_data
    return float(np.linalg.norm(residual) / max(np.linalg.norm(q_data), 1e-16))


def evaluate_case(case: LyapunovCase) -> dict[str, Any]:
    a_sparse = case.A.tocsr()
    a_dense = a_sparse.toarray()
    A = matrixOperator(a_sparse, backendName="scipy")
    B = matrixOperator(case.B, backendName="scipy")
    q_data = case.B @ case.B.T

    auto_variant = selectLyapunovAlgorithm(
        variant="auto",
        A=A,
        backendName="scipy",
        isContinuous=True,
    )

    max_iter = case.max_iter
    compression_columns = case.compression_columns
    max_shifts = case.max_shifts
    if case.strategy == "auto_tuned":
        recommended = recommendLyapunovOptions(
            variant="auto",
            A=A,
            B=B,
            backendName="scipy",
            isContinuous=True,
        )
        max_iter = recommended.get("maxIter", max_iter)
        compression_columns = recommended.get("compressionColumns", compression_columns)
        max_shifts = recommended.get("shiftOptions", {}).get("maxShifts", max_shifts)

    lradi_algo = algorithmRegistry.get(
        category="lyapunov",
        variant="lradi",
        backendName="scipy",
        tol=case.tol,
        maxIter=max_iter,
        compressionColumns=compression_columns,
        shiftOptions={"maxShifts": max_shifts},
    )
    start = time.perf_counter()
    Z = lradi_algo.solve(A, None, B)
    lradi_elapsed_ms = (time.perf_counter() - start) * 1000.0
    x_lradi = Z.toBackendData()

    if case.reference_supported:
        x_ref, dense_elapsed_ms = solve_dense_reference(a_dense, case.B)
        approximation_error = float(
            np.linalg.norm(x_lradi - x_ref) / max(np.linalg.norm(x_ref), 1e-16)
        )
    else:
        x_ref, dense_elapsed_ms, approximation_error = None, None, None

    results = [
        {
            "variant": "lradi",
            "elapsed_ms": round(lradi_elapsed_ms, 3),
            "rank": int(Z.left.shape[1]),
            "iterations": int(lradi_algo.lastIterations),
            "relative_residual": relative_residual(a_sparse, x_lradi, q_data),
            "relative_reference_error": approximation_error,
            "final_projected_residual": float(lradi_algo.lastResidualHistory[-1]),
            "num_recorded_residuals": len(lradi_algo.lastResidualHistory),
        },
        {
            "variant": "dense_reference",
            "elapsed_ms": round(dense_elapsed_ms, 3) if dense_elapsed_ms is not None else None,
            "rank": int(np.linalg.matrix_rank(x_ref)) if x_ref is not None else None,
            "iterations": 1 if x_ref is not None else None,
            "relative_residual": relative_residual(a_dense, x_ref, q_data) if x_ref is not None else None,
            "relative_reference_error": 0.0 if x_ref is not None else None,
            "final_projected_residual": None,
            "num_recorded_residuals": None,
            "supported": x_ref is not None,
        },
    ]

    return {
        "case": case.name,
        "family": case.family,
        "problem_size": case.problem_size,
        "backend": "scipy",
        "operator_kind": "scipy_sparse_csr_diagonal",
        "shape": [int(a_sparse.shape[0]), int(a_sparse.shape[1])],
        "inputs": int(case.B.shape[1]),
        "strategy": case.strategy,
        "auto_variant": auto_variant,
        "tolerance": case.tol,
        "max_iter": max_iter,
        "compression_columns": compression_columns,
        "max_shifts": max_shifts,
        "results": results,
    }


def flatten_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for result in case["results"]:
            rows.append(
                {
                    "case": case["case"],
                    "family": case["family"],
                    "problem_size": case["problem_size"],
                    "backend": case["backend"],
                    "operator_kind": case["operator_kind"],
                    "shape": "x".join(str(x) for x in case["shape"]),
                    "inputs": case["inputs"],
                    "strategy": case["strategy"],
                    "auto_variant": case["auto_variant"],
                    "variant": result["variant"],
                    "elapsed_ms": result["elapsed_ms"],
                    "rank": result["rank"],
                    "iterations": result["iterations"],
                    "relative_residual": result["relative_residual"],
                    "relative_reference_error": result["relative_reference_error"],
                    "max_iter": case["max_iter"],
                    "compression_columns": case["compression_columns"],
                    "max_shifts": case["max_shifts"],
                    "success": result["relative_residual"] < 1e-8 if result["variant"] == "lradi" else (result["relative_residual"] is not None and result["relative_residual"] < 1e-10),
                }
            )
    return rows


def build_strategy_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    analysis = []
    for row in rows:
        if row["variant"] != "lradi":
            continue
        residual = float(row["relative_residual"])
        iterations = int(row["iterations"])
        max_iter = int(row["max_iter"])
        if residual >= 1e-8:
            recommendation = "increase_iterations_or_shifts"
        elif iterations >= 0.9 * max_iter:
            recommendation = "monitor_iteration_budget"
        elif int(row["inputs"]) >= 4:
            recommendation = "monitor_input_rank_growth"
        else:
            recommendation = "keep"
        analysis.append(
            {
                "case": row["case"],
                "family": row["family"],
                "problem_size": row["problem_size"],
                "inputs": row["inputs"],
                "strategy": row["strategy"],
                "elapsed_ms": row["elapsed_ms"],
                "rank": row["rank"],
                "iterations": row["iterations"],
                "relative_residual": row["relative_residual"],
                "max_iter": row["max_iter"],
                "compression_columns": row["compression_columns"],
                "max_shifts": row["max_shifts"],
                "success": row["success"],
                "recommendation": recommendation,
            }
        )
    return analysis


def write_strategy_analysis(analysis: list[dict[str, Any]]) -> None:
    if not analysis:
        return
    with ANALYSIS_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(analysis[0].keys()))
        writer.writeheader()
        writer.writerows(analysis)
    family_lines = []
    for family in sorted({row["family"] for row in analysis}):
        family_rows = [row for row in analysis if row["family"] == family]
        failures = sum(not bool(row["success"]) for row in family_rows)
        max_residual = max(float(row["relative_residual"]) for row in family_rows)
        max_rank = max(int(row["rank"]) for row in family_rows)
        family_lines.append(
            f"- `{family}`: 案例数={len(family_rows)}, 失败数={failures}, "
            f"最大残差={max_residual:.3e}, 最大秩={max_rank}"
        )
    review_rows = [row for row in analysis if row["recommendation"] != "keep"]
    review_lines = [
        f"- `{row['case']}`: 残差={float(row['relative_residual']):.3e}, "
        f"迭代={row['iterations']}, 秩={row['rank']}, 建议=`{row['recommendation']}`"
        for row in review_rows
    ]
    ANALYSIS_MD_PATH.write_text(
        "\n".join(
            [
                "# LRADI 策略分析",
                "",
                "本文汇总 LRADI 的收敛、秩增长和参数预算表现。",
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


def write_csv(rows: list[dict[str, Any]]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: dict[str, Any]) -> None:
    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_bar_svg(
    rows: list[dict[str, Any]],
    path: Path,
    title: str,
    value_key: str,
    value_label: str,
    log10: bool = False,
) -> None:
    lradi_rows = [row for row in rows if row["variant"] == "lradi"]
    labeled_rows = [{**row, "display_label": f"{row['problem_size']}|{row['inputs']}|{row['strategy'][:1]}"} for row in lradi_rows]
    write_bar_chart_svg(labeled_rows, path, title, value_key, value_label, "display_label", log10=log10, color="#256d7b")


def write_visualizations(rows: list[dict[str, Any]]) -> None:
    write_bar_svg(
        rows,
        RESIDUAL_SVG_PATH,
        "LRADI Relative Residuals",
        "relative_residual",
        "Lower is better; bars show lradi rows only on log10 scale.",
        log10=True,
    )
    write_bar_svg(
        rows,
        RUNTIME_SVG_PATH,
        "LRADI Runtime",
        "elapsed_ms",
        "Single-run local timings in milliseconds; bars show lradi rows only.",
    )
    series_runtime = []
    series_residual = []
    for variant in ("lradi", "dense_reference"):
        variant_rows = sorted([row for row in rows if row["variant"] == variant and row["elapsed_ms"] is not None], key=lambda item: item["problem_size"])
        series_runtime.append(
            {
                "name": variant,
                "points": [{"x": idx + 1, "y": row["elapsed_ms"], "label": f"{row['problem_size']}|{row['inputs']}|{row['strategy'][:1]}"} for idx, row in enumerate(variant_rows)],
            }
        )
        if variant_rows:
            series_residual.append(
            {
                "name": variant,
                "points": [{"x": idx + 1, "y": row["relative_residual"], "label": f"{row['problem_size']}|{row['inputs']}|{row['strategy'][:1]}"} for idx, row in enumerate(variant_rows)],
            }
            )
    write_line_chart_svg(series_runtime, DETAILED_RUNTIME_SVG_PATH, "LRADI Detailed Runtime Sweep", "problem size", "elapsed ms")
    write_line_chart_svg(series_residual, DETAILED_RESIDUAL_SVG_PATH, "LRADI Detailed Residual Sweep", "problem size", "relative residual", log10=True)
    success_cases = [f"{row['problem_size']}|{row['inputs']}|{row['strategy'][:1]}" for row in rows if row["variant"] == "lradi"]
    success_map = {(row["variant"], f"{row['problem_size']}|{row['inputs']}|{row['strategy'][:1]}"): bool(row["success"]) for row in rows}
    write_success_matrix_svg(["lradi", "dense_reference"], success_cases, success_map, SUCCESS_SVG_PATH, "LRADI Success Matrix")


def write_markdown(payload: dict[str, Any]) -> None:
    case_lines = []
    for case in payload["cases"]:
        lradi_result = next(item for item in case["results"] if item["variant"] == "lradi")
        dense_result = next(item for item in case["results"] if item["variant"] == "dense_reference")
        reference_error = (
            f"{lradi_result['relative_reference_error']:.3e}"
            if lradi_result["relative_reference_error"] is not None
            else "n/a"
        )
        dense_timing = f"{dense_result['elapsed_ms']} ms" if dense_result["elapsed_ms"] is not None else "n/a"
        case_lines.append(
            f"- `{case['case']}`: auto -> `{case['auto_variant']}`, "
            f"策略 -> `{case['strategy']}`, "
            f"LRADI 残差 -> {lradi_result['relative_residual']:.3e}, "
            f"参考误差 -> {reference_error}, "
            f"迭代 -> {lradi_result['iterations']}, "
            f"LRADI/dense 耗时 -> {lradi_result['elapsed_ms']} ms / {dense_timing}"
        )

    summary = f"""# LRADI 报告

生成时间：`{payload['generated_at']}`

环境：
- Python: `{payload['environment']['python']}`
- Platform: `{payload['environment']['platform']}`
- NumPy: `{payload['environment']['numpy']}`
- SciPy: `{payload['environment']['scipy']}`

## 范围

本文汇总 classical MOR 优化中 LRADI 部分的结果：
- 稀疏连续 Lyapunov 的 `lradi` 行为
- 稀疏 operator 兼容性
- 稀疏连续案例的 Lyapunov `auto` 选择
- 与 dense reference solve 的本地耗时和残差对比

## 案例摘要

{chr(10).join(case_lines)}

## 可视化

- `{RESIDUAL_SVG_PATH.name}`
- `{RUNTIME_SVG_PATH.name}`
- `{DETAILED_RESIDUAL_SVG_PATH.name}`
- `{DETAILED_RUNTIME_SVG_PATH.name}`
- `{SUCCESS_SVG_PATH.name}`
- `{ANALYSIS_MD_PATH.name}`

## 说明

- 耗时为单次本地测量，只作为粗略比较信号。
- dense reference solve 仅在规模允许时用于精度检查，不作为稀疏默认路径。
- `auto` 的判断依据是稀疏场景适配性和 operator 兼容性。
- 标准稀疏 Poisson 案例来自 MatrixDepot 风格的一维 Poisson stencil：
    {STANDARD_BENCHMARK_SOURCES['julia_matrix_depot']['url']}
- 完整机器可读结果见：
    - `{JSON_PATH.name}`
    - `{CSV_PATH.name}`
    - `{ANALYSIS_CSV_PATH.name}`
"""
    MD_PATH.write_text(summary, encoding="utf-8-sig")


def main() -> int:
    import scipy

    reset_report_dir(REPORT_DIR)
    cases = [evaluate_case(case) for case in build_cases()]
    manifest_cases = [
        {
            "case": case.name,
            "family": case.family,
            "problem_size": case.problem_size,
            "inputs": int(case.B.shape[1]),
            "strategy": case.strategy,
            "reference_supported": case.reference_supported,
            "source_name": "Julia MatrixDepot",
            "source_case": "poisson_1d",
            "source_url": STANDARD_BENCHMARK_SOURCES["julia_matrix_depot"]["url"],
        }
        for case in build_cases()
    ]
    rows = flatten_rows(cases)
    analysis = build_strategy_analysis(rows)
    payload = {
        "phase": "lradi",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
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
    write_json(payload)
    write_csv(rows)
    write_strategy_analysis(analysis)
    manifest_json, manifest_csv = write_case_manifest(REPORT_DIR, "lradi", manifest_cases)
    write_visualizations(rows)
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
