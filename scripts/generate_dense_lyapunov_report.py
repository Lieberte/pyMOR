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

import mor  # noqa: F401
from mor.algorithm.auto.lyapunov import selectLyapunovAlgorithm
from mor.algorithm.registry import algorithmRegistry
from mor.operators import matrixOperator
from report_svg_utils import write_bar_chart_svg, write_line_chart_svg, write_success_matrix_svg
from report_run_utils import reset_report_dir, write_case_manifest
from standard_benchmark_cases import ill_conditioned_diagonal, matlab_gallery_grcar, matlab_gallery_hilbert

import torch


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "dense_lyapunov"
JSON_PATH = REPORT_DIR / "dense_lyapunov_results.json"
CSV_PATH = REPORT_DIR / "dense_lyapunov_results.csv"
MD_PATH = REPORT_DIR / "DENSE_LYAPUNOV_REPORT.md"
RESIDUAL_SVG_PATH = REPORT_DIR / "dense_lyapunov_residuals.svg"
RUNTIME_SVG_PATH = REPORT_DIR / "dense_lyapunov_runtime.svg"
DETAILED_RUNTIME_SVG_PATH = REPORT_DIR / "dense_lyapunov_detailed_runtime.svg"
DETAILED_RESIDUAL_SVG_PATH = REPORT_DIR / "dense_lyapunov_detailed_residual.svg"
SUCCESS_SVG_PATH = REPORT_DIR / "dense_lyapunov_success_matrix.svg"


@dataclass
class DenseLyapunovCase:
    name: str
    family: str
    problem_size: int
    A: np.ndarray
    B: np.ndarray
    backend: str = "scipy"


def stable_grcar(n: int) -> np.ndarray:
    G = matlab_gallery_grcar(n)
    eig_radius = max(abs(np.linalg.eigvals(G)))
    return G / (2.0 * eig_radius) - 1.5 * np.eye(n)


def stable_hilbert(n: int) -> np.ndarray:
    H = matlab_gallery_hilbert(n)
    return -H - 0.5 * np.eye(n)


def stable_diagonal(n: int) -> np.ndarray:
    return -ill_conditioned_diagonal(n, min_exp=-6.0) - 0.1 * np.eye(n)


def build_cases() -> list[DenseLyapunovCase]:
    cases = [
        DenseLyapunovCase("matlab_gallery_grcar_6_scipy", "grcar_dense", 6, stable_grcar(6), np.ones((6, 1))),
        DenseLyapunovCase("matlab_gallery_grcar_10_scipy", "grcar_dense", 10, stable_grcar(10), np.ones((10, 2))),
        DenseLyapunovCase("matlab_gallery_grcar_100_scipy", "grcar_dense", 100, stable_grcar(100), np.ones((100, 2))),
        DenseLyapunovCase("matlab_gallery_hilbert_10_scipy", "hilbert_dense", 10, stable_hilbert(10), np.ones((10, 1))),
    ]
    for n, device_name in ((10, "cpu"), (100, "cpu"), (1000, "cpu")):
        cases.append(DenseLyapunovCase(f"torch_{device_name}_diagonal_{n}", f"torch_{device_name}_diagonal", n, stable_diagonal(n), np.ones((n, 1)), backend="torch"))
    if torch.cuda.is_available():
        for n in (10, 100, 1000):
            cases.append(DenseLyapunovCase(f"torch_cuda_diagonal_{n}", "torch_cuda_diagonal", n, stable_diagonal(n), np.ones((n, 1)), backend="torch"))
    return cases


def to_numpy(data: Any) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def relative_residual(a_data: Any, x_data: Any, q_data: Any) -> float:
    a_np, x_np, q_np = to_numpy(a_data), to_numpy(x_data), to_numpy(q_data)
    residual = a_np @ x_np + x_np @ a_np.T + q_np
    return float(np.linalg.norm(residual) / max(np.linalg.norm(q_np), 1e-16))


def evaluate_case(case: DenseLyapunovCase) -> dict[str, Any]:
    if case.backend == "torch":
        device = "cuda" if "cuda" in case.family else "cpu"
        a_data = torch.tensor(case.A, dtype=torch.float64, device=device)
        b_data = torch.tensor(case.B, dtype=torch.float64, device=device)
    else:
        a_data = case.A
        b_data = case.B

    A = matrixOperator(a_data, backendName=case.backend)
    B = matrixOperator(b_data, backendName=case.backend)
    q_data = B.toBackendData() @ B.toBackendData().T
    auto_variant = selectLyapunovAlgorithm(variant="auto", A=A, backendName=case.backend, isContinuous=True)
    variants = ["sign"] if case.backend == "torch" else ["bartelsStewart", "sign"]

    results = []
    for variant in variants:
        algo = algorithmRegistry.get(category="lyapunov", variant=variant, backendName=case.backend, maxIter=80)
        start = time.perf_counter()
        solution = algo.solve(A, None, B)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        x_data = solution.toBackendData()
        rank = solution.left.shape[1] if hasattr(solution, "left") else solution.shape[1]
        results.append(
            {
                "variant": variant,
                "elapsed_ms": round(elapsed_ms, 3),
                "rank": int(rank),
                "relative_residual": relative_residual(A.toBackendData(), x_data, q_data),
                "solution_kind": type(solution).__name__,
            }
        )

    return {
        "case": case.name,
        "family": case.family,
        "problem_size": case.problem_size,
        "backend": case.backend,
        "operator_kind": "dense_matrixOperator",
        "data_regime": "standard_dense_continuous",
        "shape": list(case.A.shape),
        "inputs": int(case.B.shape[1]),
        "auto_variant": auto_variant,
        "results": results,
    }


def flatten_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                    "inputs": case["inputs"],
                    "auto_variant": case["auto_variant"],
                    "variant": result["variant"],
                    "elapsed_ms": result["elapsed_ms"],
                    "rank": result["rank"],
                    "relative_residual": result["relative_residual"],
                    "solution_kind": result["solution_kind"],
                    "success": result["relative_residual"] < 1e-10,
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: dict[str, Any]) -> None:
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _svg_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def write_bar_svg(rows: list[dict[str, Any]], path: Path, title: str, value_key: str, log10: bool = False) -> None:
    labeled_rows = [{**row, "display_label": f"{row['variant']}|{row['problem_size']}"} for row in rows]
    write_bar_chart_svg(labeled_rows, path, title, value_key, "summary bars", "display_label", log10=log10, color="#6f5b3e")


def write_markdown(payload: dict[str, Any]) -> None:
    lines = []
    for case in payload["cases"]:
        auto = next(item for item in case["results"] if item["variant"] == case["auto_variant"])
        best = min(case["results"], key=lambda item: item["elapsed_ms"])
        lines.append(
            f"- `{case['case']}`: auto -> `{case['auto_variant']}`, "
            f"auto 残差 -> {auto['relative_residual']:.3e}, "
            f"观测最快 -> `{best['variant']}` ({best['elapsed_ms']} ms)"
        )
    MD_PATH.write_text(
        f"""# Dense Lyapunov 报告

生成时间：`{payload['generated_at']}`

环境：
- Python: `{payload['environment']['python']}`
- Platform: `{payload['environment']['platform']}`
- NumPy: `{payload['environment']['numpy']}`
- Torch available: `{payload['environment']['torch_available']}`

## 范围

本文比较标准 dense 案例上的连续 Lyapunov dense 路径。

## 案例摘要

{chr(10).join(lines)}

## 可视化

- `{RESIDUAL_SVG_PATH.name}`
- `{RUNTIME_SVG_PATH.name}`
- `{DETAILED_RESIDUAL_SVG_PATH.name}`
- `{DETAILED_RUNTIME_SVG_PATH.name}`
- `{SUCCESS_SVG_PATH.name}`
- `{DETAILED_RESIDUAL_SVG_PATH.name}`
- `{DETAILED_RUNTIME_SVG_PATH.name}`
- `{SUCCESS_SVG_PATH.name}`

## 说明

- `scipy` 在 dense continuous 案例上仍默认使用 `bartelsStewart`，因为它是更稳健的直接基线。
- `torch` 默认使用 `sign`，因为 dense direct generalized scipy baseline 不是 torch 路径。
- `10000` 规模的 dense torch 案例在 manifest 中记录为 planned-but-skipped，因为常规报告运行成本过高。
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
            "backend": case.backend,
            "inputs": int(case.B.shape[1]),
            "status": "executed",
        }
        for case in build_cases()
    ]
    manifest_cases.extend(
        [
            {
                "case": "torch_cpu_diagonal_10000",
                "family": "torch_cpu_diagonal",
                "problem_size": 10000,
                "backend": "torch",
                "inputs": 1,
                "status": "skipped_dense_cost",
            },
            {
                "case": "torch_cuda_diagonal_10000",
                "family": "torch_cuda_diagonal",
                "problem_size": 10000,
                "backend": "torch",
                "inputs": 1,
                "status": "skipped_dense_cost",
            },
        ]
    )
    rows = flatten_rows(cases)
    payload = {
        "phase": "dense_lyapunov",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch_available": True,
            "torch_version": torch.__version__,
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
    manifest_json, manifest_csv = write_case_manifest(REPORT_DIR, "dense_lyapunov", manifest_cases)
    write_bar_svg(rows, RESIDUAL_SVG_PATH, "Dense Lyapunov Residuals", "relative_residual", log10=True)
    write_bar_svg(rows, RUNTIME_SVG_PATH, "Dense Lyapunov Runtime", "elapsed_ms")
    scipy_rows = [row for row in rows if row["backend"] == "scipy"]
    series_runtime = []
    series_residual = []
    for variant in ("bartelsStewart", "sign"):
        variant_rows = sorted([row for row in scipy_rows if row["variant"] == variant], key=lambda item: item["problem_size"])
        series_runtime.append(
            {
                "name": variant,
                "points": [{"x": row["problem_size"], "y": row["elapsed_ms"], "label": str(row["problem_size"])} for row in variant_rows],
            }
        )
        series_residual.append(
            {
                "name": variant,
                "points": [{"x": row["problem_size"], "y": row["relative_residual"], "label": str(row["problem_size"])} for row in variant_rows],
            }
        )
    write_line_chart_svg(series_runtime, DETAILED_RUNTIME_SVG_PATH, "Dense Lyapunov Detailed Runtime Sweep", "problem size", "elapsed ms")
    write_line_chart_svg(series_residual, DETAILED_RESIDUAL_SVG_PATH, "Dense Lyapunov Detailed Residual Sweep", "problem size", "relative residual", log10=True)
    success_cases = [row["case"] for row in scipy_rows if row["variant"] == "bartelsStewart"]
    success_map = {(row["variant"], row["case"]): bool(row["success"]) for row in rows}
    write_success_matrix_svg(sorted({row["variant"] for row in rows}), success_cases, success_map, SUCCESS_SVG_PATH, "Dense Lyapunov Success Matrix")
    write_markdown(payload)
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {MD_PATH}")
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
