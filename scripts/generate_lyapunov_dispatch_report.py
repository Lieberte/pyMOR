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
from scipy.sparse import diags

import mor  # noqa: F401
from mor.algorithm.auto.lyapunov import recommendLyapunovOptions, selectLyapunovAlgorithm
from mor.algorithm.registry import algorithmRegistry
from mor.operators import matrixOperator
from report_run_utils import reset_report_dir, write_case_manifest
from report_svg_utils import write_bar_chart_svg, write_success_matrix_svg
from standard_benchmark_cases import STANDARD_BENCHMARK_SOURCES, matlab_gallery_grcar, matrix_depot_poisson_1d

import torch


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "lyapunov"
JSON_PATH = REPORT_DIR / "lyapunov_dispatch_results.json"
CSV_PATH = REPORT_DIR / "lyapunov_dispatch_results.csv"
MD_PATH = REPORT_DIR / "LYAPUNOV_DISPATCH_ANALYSIS.md"
RUNTIME_SVG_PATH = REPORT_DIR / "lyapunov_dispatch_runtime.svg"
RESIDUAL_SVG_PATH = REPORT_DIR / "lyapunov_dispatch_residual.svg"
SUCCESS_SVG_PATH = REPORT_DIR / "lyapunov_dispatch_success_matrix.svg"


@dataclass
class DispatchCase:
    name: str
    family: str
    backend: str
    is_continuous: bool
    A: Any
    B: Any
    expected_variant: str | None
    E: Any | None = None
    should_raise: bool = False
    max_iter: int = 120


def stable_dense_grcar(n: int) -> np.ndarray:
    G = matlab_gallery_grcar(n)
    eig_radius = max(abs(np.linalg.eigvals(G)))
    return G / (2.0 * eig_radius) - 1.5 * np.eye(n)


def build_cases() -> list[DispatchCase]:
    cases = [
        DispatchCase(
            name="dense_continuous_scipy_grcar_n8",
            family="dense_continuous",
            backend="scipy",
            is_continuous=True,
            A=stable_dense_grcar(8),
            B=np.ones((8, 1)),
            expected_variant="bartelsStewart",
        ),
        DispatchCase(
            name="dense_continuous_scipy_descriptor_diagonal_n8",
            family="dense_continuous_descriptor",
            backend="scipy",
            is_continuous=True,
            A=-np.diag(np.linspace(1.0, 8.0, 8)),
            E=np.diag(np.linspace(1.0, 2.0, 8)),
            B=np.ones((8, 1)),
            expected_variant="bartelsStewart",
        ),
        DispatchCase(
            name="dense_continuous_scipy_ill_conditioned_diagonal_n20",
            family="dense_continuous_ill_conditioned",
            backend="scipy",
            is_continuous=True,
            A=-np.diag(np.logspace(0.0, -6.0, 20)) - 0.1 * np.eye(20),
            B=np.ones((20, 1)),
            expected_variant="bartelsStewart",
        ),
        DispatchCase(
            name="sparse_continuous_scipy_poisson_n100",
            family="sparse_continuous",
            backend="scipy",
            is_continuous=True,
            A=-matrix_depot_poisson_1d(100),
            B=np.ones((100, 1)),
            expected_variant="lradi",
        ),
        DispatchCase(
            name="sparse_continuous_scipy_descriptor_poisson_n50",
            family="sparse_continuous_descriptor",
            backend="scipy",
            is_continuous=True,
            A=-matrix_depot_poisson_1d(50),
            E=matrix_depot_poisson_1d(50),
            B=np.ones((50, 1)),
            expected_variant="bartelsStewart",
        ),
        DispatchCase(
            name="sparse_continuous_scipy_near_unstable_poisson_n50",
            family="sparse_continuous_near_unstable",
            backend="scipy",
            is_continuous=True,
            A=-1e-3 * matrix_depot_poisson_1d(50),
            B=np.ones((50, 1)),
            expected_variant="bartelsStewart",
        ),
        DispatchCase(
            name="sparse_continuous_scipy_multi_input_poisson_n50",
            family="sparse_continuous_multi_input",
            backend="scipy",
            is_continuous=True,
            A=-matrix_depot_poisson_1d(50),
            B=np.eye(50, 3),
            expected_variant="bartelsStewart",
        ),
        DispatchCase(
            name="sparse_discrete_scipy_diagonal_n40",
            family="sparse_discrete",
            backend="scipy",
            is_continuous=False,
            A=diags(np.linspace(0.1, 0.6, 40), format="csr"),
            B=np.ones((40, 1)),
            expected_variant="lrsmith",
        ),
        DispatchCase(
            name="dense_discrete_scipy_unsupported_n8",
            family="dense_discrete_unsupported",
            backend="scipy",
            is_continuous=False,
            A=np.eye(8) * 0.2,
            B=np.ones((8, 1)),
            expected_variant=None,
            should_raise=True,
        ),
    ]
    cases.append(
        DispatchCase(
            name="dense_continuous_torch_cpu_diagonal_n20",
            family="dense_continuous_torch",
            backend="torch",
            is_continuous=True,
            A=-np.eye(20),
            B=np.ones((20, 1)),
            expected_variant="sign",
            max_iter=40,
        )
    )
    cases.append(
        DispatchCase(
            name="dense_discrete_torch_cpu_unsupported_n8",
            family="dense_discrete_unsupported",
            backend="torch",
            is_continuous=False,
            A=np.eye(8) * 0.2,
            B=np.ones((8, 1)),
            expected_variant=None,
            should_raise=True,
        )
    )
    if torch.cuda.is_available():
        cases.append(
            DispatchCase(
                name="dense_continuous_torch_cuda_diagonal_n20",
                family="dense_continuous_torch",
                backend="torch",
                is_continuous=True,
                A=-np.eye(20),
                B=np.ones((20, 1)),
                expected_variant="sign",
                max_iter=40,
            )
        )
        cases.append(
            DispatchCase(
                name="dense_discrete_torch_cuda_unsupported_n8",
                family="dense_discrete_unsupported",
                backend="torch",
                is_continuous=False,
                A=np.eye(8) * 0.2,
                B=np.ones((8, 1)),
                expected_variant=None,
                should_raise=True,
            )
        )
    return cases


def to_numpy(data: Any) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if hasattr(data, "toarray"):
        return data.toarray()
    return np.asarray(data)


def continuous_residual(a_data: Any, x_data: Any, q_data: Any, e_data: Any | None = None) -> float:
    a_np, x_np, q_np = to_numpy(a_data), to_numpy(x_data), to_numpy(q_data)
    if e_data is None:
        residual = a_np @ x_np + x_np @ a_np.T + q_np
    else:
        e_np = to_numpy(e_data)
        residual = a_np @ x_np @ e_np.T + e_np @ x_np @ a_np.T + q_np
    return float(np.linalg.norm(residual) / max(np.linalg.norm(q_np), 1e-16))


def discrete_residual(a_data: Any, x_data: Any, q_data: Any) -> float:
    a_np, x_np, q_np = to_numpy(a_data), to_numpy(x_data), to_numpy(q_data)
    residual = a_np @ x_np @ a_np.T - x_np + q_np
    return float(np.linalg.norm(residual) / max(np.linalg.norm(q_np), 1e-16))


def make_operator_data(case: DispatchCase) -> tuple[Any, Any]:
    if case.backend == "torch":
        device = "cuda" if "cuda" in case.name else "cpu"
        return (
            torch.tensor(case.A, dtype=torch.float64, device=device),
            torch.tensor(case.B, dtype=torch.float64, device=device),
        )
    return case.A, case.B


def make_descriptor_data(case: DispatchCase) -> Any | None:
    if case.E is None:
        return None
    if case.backend == "torch":
        device = "cuda" if "cuda" in case.name else "cpu"
        return torch.tensor(case.E, dtype=torch.float64, device=device)
    return case.E


def evaluate_case(case: DispatchCase) -> dict[str, Any]:
    a_data, b_data = make_operator_data(case)
    e_data = make_descriptor_data(case)
    A = matrixOperator(a_data, backendName=case.backend)
    B = matrixOperator(b_data, backendName=case.backend)
    E = matrixOperator(e_data, backendName=case.backend) if e_data is not None else None
    try:
        auto_variant = selectLyapunovAlgorithm(
            variant="auto",
            A=A,
            E=E,
            B=B,
            backendName=case.backend,
            isContinuous=case.is_continuous,
        )
    except Exception as exc:
        return {
            "case": case.name,
            "family": case.family,
            "backend": case.backend,
            "is_continuous": case.is_continuous,
            "operator_kind": A.operatorKind,
            "structure_tags": sorted(A.structureTags),
            "descriptor_kind": E.operatorKind if E is not None else "",
            "descriptor_structure_tags": sorted(E.structureTags) if E is not None else [],
            "shape": list(A.shape),
            "inputs": B.shape[1],
            "auto_variant": "",
            "expected_variant": case.expected_variant,
            "elapsed_ms": None,
            "relative_residual": None,
            "rank": None,
            "iterations": None,
            "dispatch_success": case.should_raise,
            "algorithm_success": case.should_raise,
            "success": case.should_raise,
            "recommendation": "expected_unsupported" if case.should_raise else "fix_dispatch_exception",
            "error": f"{type(exc).__name__}: {exc}",
        }

    options = recommendLyapunovOptions(
        variant="auto",
        A=A,
        E=E,
        B=B,
        backendName=case.backend,
        isContinuous=case.is_continuous,
    )
    algo_kwargs = {"maxIter": case.max_iter}
    algo_kwargs.update(options)
    algo = algorithmRegistry.get(category="lyapunov", variant=auto_variant, backendName=case.backend, **algo_kwargs)
    start = time.perf_counter()
    solution = algo.solve(A, E, B)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    x_data = solution.toBackendData()
    q_data = B.toBackendData() @ B.toBackendData().T
    if case.is_continuous:
        residual = (
            continuous_residual(A.toBackendData(), x_data, q_data)
            if E is None
            else continuous_residual(A.toBackendData(), x_data, q_data, E.toBackendData())
        )
    else:
        residual = discrete_residual(A.toBackendData(), x_data, q_data)
    rank = int(solution.left.shape[1]) if hasattr(solution, "left") else int(solution.shape[1])
    iterations = getattr(algo, "lastIterations", 1)
    dispatch_success = auto_variant == case.expected_variant
    algorithm_success = residual < 1e-8
    success = dispatch_success
    recommendation = "keep"
    if not dispatch_success:
        recommendation = "fix_dispatch_variant"
    elif not algorithm_success:
        recommendation = "review_algorithm_strategy"
    return {
        "case": case.name,
        "family": case.family,
        "backend": case.backend,
        "is_continuous": case.is_continuous,
        "operator_kind": A.operatorKind,
        "structure_tags": sorted(A.structureTags),
        "descriptor_kind": E.operatorKind if E is not None else "",
        "descriptor_structure_tags": sorted(E.structureTags) if E is not None else [],
        "shape": list(A.shape),
        "inputs": B.shape[1],
        "auto_variant": auto_variant,
        "expected_variant": case.expected_variant,
        "elapsed_ms": round(elapsed_ms, 3),
        "relative_residual": residual,
        "rank": rank,
        "iterations": int(iterations),
        "dispatch_success": dispatch_success,
        "algorithm_success": algorithm_success,
        "success": success,
        "recommendation": recommendation,
        "error": "",
    }


def write_csv(rows: list[dict[str, Any]]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_visualizations(rows: list[dict[str, Any]]) -> None:
    executed = [row for row in rows if row["elapsed_ms"] is not None]
    labeled = [{**row, "display_label": f"{row['auto_variant']}|{row['backend']}|{row['shape'][0]}"} for row in executed]
    write_bar_chart_svg(labeled, RUNTIME_SVG_PATH, "Lyapunov Dispatch Runtime", "elapsed_ms", "Auto-selected algorithm runtime", "display_label", color="#4f6f52")
    write_bar_chart_svg(labeled, RESIDUAL_SVG_PATH, "Lyapunov Dispatch Residual", "relative_residual", "Auto-selected algorithm residual", "display_label", log10=True, color="#256d7b")
    cases = [row["case"] for row in rows]
    success_map = {(row["auto_variant"] or "unsupported", row["case"]): bool(row["success"]) for row in rows}
    write_success_matrix_svg(sorted({row["auto_variant"] or "unsupported" for row in rows}), cases, success_map, SUCCESS_SVG_PATH, "Lyapunov Dispatch Success Matrix")


def write_markdown(payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = []
    for row in rows:
        residual = f"{row['relative_residual']:.3e}" if row["relative_residual"] is not None else "n/a"
        lines.append(
            f"- `{row['case']}`: auto -> `{row['auto_variant'] or 'unsupported'}`, "
            f"预期 -> `{row['expected_variant'] or 'unsupported'}`, "
            f"残差 -> {residual}, 分发正确 -> `{row['dispatch_success']}`, "
            f"算法成功 -> `{row['algorithm_success']}`, 建议 -> `{row['recommendation']}`"
        )
    MD_PATH.write_text(
        f"""# Lyapunov 自动分发分析

生成时间：`{payload['generated_at']}`

环境：
- Python: `{payload['environment']['python']}`
- Platform: `{payload['environment']['platform']}`
- NumPy: `{payload['environment']['numpy']}`
- Torch available: `{payload['environment']['torch_available']}`

## 范围

本文验证 Lyapunov 顶层 `auto` 分发层，作为算法专项调优前的入口检查：
- dense continuous scipy -> `bartelsStewart`
- dense continuous torch -> `sign`
- small or moderately dense sparse continuous scipy -> `bartelsStewart`
- larger low-density sparse continuous scipy -> `lradi`
- sparse discrete scipy -> `lrsmith`
- dense discrete -> 设计上不支持

## 案例摘要

{chr(10).join(lines)}

## 来源

- MATLAB gallery: {STANDARD_BENCHMARK_SOURCES['matlab_gallery']['url']}
- Julia MatrixDepot: {STANDARD_BENCHMARK_SOURCES['julia_matrix_depot']['url']}

## 产物

- `{CSV_PATH.name}`
- `{RUNTIME_SVG_PATH.name}`
- `{RESIDUAL_SVG_PATH.name}`
- `{SUCCESS_SVG_PATH.name}`
""",
        encoding="utf-8-sig",
    )


def main() -> int:
    reset_report_dir(REPORT_DIR)
    rows = [evaluate_case(case) for case in build_cases()]
    payload = {
        "phase": "lyapunov_dispatch",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch_available": True,
            "torch_version": torch.__version__,
        },
        "cases": rows,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(rows)
    write_case_manifest(
        REPORT_DIR,
        "lyapunov_dispatch",
        [
            {
                "case": row["case"],
                "family": row["family"],
                "backend": row["backend"],
                "expected_variant": row["expected_variant"],
                "source_url": STANDARD_BENCHMARK_SOURCES["julia_matrix_depot"]["url"]
                if "poisson" in row["case"]
                else STANDARD_BENCHMARK_SOURCES["matlab_gallery"]["url"],
            }
            for row in rows
        ],
    )
    write_visualizations(rows)
    write_markdown(payload, rows)
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {MD_PATH}")
    print(f"Wrote {RUNTIME_SVG_PATH}")
    print(f"Wrote {RESIDUAL_SVG_PATH}")
    print(f"Wrote {SUCCESS_SVG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
