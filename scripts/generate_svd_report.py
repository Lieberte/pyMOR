from __future__ import annotations

import csv
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import mor  # noqa: F401
from mor.algorithm.auto.svd import selectSVDVariant
from mor.algorithm.registry import algorithmRegistry
from mor.operators import matrixOperator
from report_svg_utils import write_bar_chart_svg, write_line_chart_svg, write_success_matrix_svg
from report_run_utils import reset_report_dir, write_case_manifest
from standard_benchmark_cases import (
    STANDARD_BENCHMARK_SOURCES,
    extreme_scaled_diagonal,
    geometric_spectrum_matrix,
    ill_conditioned_diagonal,
    matlab_gallery_grcar,
    matlab_gallery_grcar_rectangular,
    matlab_gallery_hilbert,
    matrix_depot_poisson_2d,
    nearly_collinear_snapshots,
    rank_deficient_tall_skinny,
    singular_diagonal,
)

import torch


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "svd"
JSON_PATH = REPORT_DIR / "svd_results.json"
CSV_PATH = REPORT_DIR / "svd_results.csv"
MD_PATH = REPORT_DIR / "SVD_REPORT.md"
RESIDUAL_SVG_PATH = REPORT_DIR / "svd_auto_residuals.svg"
RUNTIME_SVG_PATH = REPORT_DIR / "svd_auto_runtime.svg"
DETAILED_RUNTIME_SVG_PATH = REPORT_DIR / "svd_detailed_runtime.svg"
DETAILED_RESIDUAL_SVG_PATH = REPORT_DIR / "svd_detailed_residual.svg"
SUCCESS_SVG_PATH = REPORT_DIR / "svd_success_matrix.svg"
DISPATCH_CSV_PATH = REPORT_DIR / "svd_dispatch_analysis.csv"
DISPATCH_MD_PATH = REPORT_DIR / "SVD_DISPATCH_ANALYSIS.md"
TIMING_REPEATS = 5
MONITOR_RUNTIME_RATIO = 1.25
REVIEW_RUNTIME_RATIO = 2.0
MATERIAL_RUNTIME_DELTA_MS = 0.5


def to_numpy(data: Any) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def reconstruction_error(U: Any, S: Any, Vt: Any, target: np.ndarray) -> float:
    u_np = to_numpy(U)
    s_np = to_numpy(S)
    vt_np = to_numpy(Vt)
    approx = u_np @ np.diag(s_np) @ vt_np
    return float(np.linalg.norm(approx - target))


def relative_reconstruction_error(U: Any, S: Any, Vt: Any, target: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(target)), 1e-16)
    return reconstruction_error(U, S, Vt, target) / denom


def top_singular_values(target: np.ndarray, rank: int) -> np.ndarray:
    return np.linalg.svd(target, compute_uv=False)[:rank]


def singular_value_error(singular_values: np.ndarray, target: np.ndarray) -> float:
    ref = top_singular_values(target, len(singular_values))
    if len(singular_values) != len(ref):
        return float("inf")
    return float(np.linalg.norm(singular_values - ref))


@dataclass
class StaticCase:
    name: str
    family: str
    problem_size: int
    backend: str
    operator: Any
    target: np.ndarray
    rank: int
    center_snapshots: bool = False


@dataclass
class IncrementalCase:
    name: str
    family: str
    problem_size: int
    backend: str
    batches: list[np.ndarray]
    rank: int


def build_sparse_cases() -> list[StaticCase]:
    static_cases = []
    for transpose in (False, True):
        sparse = matrix_depot_poisson_2d(4).tocsr()[:, :8]
        if transpose:
            sparse = sparse.T.tocsr()
        for rank in (3, 8):
            static_cases.append(StaticCase(
                name=f"matrix_depot_sparse_{'wide' if transpose else 'tall'}_rank{rank}",
                family='sparse_poisson', problem_size=16, backend='scipy',
                operator=matrixOperator(sparse, backendName='scipy'),
                target=sparse.toarray(), rank=rank,
            ))
    return static_cases


def build_cases() -> tuple[list[StaticCase], list[IncrementalCase]]:
    rng = np.random.default_rng(7)
    static_cases = build_sparse_cases()
    for n, cols in ((10, 5), (100, 10), (1000, 20), (10000, 40)):
        tall = matlab_gallery_grcar_rectangular(n, cols)
        static_cases.append(
            StaticCase(
                name=f"matlab_gallery_grcar_tall_skinny_n{n}",
                family="grcar_tall_skinny",
                problem_size=n,
                backend="scipy",
                operator=matrixOperator(tall, backendName="scipy"),
                target=tall,
                rank=cols,
            )
        )
    for n in (10, 100, 1000):
        wide = matlab_gallery_grcar_rectangular(8, n)
        static_cases.append(
            StaticCase(
                name=f"matlab_gallery_grcar_wide_n{n}",
                family="grcar_wide",
                problem_size=n,
                backend="scipy",
                operator=matrixOperator(wide, backendName="scipy"),
                target=wide,
                rank=8,
            )
        )
    for n in (10, 100, 1000, 10000):
        centered = matlab_gallery_grcar_rectangular(n, 6)
        static_cases.append(
            StaticCase(
                name=f"matlab_gallery_grcar_centered_n{n}",
                family="grcar_centered",
                problem_size=n,
                backend="scipy",
                operator=matrixOperator(centered, backendName="scipy"),
                target=centered - centered.mean(axis=1, keepdims=True),
                rank=6,
                center_snapshots=True,
            )
        )
    for n in (10, 12, 16):
        square = matlab_gallery_hilbert(n)
        static_cases.append(
            StaticCase(
                name=f"matlab_gallery_hilbert_square_n{n}",
                family="hilbert_square",
                problem_size=n,
                backend="scipy",
                operator=matrixOperator(square, backendName="scipy"),
                target=square,
                rank=n,
            )
        )
    for n, min_exp in ((10, -12.0), (30, -12.0), (100, -12.0), (300, -12.0), (100, -24.0), (300, -24.0)):
        conditioned = ill_conditioned_diagonal(n, min_exp=min_exp)
        static_cases.append(
            StaticCase(
                name=f"pathological_ill_conditioned_diagonal_n{n}_kappa1e{abs(int(min_exp))}",
                family="ill_conditioned_diagonal",
                problem_size=n,
                backend="scipy",
                operator=matrixOperator(conditioned, backendName="scipy"),
                target=conditioned,
                rank=n,
            )
        )
    for n, zero_fraction in ((10, 0.1), (30, 0.1), (100, 0.1), (300, 0.1), (100, 0.5), (300, 0.5)):
        zero_columns = max(1, int(n * zero_fraction))
        singular = singular_diagonal(n, zero_columns=zero_columns)
        static_cases.append(
            StaticCase(
                name=f"pathological_singular_diagonal_n{n}_zero{zero_columns}",
                family="singular_diagonal",
                problem_size=n,
                backend="scipy",
                operator=matrixOperator(singular, backendName="scipy"),
                target=singular,
                rank=n,
            )
        )
    for n, cols, min_exp in ((100, 20, -16.0), (10000, 40, -16.0), (10000, 40, -24.0)):
        geometric = geometric_spectrum_matrix(n, cols, min_exp=min_exp, seed=11)
        static_cases.append(
            StaticCase(
                name=f"pathological_geometric_spectrum_n{n}_cols{cols}_kappa1e{abs(int(min_exp))}",
                family="geometric_spectrum",
                problem_size=n,
                backend="scipy",
                operator=matrixOperator(geometric, backendName="scipy"),
                target=geometric,
                rank=cols,
            )
        )
    for n, cols, effective_rank in ((1000, 40, 5), (10000, 80, 8)):
        deficient = rank_deficient_tall_skinny(n, cols, effective_rank=effective_rank, seed=17)
        static_cases.append(
            StaticCase(
                name=f"pathological_rank_deficient_tall_n{n}_cols{cols}_rank{effective_rank}",
                family="rank_deficient_tall_skinny",
                problem_size=n,
                backend="scipy",
                operator=matrixOperator(deficient, backendName="scipy"),
                target=deficient,
                rank=cols,
            )
        )
    for n, cols, noise_scale in ((1000, 20, 1e-12), (10000, 40, 1e-14)):
        collinear = nearly_collinear_snapshots(n, cols, noise_scale=noise_scale, seed=23)
        static_cases.append(
            StaticCase(
                name=f"pathological_nearly_collinear_n{n}_cols{cols}_noise{noise_scale:.0e}",
                family="nearly_collinear_snapshots",
                problem_size=n,
                backend="scipy",
                operator=matrixOperator(collinear, backendName="scipy"),
                target=collinear,
                rank=cols,
            )
        )
    for n in (20, 40, 100, 300):
        scaled = extreme_scaled_diagonal(n)
        static_cases.append(
            StaticCase(
                name=f"pathological_extreme_scaled_diagonal_n{n}",
                family="extreme_scaled_diagonal",
                problem_size=n,
                backend="scipy",
                operator=matrixOperator(scaled, backendName="scipy"),
                target=scaled,
                rank=n,
            )
        )
    zero = np.zeros((1000, 40))
    static_cases.append(
        StaticCase(
            name="pathological_zero_tall_skinny_n1000_cols40",
            family="zero_tall_skinny",
            problem_size=1000,
            backend="scipy",
            operator=matrixOperator(zero, backendName="scipy"),
            target=zero,
            rank=40,
        )
    )
    for m in (3, 7, 10, 14):
        c1 = matrix_depot_poisson_2d(m).toarray()
        c2 = matlab_gallery_grcar(m * m)
        static_cases.append(
            StaticCase(
                name=f"matrix_depot_poisson_plus_grcar_n{m*m}",
                family="composed_poisson_grcar",
                problem_size=m * m,
                backend="scipy",
                operator=matrixOperator(c1, backendName="scipy") + matrixOperator(c2, backendName="scipy"),
                target=c1 + c2,
                rank=m * m,
            )
        )
    for n, cols, device_name in ((100, 10, "cpu"), (300, 20, "cpu"), (1000, 20, "cpu"), (3000, 40, "cpu"), (10000, 40, "cpu")):
        torch_data = torch.randn((n, cols), dtype=torch.float64, device=device_name)
        static_cases.append(
            StaticCase(
                name=f"torch_{device_name}_dense_n{n}",
                family=f"torch_{device_name}_dense",
                problem_size=n,
                backend="torch",
                operator=matrixOperator(torch_data, backendName="torch"),
                target=to_numpy(torch_data),
                rank=cols,
            )
        )
    if torch.cuda.is_available():
        for n, cols in ((100, 10), (300, 20), (1000, 20), (3000, 40), (10000, 40)):
            torch_data = torch.randn((n, cols), dtype=torch.float64, device="cuda")
            static_cases.append(
                StaticCase(
                    name=f"torch_cuda_dense_n{n}",
                    family="torch_cuda_dense",
                    problem_size=n,
                    backend="torch",
                    operator=matrixOperator(torch_data, backendName="torch"),
                    target=to_numpy(torch_data),
                    rank=cols,
                )
            )

    incremental_cases = []
    for n, rank in ((6, 3), (20, 5), (50, 8)):
        first = matlab_gallery_grcar(n)[:, :rank]
        incremental_cases.append(
            IncrementalCase(
                name=f"matlab_gallery_grcar_incremental_exact_subspace_n{n}_rank{rank}",
                family="incremental_exact_subspace",
                problem_size=n,
                backend="scipy",
                batches=[
                    first,
                    first @ rng.standard_normal((rank, max(2, rank // 2))),
                ],
                rank=rank,
            )
        )
    for n, rank in ((6, 5), (20, 10), (50, 15)):
        incremental_cases.append(
            IncrementalCase(
                name=f"matlab_gallery_hilbert_incremental_general_n{n}_rank{rank}",
                family="incremental_general",
                problem_size=n,
                backend="scipy",
                batches=[
                    matlab_gallery_hilbert(n)[:, : max(2, rank // 3)],
                    matlab_gallery_grcar(n)[:, : max(2, rank // 3)],
                    rng.standard_normal((n, rank - 2 * max(2, rank // 3))),
                ],
                rank=rank,
            )
        )
    return static_cases, incremental_cases


def evaluate_static_case(case: StaticCase) -> dict[str, Any]:
    variants = ["static", "economic", "dual", "qrSVD"]
    if case.backend == "torch":
        variants = ["static", "economic", "dual"]
    elif case.operator.isSparse:
        variants = ["static", "economic", "dual"]

    auto_variant = selectSVDVariant(
        backendName=case.backend,
        xOperator=case.operator,
        centerSnapshots=case.center_snapshots,
        rank=case.rank,
    )

    results = []
    referenceSpectrum = np.linalg.svd(case.target, compute_uv=False)
    scale = referenceSpectrum[0] if referenceSpectrum.size else 0.0
    expectedError = (float(np.linalg.norm(referenceSpectrum[case.rank:] / scale) /
                           np.linalg.norm(referenceSpectrum / scale)) if scale > 0 else 0.0)
    for variant in variants:
        try:
            samples = []
            U = S = Vt = None
            for _ in range(TIMING_REPEATS):
                algo = algorithmRegistry.get(category="svd", variant=variant, backendName=case.backend)
                if variant == "static":
                    algo = algorithmRegistry.get(
                        category="svd",
                        variant=variant,
                        backendName=case.backend,
                        centerSnapshots=case.center_snapshots,
                    )
                start = time.perf_counter()
                U, S, Vt = algo.decompose(case.operator, rank=case.rank)
                samples.append((time.perf_counter() - start) * 1000.0)
            assert U is not None and S is not None and Vt is not None
            elapsed_ms = statistics.median(samples)
            elapsed_std_ms = statistics.pstdev(samples) if len(samples) > 1 else 0.0
            s_np = to_numpy(S)
            ref = referenceSpectrum[:len(s_np)]
            results.append(
                {
                    "variant": variant,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "elapsed_std_ms": round(elapsed_std_ms, 3),
                    "elapsed_samples_ms": [round(sample, 3) for sample in samples],
                    "rank": int(len(s_np)),
                    "singular_values": s_np.tolist(),
                    "reference_singular_values": ref.tolist(),
                    "singular_value_error_l2": float(np.linalg.norm(s_np - ref)),
                    "reconstruction_error": reconstruction_error(U, S, Vt, case.target),
                    "relative_reconstruction_error": relative_reconstruction_error(U, S, Vt, case.target),
                    "error": "",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "variant": variant,
                    "elapsed_ms": float("inf"),
                    "elapsed_std_ms": float("inf"),
                    "elapsed_samples_ms": [],
                    "rank": 0,
                    "singular_values": [],
                    "reference_singular_values": [],
                    "singular_value_error_l2": float("inf"),
                    "reconstruction_error": float("inf"),
                    "relative_reconstruction_error": float("inf"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return {
        "case": case.name,
        "family": case.family,
        "problem_size": case.problem_size,
        "case_type": "static",
        "backend": case.backend,
        "operator_kind": type(case.operator).__name__,
        "shape": list(case.operator.shape),
        "rank": case.rank,
        "expected_relative_reconstruction_error": expectedError,
        "center_snapshots": case.center_snapshots,
        "auto_variant": auto_variant,
        "results": results,
    }


def evaluate_incremental_case(case: IncrementalCase) -> dict[str, Any]:
    variants = ["incrementalStandard", "incrementalBrand", "incrementalFast"]
    combined = np.hstack(case.batches)
    results = []
    for variant in variants:
        samples = []
        last = None
        for _ in range(TIMING_REPEATS):
            algo = algorithmRegistry.get(category="svd", variant=variant, backendName=case.backend)
            start = time.perf_counter()
            for batch in case.batches:
                last = algo.decompose(matrixOperator(batch, backendName=case.backend), rank=case.rank)
            samples.append((time.perf_counter() - start) * 1000.0)
        assert last is not None
        U, S, Vt = last
        elapsed_ms = statistics.median(samples)
        elapsed_std_ms = statistics.pstdev(samples) if len(samples) > 1 else 0.0
        s_np = to_numpy(S)
        ref = top_singular_values(combined, len(s_np))
        results.append(
            {
                "variant": variant,
                "elapsed_ms": round(elapsed_ms, 3),
                "elapsed_std_ms": round(elapsed_std_ms, 3),
                "elapsed_samples_ms": [round(sample, 3) for sample in samples],
                "rank": int(len(s_np)),
                "singular_values": s_np.tolist(),
                "reference_singular_values": ref.tolist(),
                "singular_value_error_l2": singular_value_error(s_np, combined),
                "reconstruction_error": reconstruction_error(U, S, Vt, combined),
                "relative_reconstruction_error": relative_reconstruction_error(U, S, Vt, combined),
                "error": "",
            }
        )

    return {
        "case": case.name,
        "family": case.family,
        "problem_size": case.problem_size,
        "case_type": "incremental",
        "backend": case.backend,
        "operator_kind": "matrixOperator-batches",
        "shape": [combined.shape[0], combined.shape[1]],
        "rank": case.rank,
        "num_batches": len(case.batches),
        "auto_variant": selectSVDVariant(backendName=case.backend, incremental=True),
        "results": results,
    }


def result_succeeded(case: dict[str, Any], result: dict[str, Any]) -> bool:
    expected = case.get('expected_relative_reconstruction_error', 0.0)
    actual = result.get('relative_reconstruction_error', float('inf'))
    return bool(result.get('error', '') == '' and np.isfinite(actual)
                and abs(actual - expected) <= max(1e-8, expected * 1e-6))


def flatten_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for result in case["results"]:
            rows.append(
                {
                    "case": case["case"],
                    "case_type": case["case_type"],
                    "family": case["family"],
                    "problem_size": case["problem_size"],
                    "backend": case["backend"],
                    "operator_kind": case["operator_kind"],
                    "shape": "x".join(str(x) for x in case["shape"]),
                    "rank": case["rank"],
                    "auto_variant": case["auto_variant"],
                    "variant": result["variant"],
                    "elapsed_ms": result["elapsed_ms"],
                    "elapsed_std_ms": result.get("elapsed_std_ms", 0.0),
                    "singular_value_error_l2": result.get("singular_value_error_l2", float("inf")),
                    "reconstruction_error": result.get("reconstruction_error", float("inf")),
                    "relative_reconstruction_error": result.get("relative_reconstruction_error", float("inf")),
                    "expected_relative_reconstruction_error": case.get('expected_relative_reconstruction_error', 0.0),
                    "success": result_succeeded(case, result),
                    "error": result.get("error", ""),
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_dispatch_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["case"], []).append(row)

    analysis = []
    for case, case_rows in grouped.items():
        successful = [row for row in case_rows if row["success"]]
        auto_row = next((row for row in case_rows if row["variant"] == row["auto_variant"]), None)
        fastest = min(successful, key=lambda row: float(row["elapsed_ms"])) if successful else None
        best_error = min((float(row["relative_reconstruction_error"]) for row in successful), default=float("inf"))
        auto_elapsed = float(auto_row["elapsed_ms"]) if auto_row is not None else float("inf")
        fastest_elapsed = float(fastest["elapsed_ms"]) if fastest is not None else float("inf")
        ratio = auto_elapsed / max(fastest_elapsed, 1e-12)
        runtime_delta = auto_elapsed - fastest_elapsed
        recommendation = "keep"
        rationale = "default_is_successful_and_competitive"
        if auto_row is None or not auto_row["success"]:
            recommendation = "fix_auto_correctness"
            rationale = "auto_variant_failed_or_is_missing"
        elif ratio > REVIEW_RUNTIME_RATIO and runtime_delta >= MATERIAL_RUNTIME_DELTA_MS:
            recommendation = "review_dispatch_speed"
            rationale = "material_absolute_and_relative_slowdown"
        elif ratio > MONITOR_RUNTIME_RATIO:
            recommendation = "monitor_dispatch_speed"
            rationale = (
                "ratio_only_submillisecond_difference"
                if runtime_delta < MATERIAL_RUNTIME_DELTA_MS
                else "moderate_relative_slowdown"
            )
        if (
            case_rows[0]["family"] == "geometric_spectrum"
            and case_rows[0]["auto_variant"] == "qrSVD"
        ):
            rationale += ";qrSVD_preserves_stability_on_ill_conditioned_tall_skinny_inputs"
        analysis.append(
            {
                "case": case,
                "short_label": short_case_label(case_rows[0]),
                "family": case_rows[0]["family"],
                "backend": case_rows[0]["backend"],
                "shape": case_rows[0]["shape"],
                "rank": case_rows[0]["rank"],
                "auto_variant": case_rows[0]["auto_variant"],
                "fastest_successful_variant": fastest["variant"] if fastest is not None else "",
                "auto_elapsed_ms": round(auto_elapsed, 6) if np.isfinite(auto_elapsed) else "inf",
                "fastest_elapsed_ms": round(fastest_elapsed, 6) if np.isfinite(fastest_elapsed) else "inf",
                "auto_elapsed_std_ms": auto_row.get("elapsed_std_ms", "") if auto_row is not None else "",
                "fastest_elapsed_std_ms": fastest.get("elapsed_std_ms", "") if fastest is not None else "",
                "auto_runtime_ratio": round(ratio, 6) if np.isfinite(ratio) else "inf",
                "auto_runtime_delta_ms": round(runtime_delta, 6) if np.isfinite(runtime_delta) else "inf",
                "auto_success": bool(auto_row and auto_row["success"]),
                "best_successful_relative_error": best_error,
                "recommendation": recommendation,
                "rationale": rationale,
            }
        )
    return sorted(analysis, key=lambda row: (row["family"], int(row["rank"]), row["case"]))


def write_dispatch_analysis(analysis: list[dict[str, Any]]) -> None:
    if not analysis:
        return
    with DISPATCH_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(analysis[0].keys()))
        writer.writeheader()
        writer.writerows(analysis)

    family_groups: dict[str, list[dict[str, Any]]] = {}
    for row in analysis:
        family_groups.setdefault(row["family"], []).append(row)

    family_lines = []
    for family, family_rows in sorted(family_groups.items()):
        ratios = [float(row["auto_runtime_ratio"]) for row in family_rows if row["auto_runtime_ratio"] != "inf"]
        max_ratio = max(ratios, default=float("nan"))
        mean_ratio = sum(ratios) / len(ratios) if ratios else float("nan")
        reviews = sum(row["recommendation"] != "keep" for row in family_rows)
        fastest_counts: dict[str, int] = {}
        for row in family_rows:
            fastest_counts[row["fastest_successful_variant"]] = fastest_counts.get(row["fastest_successful_variant"], 0) + 1
        fastest_summary = ", ".join(f"{variant}:{count}" for variant, count in sorted(fastest_counts.items()))
        family_lines.append(
            f"- `{family}`: 案例数={len(family_rows)}, 需复核={reviews}, "
            f"auto/fastest 平均比值={mean_ratio:.2f}, 最大比值={max_ratio:.2f}, 最快变体分布={fastest_summary}"
        )

    review_rows = [row for row in analysis if row["recommendation"] != "keep"]
    review_lines = [
        f"- `{row['case']}`: auto 为 `{row['auto_variant']}`，最快成功变体为 `{row['fastest_successful_variant']}`, "
        f"耗时比={row['auto_runtime_ratio']}, 建议=`{row['recommendation']}`"
        f", delta_ms={row['auto_runtime_delta_ms']}, rationale=`{row['rationale']}`"
        for row in sorted(review_rows, key=lambda item: float(item["auto_runtime_ratio"]), reverse=True)
    ]

    DISPATCH_MD_PATH.write_text(
        "\n".join(
            [
                "# SVD 自动分发分析",
                "",
                "本文汇总 SVD `auto` 分发与每个案例最快成功变体之间的对比。",
                f"耗时为本地运行 {TIMING_REPEATS} 次的中位数，只作为分发策略信号，不作为绝对基准。",
                "",
                "## 案例族汇总",
                "",
                *family_lines,
                "",
                "## 需要复核的案例",
                "",
                *(review_lines or ["- 无"]),
                "",
                f"完整逐案例分发数据见：`{DISPATCH_CSV_PATH.name}`",
                "",
            ]
        ),
        encoding="utf-8-sig",
    )


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


def short_case_label(row: dict[str, Any]) -> str:
    family = row["family"]
    size = int(row["problem_size"])
    size_label = f"{size // 1000}k" if size >= 1000 and size % 1000 == 0 else str(size)
    case = row["case"]
    if family == "grcar_tall_skinny":
        return f"TS{size_label}"
    if family == "grcar_wide":
        return f"W{size_label}"
    if family == "grcar_centered":
        return f"C{size_label}"
    if family == "hilbert_square":
        return f"H{size_label}"
    if family == "ill_conditioned_diagonal":
        kappa = case.rsplit("kappa", 1)[-1].replace("1e", "e") if "kappa" in case else "e12"
        return f"IC{size_label}-{kappa}"
    if family == "singular_diagonal":
        zero = case.rsplit("_zero", 1)[-1] if "_zero" in case else "z"
        return f"SG{size_label}-z{zero}"
    if family == "geometric_spectrum":
        kappa = case.rsplit("kappa", 1)[-1].replace("1e", "e")
        return f"GS{size_label}-{kappa}"
    if family == "rank_deficient_tall_skinny":
        rank = case.rsplit("_rank", 1)[-1]
        return f"RD{size_label}-r{rank}"
    if family == "nearly_collinear_snapshots":
        noise = case.rsplit("_noise", 1)[-1].replace("1e-", "e-")
        return f"NC{size_label}-{noise}"
    if family == "extreme_scaled_diagonal":
        return f"XS{size_label}"
    if family == "zero_tall_skinny":
        return f"Z{size_label}"
    if family == "composed_poisson_grcar":
        return f"PG{size_label}"
    if family == "torch_cpu_dense":
        return f"TC{size_label}"
    if family == "torch_cuda_dense":
        return f"TG{size_label}"
    if family == "incremental_exact_subspace":
        return f"IE{size_label}"
    if family == "incremental_general":
        return f"IG{size_label}"
    return f"{family[:3].upper()}{size_label}"


def case_source_metadata(case_name: str, family: str) -> dict[str, str]:
    if family in {"grcar_tall_skinny", "grcar_wide", "grcar_centered", "incremental_exact_subspace"}:
        return {
            "source_name": "MATLAB gallery",
            "source_case": "grcar",
            "source_url": STANDARD_BENCHMARK_SOURCES["matlab_gallery"]["url"],
        }
    if family in {"hilbert_square", "incremental_general"}:
        return {
            "source_name": "MATLAB gallery",
            "source_case": "hilbert/grcar",
            "source_url": STANDARD_BENCHMARK_SOURCES["matlab_gallery"]["url"],
        }
    if family == "composed_poisson_grcar":
        return {
            "source_name": "MatrixDepot + MATLAB gallery",
            "source_case": "poisson_2d + grcar",
            "source_url": (
                f"{STANDARD_BENCHMARK_SOURCES['julia_matrix_depot']['url']} ; "
                f"{STANDARD_BENCHMARK_SOURCES['matlab_gallery']['url']}"
            ),
        }
    if family.startswith("torch_"):
        return {
            "source_name": "local-generated",
            "source_case": "random dense torch matrix",
            "source_url": "local-generated",
        }
    return {
        "source_name": "local-generated pathological",
        "source_case": family,
        "source_url": STANDARD_BENCHMARK_SOURCES["pathological"]["url"],
    }


def write_bar_svg(
    rows: list[dict[str, Any]],
    path: Path,
    title: str,
    value_key: str,
    value_label: str,
    log10: bool = False,
) -> None:
    auto_rows = [row for row in rows if row["variant"] == row["auto_variant"]]
    labeled_rows = [{**row, "display_label": short_case_label(row)} for row in auto_rows]
    write_bar_chart_svg(labeled_rows, path, title, value_key, value_label, "display_label", log10=log10, color="#4f6f52")


def write_visualizations(rows: list[dict[str, Any]]) -> None:
    write_bar_svg(
        rows,
        RESIDUAL_SVG_PATH,
        "SVD Auto Relative Reconstruction Error",
        "relative_reconstruction_error",
        "Lower is better; bars show each case's auto-selected variant on log10 scale.",
        log10=True,
    )
    write_bar_svg(
        rows,
        RUNTIME_SVG_PATH,
        "SVD Auto Runtime",
        "elapsed_ms",
        f"Median local timings over {TIMING_REPEATS} runs in milliseconds; bars show each case's auto-selected variant.",
    )
    detailed_rows = [
        row for row in rows
        if row["family"] in {
            "grcar_tall_skinny",
            "grcar_wide",
            "grcar_centered",
            "ill_conditioned_diagonal",
            "singular_diagonal",
            "geometric_spectrum",
            "rank_deficient_tall_skinny",
            "nearly_collinear_snapshots",
            "extreme_scaled_diagonal",
            "zero_tall_skinny",
            "sparse_poisson",
        }
    ]
    series_runtime = []
    series_residual = []
    for variant in sorted({row["variant"] for row in detailed_rows if row["backend"] == "scipy"}):
        variant_rows = sorted(
            [row for row in detailed_rows if row["variant"] == variant and row["backend"] == "scipy"],
            key=lambda item: (item["family"], item["problem_size"]),
        )
        series_runtime.append(
            {
                "name": variant,
                "points": [{"x": idx + 1, "y": row["elapsed_ms"], "label": short_case_label(row)} for idx, row in enumerate(variant_rows)],
            }
        )
        series_residual.append(
            {
                "name": variant,
                "points": [{"x": idx + 1, "y": row["relative_reconstruction_error"], "label": short_case_label(row)} for idx, row in enumerate(variant_rows)],
            }
        )
    write_line_chart_svg(series_runtime, DETAILED_RUNTIME_SVG_PATH, "SVD Detailed Runtime Sweep", "case index", "elapsed ms")
    write_line_chart_svg(series_residual, DETAILED_RESIDUAL_SVG_PATH, "SVD Detailed Residual Sweep", "case index", "relative reconstruction error", log10=True)
    success_cases = [
        short_case_label(row)
        for row in sorted(detailed_rows, key=lambda item: (item["family"], item["problem_size"], item["case"]))
        if row["variant"] == "economic"
    ]
    success_map = {}
    for row in detailed_rows:
        success_map[(row["variant"], short_case_label(row))] = bool(row["success"])
    write_success_matrix_svg(
        sorted({row["variant"] for row in detailed_rows}),
        success_cases,
        success_map,
        SUCCESS_SVG_PATH,
        "SVD Success Matrix",
    )


def write_markdown(payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    case_lines = []
    pathological_families = {
        "ill_conditioned_diagonal",
        "singular_diagonal",
        "geometric_spectrum",
        "rank_deficient_tall_skinny",
        "nearly_collinear_snapshots",
        "extreme_scaled_diagonal",
        "zero_tall_skinny",
    }
    pathological_lines = []
    for family in sorted(pathological_families):
        family_cases = [case for case in payload["cases"] if case["family"] == family]
        if not family_cases:
            continue
        sizes = sorted({case["problem_size"] for case in family_cases})
        auto_counts: dict[str, int] = {}
        best_errors = []
        for case in family_cases:
            auto_counts[case["auto_variant"]] = auto_counts.get(case["auto_variant"], 0) + 1
            successful = [
                result for result in case["results"]
                if result_succeeded(case, result)
            ]
            best_errors.append(min((result["relative_reconstruction_error"] for result in successful), default=float("inf")))
        auto_summary = ", ".join(f"{variant}:{count}" for variant, count in sorted(auto_counts.items()))
        pathological_lines.append(
            f"- `{family}`: 案例数={len(family_cases)}, 尺寸={sizes}, auto 分布={auto_summary}, "
            f"最大最佳成功误差={max(best_errors):.3e}"
        )

    dispatch_analysis = build_dispatch_analysis(rows)
    benchmark_lines = []
    for family in sorted({row["family"] for row in dispatch_analysis}):
        family_rows = [row for row in dispatch_analysis if row["family"] == family]
        ratios = [float(row["auto_runtime_ratio"]) for row in family_rows if row["auto_runtime_ratio"] != "inf"]
        reviews = sum(row["recommendation"] != "keep" for row in family_rows)
        fastest_counts: dict[str, int] = {}
        for row in family_rows:
            fastest_counts[row["fastest_successful_variant"]] = fastest_counts.get(row["fastest_successful_variant"], 0) + 1
        fastest_summary = ", ".join(f"{variant}:{count}" for variant, count in sorted(fastest_counts.items()))
        benchmark_lines.append(
            f"- `{family}`: auto/fastest 平均比值={sum(ratios) / len(ratios):.2f}, "
            f"最大比值={max(ratios):.2f}, 需复核={reviews}, 最快变体分布={fastest_summary}"
        )

    for case in payload["cases"]:
        successful = [
            result for result in case["results"]
            if result_succeeded(case, result)
        ]
        best = min(successful or case["results"], key=lambda item: item["elapsed_ms"])
        fastest_label = "最快成功变体" if successful else "最快失败变体"
        case_lines.append(
            f"- `{case['case']}`: auto -> `{case['auto_variant']}`, "
            f"{fastest_label} -> `{best['variant']}` ({best['elapsed_ms']} ms), "
            f"最佳成功相对重构误差 -> "
            f"{min((r['relative_reconstruction_error'] for r in successful), default=float('inf')):.3e}"
        )

    summary = f"""# SVD 报告

生成时间：`{payload['generated_at']}`

环境：
- Python: `{payload['environment']['python']}`
- Platform: `{payload['environment']['platform']}`
- NumPy: `{payload['environment']['numpy']}`
- Torch available: `{payload['environment']['torch_available']}`

## 范围

本文汇总 classical MOR 优化中 SVD 部分的结果：
- SVD 算法正确性
- operator 兼容性
- SVD `auto` 选择行为

## 案例摘要

{chr(10).join(case_lines)}

## 病态矩阵覆盖

{chr(10).join(pathological_lines)}

## Benchmark 对比结论

{chr(10).join(benchmark_lines)}

## 说明

- 耗时为本地运行 `{TIMING_REPEATS}` 次的中位数，只用于分发策略对比。
- `auto` 的判断依据是场景适配性，不要求每一行都选择全局最快变体。
- SVG 图使用短标签以保持可读性，完整标签映射见 `svd_case_manifest.csv`。

## 基准来源

- MATLAB gallery 测试矩阵：{STANDARD_BENCHMARK_SOURCES['matlab_gallery']['url']}
- Julia MatrixDepot 测试矩阵集合：{STANDARD_BENCHMARK_SOURCES['julia_matrix_depot']['url']}
- SuiteSparse Matrix Collection 作为可选外部稀疏案例参考：{STANDARD_BENCHMARK_SOURCES['suitesparse_matrix_collection']['url']}
- 本地构造的病态案例在 manifest 中标记为 `local-generated`。

case manifest 包含 `source_name`、`source_case`、`source_url` 字段，用于逐案例来源追踪。

## 可视化
  - `{RESIDUAL_SVG_PATH.name}`
  - `{RUNTIME_SVG_PATH.name}`
  - `{DETAILED_RESIDUAL_SVG_PATH.name}`
  - `{DETAILED_RUNTIME_SVG_PATH.name}`
  - `{SUCCESS_SVG_PATH.name}`

## 机器可读结果
  - `{JSON_PATH.name}`
  - `{CSV_PATH.name}`

## 分发分析
  - `{DISPATCH_MD_PATH.name}`
  - `{DISPATCH_CSV_PATH.name}`
"""
    with MD_PATH.open("w", encoding="utf-8-sig") as f:
        f.write(summary)


def main() -> int:
    reset_report_dir(REPORT_DIR)
    static_cases, incremental_cases = build_cases()
    evaluated = [evaluate_static_case(case) for case in static_cases]
    evaluated.extend(evaluate_incremental_case(case) for case in incremental_cases)
    manifest_cases = [
        {
            "case": case.name,
            "short_label": short_case_label({"case": case.name, "family": case.family, "problem_size": case.problem_size}),
            **case_source_metadata(case.name, case.family),
            "family": case.family,
            "problem_size": case.problem_size,
            "backend": case.backend,
            "rank": case.rank,
            "center_snapshots": case.center_snapshots if isinstance(case, StaticCase) else False,
        }
        for case in static_cases
    ] + [
        {
            "case": case.name,
            "short_label": short_case_label({"case": case.name, "family": case.family, "problem_size": case.problem_size}),
            **case_source_metadata(case.name, case.family),
            "family": case.family,
            "problem_size": case.problem_size,
            "backend": case.backend,
            "rank": case.rank,
            "center_snapshots": False,
        }
        for case in incremental_cases
    ]
    payload = {
        "phase": "svd",
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
        "cases": evaluated,
    }
    rows = flatten_rows(evaluated)
    dispatch_analysis = build_dispatch_analysis(rows)
    write_json(payload)
    write_csv(rows)
    write_dispatch_analysis(dispatch_analysis)
    manifest_json, manifest_csv = write_case_manifest(REPORT_DIR, "svd", manifest_cases)
    write_visualizations(rows)
    write_markdown(payload, rows)
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {MD_PATH}")
    print(f"Wrote {DISPATCH_MD_PATH}")
    print(f"Wrote {DISPATCH_CSV_PATH}")
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
