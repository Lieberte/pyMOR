from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "classical_mor"
TEX_PATH = REPORT_DIR / "classical_mor_layered_report.tex"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latex_escape(value) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def number(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def svd_algorithm_summary() -> list[str]:
    rows = read_csv(ROOT / "reports" / "svd" / "svd_dispatch_analysis.csv")
    lines = []
    for family in sorted({row["family"] for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        ratios = [number(row["auto_runtime_ratio"]) for row in family_rows if row["auto_runtime_ratio"] != "inf"]
        reviews = sum(row["recommendation"] != "keep" for row in family_rows)
        fastest_counts: dict[str, int] = {}
        for row in family_rows:
            fastest = row["fastest_successful_variant"]
            fastest_counts[fastest] = fastest_counts.get(fastest, 0) + 1
        fastest_summary = ", ".join(f"{name}:{count}" for name, count in sorted(fastest_counts.items()))
        lines.append(
            rf"{latex_escape(family)} & {len(family_rows)} & {sum(ratios) / len(ratios):.2f} & {max(ratios):.2f} & {reviews} & {latex_escape(fastest_summary)} \\"
        )
    return lines


def pathological_summary() -> list[str]:
    rows = read_csv(ROOT / "reports" / "svd" / "svd_results.csv")
    families = {
        "ill_conditioned_diagonal",
        "singular_diagonal",
        "geometric_spectrum",
        "rank_deficient_tall_skinny",
        "nearly_collinear_snapshots",
        "extreme_scaled_diagonal",
        "zero_tall_skinny",
    }
    lines = []
    for family in sorted(families):
        family_rows = [row for row in rows if row["family"] == family]
        cases = sorted({row["case"] for row in family_rows})
        auto_variants = sorted({row["auto_variant"] for row in family_rows})
        best_errors = []
        for case in cases:
            case_rows = [row for row in family_rows if row["case"] == case and row["success"] == "True"]
            best_errors.append(min((number(row["relative_reconstruction_error"], float("inf")) for row in case_rows), default=float("inf")))
        lines.append(
            rf"{latex_escape(family)} & {len(cases)} & {latex_escape(', '.join(auto_variants))} & {max(best_errors):.3e} \\"
        )
    return lines


def strategy_summary(path: Path, residual_key: str = "relative_residual") -> list[str]:
    rows = read_csv(path)
    lines = []
    for family in sorted({row["family"] for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        failures = sum(row.get("success") not in {"True", "true", "1"} for row in family_rows)
        max_residual = max(number(row[residual_key]) for row in family_rows)
        max_rank = max(int(float(row["rank"])) for row in family_rows)
        recommendations = sorted({row["recommendation"] for row in family_rows if row["recommendation"] != "keep"})
        lines.append(
            rf"{latex_escape(family)} & {len(family_rows)} & {failures} & {max_residual:.3e} & {max_rank} & {latex_escape(', '.join(recommendations) or 'keep')} \\"
        )
    return lines


def lyapunov_dispatch_summary() -> list[str]:
    rows = read_csv(ROOT / "reports" / "lyapunov" / "lyapunov_dispatch_results.csv")
    lines = []
    for row in rows:
        lines.append(
            rf"{latex_escape(row['case'])} & {latex_escape(row['auto_variant'] or 'unsupported')} & {latex_escape(row['expected_variant'] or 'unsupported')} & {latex_escape(row['dispatch_success'])} & {latex_escape(row['algorithm_success'])} \\"
        )
    return lines


def dense_lyapunov_summary() -> list[str]:
    rows = read_csv(ROOT / "reports" / "dense_lyapunov" / "dense_lyapunov_results.csv")
    cases = sorted({row["case"] for row in rows})
    lines = []
    for case in cases:
        case_rows = [row for row in rows if row["case"] == case]
        auto_variant = case_rows[0]["auto_variant"]
        auto_row = next(row for row in case_rows if row["variant"] == auto_variant)
        fastest = min(case_rows, key=lambda row: number(row["elapsed_ms"], float("inf")))
        lines.append(
            rf"{latex_escape(case)} & {latex_escape(auto_variant)} & {number(auto_row['relative_residual']):.3e} & {latex_escape(fastest['variant'])} & {number(fastest['elapsed_ms']):.3f} \\"
        )
    return lines


def dmd_summary() -> list[str]:
    rows = read_csv(ROOT / "reports" / "dmd" / "dmd_results.csv")
    lines = []
    for row in rows:
        lines.append(
            rf"{latex_escape(row['case'])} & {latex_escape(row['variant'])} & {row['rank']} & {number(row['eigenvalue_error']):.3e} & {number(row['reconstruction_error']):.3e} & {number(row['elapsed_ms']):.3f} \\"
        )
    return lines


def table(headers: list[str], rows: list[str]) -> str:
    columns = "p{0.27\\linewidth}" + "p{0.11\\linewidth}" * (len(headers) - 1)
    header = " & ".join(latex_escape(item) for item in headers) + r" \\"
    return "\n".join(
        [
            r"\begin{longtable}{" + columns + "}",
            header,
            r"\hline",
            *rows,
            r"\end{longtable}",
        ]
    )


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    svd_payload = read_json(ROOT / "reports" / "svd" / "svd_results.json")
    lradi_payload = read_json(ROOT / "reports" / "lradi" / "lradi_results.json")
    lrsmith_payload = read_json(ROOT / "reports" / "lrsmith" / "lrsmith_results.json")
    dmd_payload = read_json(ROOT / "reports" / "dmd" / "dmd_results.json")
    text = rf"""\documentclass[UTF8]{{ctexart}}
\usepackage{{geometry}}
\usepackage{{longtable}}
\usepackage{{booktabs}}
\usepackage{{hyperref}}
\geometry{{a4paper, margin=2.2cm}}
\title{{Classical MOR 分层测试与自动分发报告}}
\author{{pyMOR classical optimization}}
\date{{SVD: {latex_escape(svd_payload['generated_at'])}\\LRADI: {latex_escape(lradi_payload['generated_at'])}\\LRSmith: {latex_escape(lrsmith_payload['generated_at'])}\\DMD: {latex_escape(dmd_payload['generated_at'])}}}

\begin{{document}}
\maketitle

\section{{分层原则}}
本报告保留现有 JSON、CSV、Markdown 和 SVG 产物，同时以 LaTeX 汇总分层证据链。建议顺序为：底层算法 benchmark、operator 兼容性、auto dispatch 策略、上层集成验证。底层测试确定默认策略后，上层调用只验证 auto 路径，不重复铺满全部算法组合。

\section{{Algorithm Layer：底层算法 benchmark}}
\subsection{{SVD 算法与病态矩阵}}
{table(['family', 'cases', 'auto/fastest mean', 'max ratio', 'reviews', 'fastest variants'], svd_algorithm_summary())}

\subsection{{SVD 病态矩阵覆盖}}
{table(['family', 'cases', 'auto variants', 'max best error'], pathological_summary())}

\subsection{{LRADI 策略表现}}
{table(['family', 'cases', 'failures', 'max residual', 'max rank', 'recommendations'], strategy_summary(ROOT / 'reports' / 'lradi' / 'lradi_strategy_analysis.csv'))}

\subsection{{LRSmith 策略表现}}
{table(['family', 'cases', 'failures', 'max residual', 'max rank', 'recommendations'], strategy_summary(ROOT / 'reports' / 'lrsmith' / 'lrsmith_strategy_analysis.csv'))}

\subsection{{DMD 谱与轨迹误差}}
{table(['case', 'variant', 'rank', 'eigenvalue error', 'reconstruction error', 'elapsed ms'], dmd_summary())}

\section{{Operator Layer：operator 兼容性}}
operator 兼容性由 regression tests 和各算法报告共同覆盖：dense matrixOperator、scipy sparse matrixOperator、lowRankOperator、scaledOperator、sumOperator、descriptor/generalized pair。详细明细保留在各算法的 case manifest 与 CSV 中。

\section{{Dispatch Layer：auto 策略验证}}
\subsection{{Lyapunov 顶层 dispatch}}
{table(['case', 'auto', 'expected', 'dispatch ok', 'algorithm ok'], lyapunov_dispatch_summary())}

\subsection{{Dense Lyapunov auto 对比}}
{table(['case', 'auto', 'auto residual', 'fastest observed', 'fastest ms'], dense_lyapunov_summary())}

\section{{Integration Layer：上层调用策略}}
上层模型和 reductor 应默认走 auto。底层 algorithm layer 确定候选算法和风险边界，operator layer 确定输入契约，dispatch layer 固化默认策略。上层只需要验证 auto 路径下的端到端误差、形状契约和失败模式。

\section{{保留产物}}
完整机器可读数据保留在各目录的 JSON/CSV 中；可视化保留为 SVG；中文 Markdown 报告继续作为人工快速阅读入口。本 LaTeX 文件只作为跨算法分层汇总。

\end{{document}}
"""
    TEX_PATH.write_text(text, encoding="utf-8-sig")
    print(f"Wrote {TEX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
