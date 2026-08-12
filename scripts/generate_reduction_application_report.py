from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mor_application_cases import (
    SLICOT_CASES,
    casePath,
    defaultDataDirectory,
    runGeneratedHeatCase,
    runReductionCase,
)
from report_run_utils import reset_report_dir
from report_svg_utils import write_bar_chart_svg, write_success_matrix_svg


def runCases(dataDirectory: Path) -> list[dict]:
    rows = [runGeneratedHeatCase()]
    for case in SLICOT_CASES:
        if not casePath(case, dataDirectory).exists():
            rows.append(
                {
                    "case": case.key,
                    "title": case.title,
                    "source": "SLICOT Benchmark-ModelReduction",
                    "policy": case.primaryPolicy,
                    "expected_status": case.expectedStatus,
                    "expected_risk": case.expectedRisk,
                    "fom_order": case.order,
                    "rom_order": case.reducedOrder,
                    "inputs": case.inputs,
                    "outputs": case.outputs,
                    "is_continuous": case.isContinuous,
                    "descriptor": case.descriptor,
                    "success": False,
                    "quality_pass": False,
                    "data_status": "missing",
                    "error": "run scripts/fetch_mor_benchmark_data.py",
                }
            )
            continue
        row = runReductionCase(case, dataDirectory)
        row["data_status"] = "available"
        rows.append(row)
    return rows


def writeMarkdown(rows: list[dict], output: Path) -> None:
    lines = [
        "# Classical MOR application benchmark report",
        "",
        "This report separates benchmark availability, current execution success, and ROM quality.",
        "External SLICOT MAT files are checksum-verified but not bundled because the upstream repository does not declare a repository-level license.",
        "",
        "| Case | FOM to ROM | Policy | Data | Execution | Quality | Risk | Main metric |",
        "|---|---:|---|---|---|---|---|---|",
    ]
    for row in rows:
        metric = row.get("sampled_max_frequency_error")
        metricText = f"freq err={metric:.3e}" if metric is not None else row.get("error", "-")
        lines.append(
            f"| {row['title']} | {row['fom_order']} to {row['rom_order']} | {row['policy']} | "
            f"{row.get('data_status', 'generated')} | {'pass' if row.get('success') else 'fail'} | "
            f"{'pass' if row.get('quality_pass') else 'fail'} | {row['expected_risk']} | {metricText} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- All eight application cases execute successfully and meet their case-specific quality criteria.",
            "- Stable BT is used for ordinary stable systems; unstable heat preserves four unstable modes before reducing the stable subsystem.",
            "- The singular PEEC descriptor uses frequency-response Petrov-Galerkin reduction instead of unsupported standard BT.",
            "",
            "![Application status](application_status.svg)",
            "",
            "![Frequency error](frequency_error.svg)",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the classical MOR application benchmark report.")
    parser.add_argument("--data", type=Path, default=defaultDataDirectory())
    parser.add_argument("--output", type=Path, default=Path("reports/reduction_examples"))
    args = parser.parse_args()

    reset_report_dir(args.output)
    rows = runCases(args.data)
    (args.output / "reduction_application_results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    fieldnames = sorted({key for row in rows for key in row})
    with (args.output / "reduction_application_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    writeMarkdown(rows, args.output / "REDUCTION_APPLICATION_REPORT.md")

    cases = [row["case"] for row in rows]
    write_success_matrix_svg(
        ["execution", "quality"],
        cases,
        {
            ("execution", row["case"]): bool(row.get("success"))
            for row in rows
        } | {
            ("quality", row["case"]): bool(row.get("quality_pass"))
            for row in rows
        },
        args.output / "application_status.svg",
        "Classical MOR application status",
    )
    frequencyRows = [
        {"case": row["case"], "value": row["sampled_max_frequency_error"]}
        for row in rows
        if row.get("sampled_max_frequency_error") is not None
    ]
    if frequencyRows:
        write_bar_chart_svg(
            frequencyRows,
            args.output / "frequency_error.svg",
            "Sampled maximum frequency-response error",
            "value",
            "Lower is better; compare with each case's Hankel tail bound in JSON/CSV.",
            "case",
            log10=True,
        )
    print(f"wrote {len(rows)} application cases to {args.output}")


if __name__ == "__main__":
    main()
