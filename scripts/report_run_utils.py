from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def reset_report_dir(report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    for child in report_dir.iterdir():
        if child.is_file():
            child.unlink()


def write_case_manifest(report_dir: Path, phase_name: str, cases: list[dict[str, Any]]) -> tuple[Path, Path]:
    json_path = report_dir / f"{phase_name}_case_manifest.json"
    csv_path = report_dir / f"{phase_name}_case_manifest.csv"
    json_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    fieldnames = sorted({key for case in cases for key in case.keys()}) if cases else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cases)
    return json_path, csv_path
