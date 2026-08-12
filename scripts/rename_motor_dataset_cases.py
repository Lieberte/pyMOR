#!/usr/bin/env python3
"""Rename generated motor dataset case identities without touching array data."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_CASE_MAPPING = {
    "2500": "tutorial-a",
    "5000": "tutorial-b",
    "7500": "tutorial-c",
    "10000": "tutorial-d",
    "12500": "tutorial-e",
    "15000": "tutorial-f",
    "17500": "tutorial-g",
    "20000": "tutorial-h",
    "22500": "tutorial-i",
    "25000": "tutorial-j",
}

CASE_ID_KEYS = {"caseId", "componentCaseId", "oppositeComponentCaseId"}
CASE_ID_LIST_KEYS = {"caseIds", "componentCaseIds", "oppositeComponentCaseIds"}
PROVENANCE_KEYS = {
    "resultFile",
    "sourceCase",
    "sourceCaseId",
    "sourceDirectory",
    "sourceParentCaseId",
    "sourceParentCaseIds",
    "sourcePath",
}


def _rename_case_id(value: str, mapping: dict[str, str]) -> str:
    for old, new in mapping.items():
        if value == old:
            return new
        if value.startswith(f"{old}-"):
            return f"{new}{value[len(old):]}"
    return value


def _rename_path(value: str, mapping: dict[str, str]) -> str:
    """Rename only complete slash-delimited case path segments."""
    separator = "\\" if "\\" in value and "/" not in value else "/"
    parts = value.split(separator)
    return separator.join(_rename_case_id(part, mapping) for part in parts)


def transform_json(value: Any, mapping: dict[str, str], *, key: str | None = None) -> Any:
    """Transform generated identities while preserving source provenance fields."""
    if key in PROVENANCE_KEYS:
        return value

    if isinstance(value, dict):
        return {item_key: transform_json(item, mapping, key=item_key) for item_key, item in value.items()}

    if isinstance(value, list):
        if key == "parentCaseIds":
            return [mapping.get(item, item) if isinstance(item, str) else item for item in value]
        if key in CASE_ID_LIST_KEYS:
            return [_rename_case_id(item, mapping) if isinstance(item, str) else item for item in value]
        return [transform_json(item, mapping) for item in value]

    if not isinstance(value, str):
        return value
    if key == "parentCaseId":
        return mapping.get(value, value)
    if key in CASE_ID_KEYS:
        return _rename_case_id(value, mapping)
    if key == "path":
        return _rename_path(value, mapping)
    return value


def _directory_renames(root: Path, mapping: dict[str, str]) -> list[tuple[Path, Path]]:
    renames: list[tuple[Path, Path]] = []
    for view in ("point", "graph"):
        component_root = root / view / "motors" / "motor-htc-prototype" / "cases"
        whole_root = root / view / "whole_motor" / "cases"
        for old, new in mapping.items():
            if component_root.is_dir():
                for source in sorted(component_root.glob(f"{old}-*")):
                    renames.append((source, source.with_name(f"{new}{source.name[len(old):]}")))
            renames.append((whole_root / old, whole_root / new))
    return renames


def _preflight(root: Path, mapping: dict[str, str]) -> tuple[list[tuple[Path, Path]], dict[Path, bytes]]:
    if not (root / "views.json").is_file():
        raise ValueError(f"Not a dual motor dataset root: {root}")

    renames = _directory_renames(root, mapping)
    expected_count = len(mapping) * 2 * 9  # eight components plus one whole-motor case per view
    if len(renames) != expected_count:
        raise ValueError(f"Expected {expected_count} directory renames, found {len(renames)}")

    missing = [source for source, _ in renames if not source.is_dir()]
    occupied = [target for _, target in renames if target.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source directories, first: {missing[0]}")
    if occupied:
        raise FileExistsError(f"Target directories already exist, first: {occupied[0]}")

    transformed: dict[Path, bytes] = {}
    for path in sorted(root.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        updated = transform_json(data, mapping)
        transformed[path] = (json.dumps(updated, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    return renames, transformed


def rename_dataset(root: Path, mapping: dict[str, str], *, apply: bool = False) -> dict[str, int]:
    root = root.resolve()
    renames, transformed = _preflight(root, mapping)
    changed_json = {path: content for path, content in transformed.items() if path.read_bytes() != content}
    result = {
        "directoryRenames": len(renames),
        "jsonFilesScanned": len(transformed),
        "jsonFilesChanged": len(changed_json),
    }
    if not apply:
        return result

    temporary_files: dict[Path, Path] = {}
    completed_renames: list[tuple[Path, Path]] = []
    original_json = {path: path.read_bytes() for path in changed_json}
    replaced_json: list[Path] = []
    try:
        for path, content in changed_json.items():
            temporary = path.with_name(f".{path.name}.rename-cases.tmp")
            temporary.write_bytes(content)
            temporary_files[path] = temporary

        for source, target in renames:
            source.rename(target)
            completed_renames.append((source, target))

        for original_path, temporary in temporary_files.items():
            current_path = original_path
            for source, target in renames:
                try:
                    relative = original_path.relative_to(source)
                except ValueError:
                    continue
                current_path = target / relative
                break
            temporary_current = temporary
            if current_path != original_path:
                temporary_current = current_path.with_name(temporary.name)
            os.replace(temporary_current, current_path)
            replaced_json.append(current_path)
    except Exception:
        for path in reversed(replaced_json):
            original_path = path
            for source, target in renames:
                try:
                    relative = path.relative_to(target)
                except ValueError:
                    continue
                original_path = source / relative
                break
            path.write_bytes(original_json[original_path])
        for source, target in reversed(completed_renames):
            if target.exists() and not source.exists():
                target.rename(source)
        for temporary in temporary_files.values():
            if temporary.exists():
                temporary.unlink()
        raise

    return result


def _validate_identity_fields(value: Any, mapping: dict[str, str], errors: list[str], location: str, *, key: str | None = None) -> None:
    if key in PROVENANCE_KEYS:
        return
    if isinstance(value, dict):
        for item_key, item in value.items():
            _validate_identity_fields(item, mapping, errors, f"{location}.{item_key}", key=item_key)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_identity_fields(item, mapping, errors, f"{location}[{index}]", key=key)
        return
    if not isinstance(value, str):
        return
    if key in {"parentCaseId", "parentCaseIds", *CASE_ID_KEYS, *CASE_ID_LIST_KEYS, "path"}:
        for old in mapping:
            if value == old or value.startswith(f"{old}-") or f"/{old}" in value or f"\\{old}" in value:
                errors.append(f"Old generated identity remains at {location}: {value}")


def validate_dataset(root: Path, mapping: dict[str, str]) -> dict[str, int]:
    """Validate renamed indexes, manifests, references, directories and NPZ containers."""
    root = root.resolve()
    expected_parents = list(mapping.values())
    expected_parent_set = set(expected_parents)
    errors: list[str] = []
    json_count = 0

    views = json.loads((root / "views.json").read_text(encoding="utf-8"))
    if views.get("parentCaseIds") != expected_parents:
        errors.append("views.json parentCaseIds do not match the requested tutorial order")

    for path in root.rglob("*.json"):
        json_count += 1
        data = json.loads(path.read_text(encoding="utf-8"))
        _validate_identity_fields(data, mapping, errors, str(path))

    component_count = 0
    whole_count = 0
    npz_count = 0
    for view in ("point", "graph"):
        index_path = root / view / "dataset.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        cases = index.get("cases", [])
        if len(cases) != len(mapping) * 8:
            errors.append(f"{view} component index contains {len(cases)} cases, expected {len(mapping) * 8}")
        indexed_case_ids = {case.get("caseId") for case in cases}
        component_count += len(cases)
        for case in cases:
            case_id = case.get("caseId")
            parent_id = case.get("parentCaseId")
            case_dir = root / view / case.get("path", "")
            if parent_id not in expected_parent_set or not str(case_id).startswith(f"{parent_id}-"):
                errors.append(f"Invalid {view} component identity: {case_id} / {parent_id}")
            if not case_dir.is_dir():
                errors.append(f"Missing indexed component directory: {case_dir}")
                continue
            manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("caseId") != case_id or manifest.get("parentCaseId") != parent_id:
                errors.append(f"Manifest identity does not match index: {case_dir}")
            npz_path = case_dir / "data.npz"
            if not npz_path.is_file() or npz_path.stat().st_size == 0:
                errors.append(f"Missing or empty NPZ: {npz_path}")
            else:
                try:
                    with zipfile.ZipFile(npz_path) as archive:
                        if not archive.infolist():
                            errors.append(f"Empty NPZ archive: {npz_path}")
                except zipfile.BadZipFile:
                    errors.append(f"Invalid NPZ archive: {npz_path}")
                npz_count += 1

        whole_index_path = root / view / "whole_motor" / "dataset.json"
        whole_index = json.loads(whole_index_path.read_text(encoding="utf-8"))
        whole_cases = whole_index.get("cases", [])
        if len(whole_cases) != len(mapping):
            errors.append(f"{view} whole-motor index contains {len(whole_cases)} cases, expected {len(mapping)}")
        whole_count += len(whole_cases)
        for case in whole_cases:
            parent_id = case.get("parentCaseId")
            case_dir = whole_index_path.parent / case.get("path", "")
            if parent_id not in expected_parent_set or case_dir.name != parent_id:
                errors.append(f"Invalid {view} whole-motor identity: {parent_id} / {case_dir.name}")
            if not case_dir.is_dir():
                errors.append(f"Missing indexed whole-motor directory: {case_dir}")
                continue
            manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("parentCaseId") != parent_id:
                errors.append(f"Whole-motor manifest identity does not match index: {case_dir}")
            components = manifest.get("components", [])
            if len(components) != 8:
                errors.append(f"Whole-motor case does not contain eight components: {case_dir}")
            for component in components:
                component_id = component.get("caseId")
                if component_id not in indexed_case_ids:
                    errors.append(f"Unknown whole-motor component reference: {component_id}")
                component_path = (case_dir / component.get("path", "")).resolve()
                if not component_path.is_dir():
                    errors.append(f"Broken whole-motor component path: {component_path}")
            for interface in manifest.get("interfaces", []):
                for side in interface.get("sides", []):
                    for field in ("componentCaseId", "oppositeComponentCaseId"):
                        component_id = side.get(field)
                        if component_id is not None and component_id not in indexed_case_ids:
                            errors.append(f"Unknown interface reference {field}: {component_id}")

    old_directories = []
    for view in ("point", "graph"):
        component_root = root / view / "motors" / "motor-htc-prototype" / "cases"
        whole_root = root / view / "whole_motor" / "cases"
        for old in mapping:
            old_directories.extend(component_root.glob(f"{old}-*"))
            if (whole_root / old).exists():
                old_directories.append(whole_root / old)
    if old_directories:
        errors.append(f"Old numeric case directories remain, first: {old_directories[0]}")
    assets = [path for path in root.rglob("assets") if path.is_dir()]
    if assets:
        errors.append(f"Unexpected assets directory remains, first: {assets[0]}")

    if errors:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(errors[:20]))
    return {
        "jsonFiles": json_count,
        "componentCases": component_count,
        "wholeMotorCases": whole_count,
        "npzArchives": npz_count,
        "assetsDirectories": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--apply", action="store_true", help="Apply changes; otherwise perform preflight only")
    parser.add_argument("--validate", action="store_true", help="Validate an already-renamed dataset")
    args = parser.parse_args()
    if args.apply and args.validate:
        parser.error("--apply and --validate are mutually exclusive")
    if args.validate:
        print(json.dumps({"mode": "validated", **validate_dataset(args.dataset_root, DEFAULT_CASE_MAPPING)}, indent=2))
        return
    result = rename_dataset(args.dataset_root, DEFAULT_CASE_MAPPING, apply=args.apply)
    mode = "renamed" if args.apply else "preflight"
    print(json.dumps({"mode": mode, **result}, indent=2))


if __name__ == "__main__":
    main()
