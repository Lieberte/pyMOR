#!/usr/bin/env python3
"""Map numbered disk FEM instances to the magnetic-isolation-plate semantic type."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DISK_INSTANCE_TYPES = {'disk1', 'disk2'}
COMPONENT_TYPE = 'magnetic_isolation_plate'


def transform(value: Any) -> Any:
    if isinstance(value, list):
        return [transform(item) for item in value]
    if not isinstance(value, dict):
        return value

    updated = {key: transform(item) for key, item in value.items()}
    oldType = updated.get('componentType')
    if isinstance(oldType, str):
        updated.setdefault('componentInstance', oldType)
        if oldType in DISK_INSTANCE_TYPES or oldType == 'disk':
            updated['componentType'] = COMPONENT_TYPE

        metadata = updated.get('metadata')
        if isinstance(metadata, dict) and (
            'caseId' in updated or 'geometryId' in updated
        ):
            metadata.setdefault('componentInstance', updated['componentInstance'])
    return updated


def _jsonFiles(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.rglob('*.json')))


def updateDataset(root: Path, *, apply: bool = False) -> dict[str, int]:
    root = root.resolve()
    if not (root / 'views.json').is_file():
        raise ValueError(f'not a dual motor dataset root: {root}')
    changed: dict[Path, bytes] = {}
    files = _jsonFiles(root)
    for path in files:
        original = json.loads(path.read_text(encoding='utf-8'))
        content = (json.dumps(transform(original), indent=2, ensure_ascii=False) + '\n').encode('utf-8')
        if content != path.read_bytes():
            changed[path] = content
    if apply:
        for path, content in changed.items():
            temporary = path.with_name(f'.{path.name}.merge-disk-type.tmp')
            temporary.write_bytes(content)
            os.replace(temporary, path)
    return {'jsonFilesScanned': len(files), 'jsonFilesChanged': len(changed)}


def validateDataset(root: Path) -> dict[str, int]:
    root = root.resolve()
    oldTypeCount = 0
    diskTypeCount = 0
    missingInstanceCount = 0
    componentRecords = 0
    for path in _jsonFiles(root):
        data = json.loads(path.read_text(encoding='utf-8'))
        stack = [data]
        while stack:
            value = stack.pop()
            if isinstance(value, list):
                stack.extend(value)
            elif isinstance(value, dict):
                stack.extend(value.values())
                componentType = value.get('componentType')
                if isinstance(componentType, str):
                    componentRecords += 1
                    if componentType in DISK_INSTANCE_TYPES or componentType == 'disk':
                        oldTypeCount += 1
                    if componentType == COMPONENT_TYPE:
                        diskTypeCount += 1
                        if value.get('componentInstance') not in DISK_INSTANCE_TYPES:
                            missingInstanceCount += 1
    if oldTypeCount or missingInstanceCount or diskTypeCount == 0:
        raise ValueError(
            'component type validation failed: '
            f'oldDiskTypes={oldTypeCount}, invalidDiskInstances={missingInstanceCount}, '
            f'magneticIsolationPlateRecords={diskTypeCount}'
        )
    return {
        'componentRecords': componentRecords,
        'magneticIsolationPlateRecords': diskTypeCount,
        'oldDiskTypeRecords': oldTypeCount,
        'invalidDiskInstanceRecords': missingInstanceCount,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_root', type=Path)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--validate', action='store_true')
    arguments = parser.parse_args()
    if arguments.apply and arguments.validate:
        parser.error('--apply and --validate are mutually exclusive')
    if arguments.validate:
        result = {'mode': 'validated', **validateDataset(arguments.dataset_root)}
    else:
        result = {
            'mode': 'updated' if arguments.apply else 'preflight',
            **updateDataset(arguments.dataset_root, apply=arguments.apply),
        }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
