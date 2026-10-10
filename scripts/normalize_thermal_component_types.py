#!/usr/bin/env python3
"""Recompute semantic componentType labels without changing component instances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mor.nn.data.motorFemMetadata import (
    componentTypeName,
)
from mor.nn.data.motorFemIO import (
    loadH5py,
)
from mor.nn.data.motorFullResolutionDataset import (
    _geometryClassificationTask,
    _writeJson,
)
from mor.nn.data.motorTransientDataset import writeMotorThermalAnalysisIndexes


def _transform(value: Any) -> tuple[Any, int]:
    if isinstance(value, list):
        result = []
        changes = 0
        for item in value:
            updated, count = _transform(item)
            result.append(updated)
            changes += count
        return result, changes
    if not isinstance(value, dict):
        return value, 0

    updated = {}
    changes = 0
    for key, item in value.items():
        transformed, count = _transform(item)
        updated[key] = transformed
        changes += count
    instance = updated.get('componentInstance')
    oldType = updated.get('componentType')
    if isinstance(instance, str) and isinstance(oldType, str):
        newType = componentTypeName(instance)
        if newType != oldType:
            updated['componentType'] = newType
            changes += 1
    return updated, changes


def normalizeDataset(root: Path, *, apply: bool) -> dict[str, int]:
    root = root.resolve()
    if not (root / 'views.json').is_file():
        raise ValueError(f'not a full-resolution thermal dataset: {root}')

    changedJson = 0
    changedRecords = 0
    for path in sorted(root.rglob('*.json')):
        original = json.loads(path.read_text(encoding='utf-8'))
        updated, changes = _transform(original)
        if changes:
            changedJson += 1
            changedRecords += changes
            if apply:
                _writeJson(path, updated)

    changedSeries = 0
    h5py = loadH5py()
    for path in sorted(root.rglob('series.h5')):
        mode = 'r+' if apply else 'r'
        with h5py.File(path, mode) as series:
            instance = series.attrs.get('componentInstance')
            oldType = series.attrs.get('componentType')
            if isinstance(instance, bytes):
                instance = instance.decode('utf-8')
            if isinstance(oldType, bytes):
                oldType = oldType.decode('utf-8')
            if not isinstance(instance, str) or not isinstance(oldType, str):
                continue
            newType = componentTypeName(instance)
            if newType != oldType:
                changedSeries += 1
                if apply:
                    series.attrs['componentType'] = newType

    if apply:
        for view in ('point', 'graph'):
            indexPath = root / view / 'dataset.json'
            index = json.loads(indexPath.read_text(encoding='utf-8'))
            index.setdefault('tasks', {})['geometryComponentClassification'] = (
                _geometryClassificationTask(index.get('cases', ()))
            )
            _writeJson(indexPath, index)
        writeMotorThermalAnalysisIndexes(root)

    return {
        'jsonFilesChanged': changedJson,
        'componentRecordsChanged': changedRecords,
        'seriesFilesChanged': changedSeries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_root', type=Path)
    parser.add_argument('--apply', action='store_true')
    arguments = parser.parse_args()
    result = normalizeDataset(arguments.dataset_root, apply=arguments.apply)
    print(json.dumps({'applied': arguments.apply, **result}, indent=2))


if __name__ == '__main__':
    main()
