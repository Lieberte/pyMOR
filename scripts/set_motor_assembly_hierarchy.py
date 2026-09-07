#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from mor.nn.data.motorFullResolutionDataset import _writeJson
from mor.nn.data.motorTransientDataset import writeMotorThermalAnalysisIndexes


SYSTEM_TYPE = 'electric_machine'
ASSEMBLY_TYPE = 'motor'


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _fields(motorId: str) -> dict[str, str]:
    return {
        'systemType': SYSTEM_TYPE,
        'assemblyType': ASSEMBLY_TYPE,
        'assemblyId': str(motorId),
    }


def _datasetMetadata(index: dict) -> None:
    entries = list(index.get('cases', ()))
    metadata = index.setdefault('metadata', {})
    metadata.update(
        {
            'systemTypes': sorted({entry['systemType'] for entry in entries}),
            'assemblyTypes': sorted({entry['assemblyType'] for entry in entries}),
            'assemblyIds': sorted({entry['assemblyId'] for entry in entries}),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Classify existing motor cases in the generic assembly hierarchy.'
    )
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    arguments = parser.parse_args()

    root = arguments.dataset.resolve()
    componentManifestCount = 0
    wholeManifestCount = 0
    seriesCount = 0
    assemblyIds = set()
    updatedIndexes: list[tuple[Path, dict]] = []

    for view in ('point', 'graph'):
        indexPath = root / view / 'dataset.json'
        index = _load(indexPath)
        for entry in index.get('cases', ()):
            labels = _fields(entry['motorId'])
            assemblyIds.add(labels['assemblyId'])
            entry.update(labels)
            manifestPath = indexPath.parent / entry['path'] / 'manifest.json'
            manifest = _load(manifestPath)
            manifest.update(labels)
            manifest.setdefault('metadata', {}).update(labels)
            if not arguments.dry_run:
                _writeJson(manifestPath, manifest)
                seriesPath = manifestPath.parent / manifest.get('seriesFile', 'series.h5')
                if manifest.get('timeSeriesStored') and seriesPath.is_file():
                    with h5py.File(seriesPath, 'r+') as series:
                        for key, value in labels.items():
                            series.attrs[key] = value
                    seriesCount += 1
            componentManifestCount += 1
        _datasetMetadata(index)
        updatedIndexes.append((indexPath, index))

        wholeIndexPath = root / view / 'whole_motor' / 'dataset.json'
        wholeIndex = _load(wholeIndexPath)
        for entry in wholeIndex.get('cases', ()):
            labels = _fields(entry['motorId'])
            entry.update(labels)
            entry.setdefault('metadata', {}).update(labels)
            manifestPath = wholeIndexPath.parent / entry['path'] / 'manifest.json'
            manifest = _load(manifestPath)
            manifest.update(labels)
            manifest.setdefault('metadata', {}).update(labels)
            if not arguments.dry_run:
                _writeJson(manifestPath, manifest)
            wholeManifestCount += 1
        _datasetMetadata(wholeIndex)
        updatedIndexes.append((wholeIndexPath, wholeIndex))

    if not arguments.dry_run:
        for path, index in updatedIndexes:
            _writeJson(path, index)
        viewsPath = root / 'views.json'
        views = _load(viewsPath)
        views.update(
            {
                'systemTypes': [SYSTEM_TYPE],
                'assemblyTypes': [ASSEMBLY_TYPE],
                'assemblyIds': sorted(assemblyIds),
            }
        )
        _writeJson(viewsPath, views)
        writeMotorThermalAnalysisIndexes(root)

    print(
        json.dumps(
            {
                'dataset': str(root),
                'dryRun': arguments.dry_run,
                'systemType': SYSTEM_TYPE,
                'assemblyType': ASSEMBLY_TYPE,
                'assemblyIds': sorted(assemblyIds),
                'componentManifests': componentManifestCount,
                'wholeAssemblyManifests': wholeManifestCount,
                'transientSeriesLabeled': seriesCount,
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
