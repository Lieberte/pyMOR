#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mor.nn.data.motorFemComponentCase import _thermalAnalysisMetadata
from mor.nn.data.motorStaDataset import _staTemporalMetadata, inspectMotorStaArchive


TEMPORAL_KEYS = (
    'analysisType',
    'isTransient',
    'temporalSampleType',
    'timeSeriesStored',
    'sourceTimeStepS',
    'sourceEndTimeS',
    'snapshotTimeS',
    'sourceSnapshotCount',
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'JSON file must contain an object: {path}')
    return value


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def _controlDirectory(resultFile: str) -> Path:
    path = Path(resultFile)
    for parent in (path.parent, *path.parents):
        if (parent / 'controlData.json').is_file():
            return parent
    raise FileNotFoundError(f'cannot find controlData.json above result file: {path}')


def _temporalMetadata(manifest: dict, cache: dict[str, dict]) -> dict:
    metadata = manifest.get('metadata', {})
    sourceSta = metadata.get('sourceSta')
    if sourceSta:
        key = f'sta:{Path(sourceSta).resolve()}'
        if key not in cache:
            cache[key] = _staTemporalMetadata(inspectMotorStaArchive(sourceSta))
        return cache[key]
    resultFile = metadata.get('resultFile')
    if not resultFile:
        raise ValueError(f'manifest has no sourceSta or resultFile: {manifest.get("caseId")}')
    controlDirectory = _controlDirectory(str(resultFile))
    key = f'control:{controlDirectory.resolve()}'
    if key not in cache:
        control = _load(controlDirectory / 'controlData.json')
        cache[key] = _thermalAnalysisMetadata(control)
    return cache[key]


def _publicFields(metadata: dict) -> dict:
    fields = {
        'analysisType': metadata['analysisType'],
        'temporalSampleType': metadata['temporalSampleType'],
    }
    if metadata.get('snapshotTimeS') is not None:
        fields['snapshotTimeS'] = metadata['snapshotTimeS']
    return fields


def _replaceTemporalMetadata(target: dict, labels: dict) -> None:
    for key in TEMPORAL_KEYS:
        target.pop(key, None)
    target.update({key: value for key, value in labels.items() if value is not None})


def _updateComponentView(
    viewRoot: Path,
    *,
    dryRun: bool,
    cache: dict[str, dict],
) -> tuple[dict[str, dict], list[Path]]:
    indexPath = viewRoot / 'dataset.json'
    index = _load(indexPath)
    labelsByParent: dict[str, dict] = {}
    changedPaths = []
    for entry in index.get('cases', ()):
        manifestPath = viewRoot / entry['path'] / 'manifest.json'
        manifest = _load(manifestPath)
        labels = _temporalMetadata(manifest, cache)
        parentCaseId = str(entry['parentCaseId'])
        previous = labelsByParent.setdefault(parentCaseId, labels)
        if previous != labels:
            raise ValueError(f'conflicting temporal labels in parent case {parentCaseId!r}')

        originalManifest = json.dumps(manifest, sort_keys=True)
        _replaceTemporalMetadata(manifest.setdefault('metadata', {}), labels)
        for key in ('analysisType', 'temporalSampleType', 'snapshotTimeS'):
            manifest.pop(key, None)
        manifest.update(_publicFields(labels))
        operating = manifest.setdefault('operatingFeatures', {})
        operating.pop('snapshotTimeS', None)
        if labels.get('snapshotTimeS') is not None:
            operating['snapshotTimeS'] = labels['snapshotTimeS']
        if json.dumps(manifest, sort_keys=True) != originalManifest:
            changedPaths.append(manifestPath)
            if not dryRun:
                _write(manifestPath, manifest)

        for key in ('analysisType', 'temporalSampleType', 'snapshotTimeS'):
            entry.pop(key, None)
        entry.update(_publicFields(labels))

    index.setdefault('metadata', {})['analysisTypes'] = sorted(
        {labels['analysisType'] for labels in labelsByParent.values()}
    )
    originalIndex = _load(indexPath)
    if index != originalIndex:
        changedPaths.append(indexPath)
        if not dryRun:
            _write(indexPath, index)
    return labelsByParent, changedPaths


def _updateWholeMotorView(
    root: Path,
    labelsByParent: dict[str, dict],
    *,
    dryRun: bool,
) -> list[Path]:
    indexPath = root / 'dataset.json'
    index = _load(indexPath)
    changedPaths = []
    for entry in index.get('cases', ()):
        parentCaseId = str(entry['parentCaseId'])
        labels = labelsByParent[parentCaseId]
        manifestPath = root / entry['path'] / 'manifest.json'
        manifest = _load(manifestPath)
        originalManifest = json.dumps(manifest, sort_keys=True)
        for key in ('analysisType', 'temporalSampleType', 'snapshotTimeS'):
            manifest.pop(key, None)
        manifest.update(_publicFields(labels))
        _replaceTemporalMetadata(manifest.setdefault('metadata', {}), labels)
        if json.dumps(manifest, sort_keys=True) != originalManifest:
            changedPaths.append(manifestPath)
            if not dryRun:
                _write(manifestPath, manifest)

        for key in ('analysisType', 'temporalSampleType', 'snapshotTimeS'):
            entry.pop(key, None)
        entry.update(_publicFields(labels))
        _replaceTemporalMetadata(entry.setdefault('metadata', {}), labels)

    index.setdefault('metadata', {})['analysisTypes'] = sorted(
        {labels['analysisType'] for labels in labelsByParent.values()}
    )
    originalIndex = _load(indexPath)
    if index != originalIndex:
        changedPaths.append(indexPath)
        if not dryRun:
            _write(indexPath, index)
    return changedPaths


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Label motor cases as steady solutions or transient snapshots.'
    )
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    arguments = parser.parse_args()

    cache: dict[str, dict] = {}
    allChanged = []
    referenceLabels = None
    for view in ('point', 'graph'):
        viewRoot = arguments.dataset / view
        labels, changed = _updateComponentView(
            viewRoot, dryRun=arguments.dry_run, cache=cache
        )
        if referenceLabels is not None and labels != referenceLabels:
            raise ValueError('point and graph views have different temporal labels')
        referenceLabels = labels
        allChanged.extend(changed)
        allChanged.extend(
            _updateWholeMotorView(
                viewRoot / 'whole_motor', labels, dryRun=arguments.dry_run
            )
        )

    viewsPath = arguments.dataset / 'views.json'
    views = _load(viewsPath)
    analysisTypes = sorted(
        {labels['analysisType'] for labels in (referenceLabels or {}).values()}
    )
    if views.get('analysisTypes') != analysisTypes:
        views['analysisTypes'] = analysisTypes
        allChanged.append(viewsPath)
        if not arguments.dry_run:
            _write(viewsPath, views)

    counts = {analysisType: 0 for analysisType in analysisTypes}
    for labels in (referenceLabels or {}).values():
        counts[labels['analysisType']] += 1
    print(
        json.dumps(
            {
                'dataset': str(arguments.dataset.resolve()),
                'dryRun': arguments.dry_run,
                'parentCaseCounts': counts,
                'changedFileCount': len(set(allChanged)),
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
