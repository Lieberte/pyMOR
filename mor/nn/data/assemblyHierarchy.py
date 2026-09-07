from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any


ASSEMBLY_HIERARCHY_FORMAT_VERSION = 'thermal-assembly-hierarchy-v1'


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'JSON file must contain an object: {path}')
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def _identifier(value: Any, field: str) -> str:
    normalized = str(value).strip()
    if not normalized or not re.fullmatch(r'[A-Za-z0-9_-]+', normalized):
        raise ValueError(f'{field} must be a non-empty path-safe identifier: {value!r}')
    return normalized


def _hierarchy(entry: dict) -> tuple[str, str, str]:
    return (
        _identifier(entry.get('systemType'), 'systemType'),
        _identifier(entry.get('assemblyType'), 'assemblyType'),
        _identifier(entry.get('assemblyId'), 'assemblyId'),
    )


def _relocated(entry: dict, sourceRoot: Path, targetIndexParent: Path) -> dict:
    target = (sourceRoot / entry['path']).resolve()
    return {
        **entry,
        'path': Path(os.path.relpath(target, start=targetIndexParent.resolve())).as_posix(),
    }


def _metadata(entries: list[dict], systemType: str, assemblyType: str, analysisType: str) -> dict:
    return {
        'systemType': systemType,
        'systemTypes': [systemType],
        'assemblyType': assemblyType,
        'assemblyTypes': [assemblyType],
        'assemblyIds': sorted({entry['assemblyId'] for entry in entries}),
        'analysisType': analysisType,
        'analysisTypes': [analysisType],
        'motorIds': sorted(
            {entry['motorId'] for entry in entries if entry.get('motorId')}
        ),
        'parentCaseIds': sorted({entry['parentCaseId'] for entry in entries}),
        'caseCount': len(entries),
        'hierarchy': (
            'systemType/assemblyType/assemblyId/componentType/componentInstance'
        ),
    }


def writeAssemblyHierarchyIndexes(datasetRoot: str | Path) -> Path:
    root = Path(datasetRoot)
    viewsPath = root / 'views.json'
    views = _load(viewsPath)
    catalog: dict[str, Any] = {
        'formatVersion': ASSEMBLY_HIERARCHY_FORMAT_VERSION,
        'levels': [
            'systemType',
            'assemblyType',
            'assemblyId',
            'componentType',
            'componentInstance',
        ],
        'systems': {},
    }
    genericViews = {}

    for view in ('point', 'graph'):
        viewRoot = root / view
        componentSourcePath = viewRoot / 'dataset.json'
        componentSource = _load(componentSourcePath)
        wholeSourcePath = viewRoot / 'whole_motor' / 'dataset.json'
        wholeSource = _load(wholeSourcePath)
        wholeByParent = {
            (entry['assemblyId'], entry['parentCaseId']): entry
            for entry in wholeSource.get('cases', ())
        }

        genericWholeRoot = viewRoot / 'whole_assembly'
        genericWholeEntries = [
            _relocated(entry, wholeSourcePath.parent, genericWholeRoot)
            for entry in wholeSource.get('cases', ())
        ]
        _write(
            genericWholeRoot / 'dataset.json',
            {
                'formatVersion': 'whole-assembly-dataset-v1',
                'viewType': 'wholeAssembly',
                'metadata': componentSource.get('metadata', {}),
                'cases': genericWholeEntries,
            },
        )
        genericViews[f'{view}WholeAssembly'] = f'{view}/whole_assembly/dataset.json'

        combinations = sorted(
            {
                (*_hierarchy(entry)[:2], str(entry['analysisType']))
                for entry in componentSource.get('cases', ())
            }
        )
        for systemType, assemblyType, analysisType in combinations:
            entries = [
                entry
                for entry in componentSource.get('cases', ())
                if entry['systemType'] == systemType
                and entry['assemblyType'] == assemblyType
                and entry['analysisType'] == analysisType
            ]
            branchRoot = (
                viewRoot / 'by_system' / systemType / assemblyType / analysisType
            )
            branchEntries = [
                _relocated(entry, componentSourcePath.parent, branchRoot)
                for entry in entries
            ]
            analysisIndexPath = viewRoot / analysisType / 'dataset.json'
            analysisIndex = _load(analysisIndexPath)
            branchMetadata = _metadata(
                entries, systemType, assemblyType, analysisType
            )
            _write(
                branchRoot / 'dataset.json',
                {
                    'formatVersion': componentSource.get('formatVersion'),
                    'viewType': componentSource.get('viewType'),
                    'systemType': systemType,
                    'assemblyType': assemblyType,
                    'analysisType': analysisType,
                    'connectivityStored': componentSource.get('connectivityStored'),
                    'selfContainedCases': True,
                    'metadata': branchMetadata,
                    'tasks': analysisIndex.get('tasks', {}),
                    'cases': branchEntries,
                },
            )

            parentKeys = {
                (entry['assemblyId'], entry['parentCaseId']) for entry in entries
            }
            wholeEntries = [wholeByParent[key] for key in sorted(parentKeys)]
            wholeBranchRoot = branchRoot / 'whole_assembly'
            relocatedWhole = [
                _relocated(entry, wholeSourcePath.parent, wholeBranchRoot)
                for entry in wholeEntries
            ]
            _write(
                wholeBranchRoot / 'dataset.json',
                {
                    'formatVersion': 'whole-assembly-dataset-v1',
                    'viewType': 'wholeAssembly',
                    'systemType': systemType,
                    'assemblyType': assemblyType,
                    'analysisType': analysisType,
                    'metadata': {
                        **branchMetadata,
                        'caseCount': len(relocatedWhole),
                    },
                    'cases': relocatedWhole,
                },
            )

            system = catalog['systems'].setdefault(
                systemType, {'assemblyTypes': {}}
            )
            assembly = system['assemblyTypes'].setdefault(
                assemblyType,
                {
                    'assemblyIds': [],
                    'views': {},
                },
            )
            assembly['assemblyIds'] = sorted(
                set(assembly['assemblyIds'])
                | {entry['assemblyId'] for entry in entries}
            )
            assembly['views'].setdefault(view, {})[analysisType] = {
                'components': (branchRoot / 'dataset.json').relative_to(root).as_posix(),
                'wholeAssembly': (
                    wholeBranchRoot / 'dataset.json'
                ).relative_to(root).as_posix(),
            }

            analysisWholeAliasRoot = viewRoot / analysisType / 'whole_assembly'
            analysisWholeSource = viewRoot / analysisType / 'whole_motor' / 'dataset.json'
            if analysisWholeSource.is_file():
                analysisWhole = _load(analysisWholeSource)
                aliasEntries = [
                    _relocated(entry, analysisWholeSource.parent, analysisWholeAliasRoot)
                    for entry in analysisWhole.get('cases', ())
                ]
                _write(
                    analysisWholeAliasRoot / 'dataset.json',
                    {
                        **analysisWhole,
                        'viewType': 'wholeAssembly',
                        'cases': aliasEntries,
                    },
                )
                genericViews[
                    f'{view}{analysisType.capitalize()}WholeAssembly'
                ] = (
                    f'{view}/{analysisType}/whole_assembly/dataset.json'
                )

        componentSource.setdefault('metadata', {})['hierarchyCatalog'] = (
            '../hierarchy.json'
        )
        _write(componentSourcePath, componentSource)

    views.setdefault('views', {}).update(genericViews)
    views['hierarchyCatalog'] = 'hierarchy.json'
    views['hierarchyLevels'] = catalog['levels']
    _write(viewsPath, views)
    hierarchyPath = root / 'hierarchy.json'
    _write(hierarchyPath, catalog)
    return hierarchyPath
