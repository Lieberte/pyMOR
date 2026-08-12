from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .thermalComponentCase import loadThermalComponentCase, thermalComponentCase


WHOLE_MOTOR_FORMAT_VERSION = 'whole-motor-dataset-v1'


@dataclass(frozen=True)
class wholeMotorCase:
    motorId: str
    parentCaseId: str
    components: tuple[thermalComponentCase, ...]
    componentPaths: tuple[Path, ...]
    interfaces: tuple[Mapping[str, Any], ...]
    externalPatches: tuple[Mapping[str, Any], ...]
    inputPolicy: Mapping[str, Any]
    summary: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class wholeMotorDataset:
    sourcePath: Path
    cases: tuple[wholeMotorCase, ...]
    metadata: Mapping[str, Any]


def _jsonReady(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonReady(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonReady(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'unsupported whole-motor JSON value: {type(value).__name__}')


def _patchRecord(
    case: thermalComponentCase,
    patch: Mapping[str, Any],
    regionToCaseId: Mapping[str, str],
) -> dict[str, Any]:
    oppositeRegionUuid = str(patch.get('oppositeRegionUuid', ''))
    return {
        'componentCaseId': case.caseId,
        'componentType': case.componentType,
        'componentInstance': case.metadata.get('componentInstance', case.componentType),
        'regionUuid': case.metadata['regionUuid'],
        'patchId': int(patch['patchId']),
        'patchUuid': str(patch['patchUuid']),
        'patchName': str(patch['patchName']),
        'surfaceAreaM2': float(patch['surfaceAreaM2']),
        'boundaryInputType': str(patch['boundaryInputType']),
        'boundaryTypeId': int(patch['boundaryTypeId']),
        'htcWPerM2K': float(patch['htcWPerM2K']),
        'referenceTemperatureK': float(patch['ambientTemperatureK']),
        'expectedHeatFlowRateW': float(patch['expectedHeatFlowRateW']),
        'sampledHeatFlowRateW': float(patch['sampledReconstructedHeatFlowRateW']),
        'oppositeRegionUuid': oppositeRegionUuid or None,
        'oppositePatchUuid': str(patch.get('oppositePatchUuid', '')) or None,
        'oppositeComponentCaseId': regionToCaseId.get(oppositeRegionUuid),
    }


def loadWholeMotorDataset(directory: str | Path) -> wholeMotorDataset:
    source = Path(directory)
    indexPath = source if source.is_file() else source / 'dataset.json'
    index = json.loads(indexPath.read_text(encoding='utf-8'))
    if index.get('formatVersion') != WHOLE_MOTOR_FORMAT_VERSION:
        raise ValueError(
            f'unsupported whole-motor dataset: {index.get("formatVersion")!r}'
        )
    if index.get('viewType') != 'wholeMotor':
        raise ValueError(f'expected wholeMotor view, found {index.get("viewType")!r}')

    loadedCases = []
    seenGroups = set()
    for entry in index.get('cases', ()):
        motorId = str(entry.get('motorId', '')).strip()
        parentCaseId = str(entry.get('parentCaseId', '')).strip()
        if not motorId or not parentCaseId:
            raise ValueError('whole-motor cases require motorId and parentCaseId')
        groupKey = (motorId, parentCaseId)
        if groupKey in seenGroups:
            raise ValueError(
                f'duplicate whole-motor case: motorId={motorId!r}, '
                f'parentCaseId={parentCaseId!r}'
            )
        seenGroups.add(groupKey)

        caseRoot = indexPath.parent / entry['path']
        manifestPath = caseRoot / 'manifest.json'
        manifest = json.loads(manifestPath.read_text(encoding='utf-8'))
        if manifest.get('formatVersion') != WHOLE_MOTOR_FORMAT_VERSION:
            raise ValueError(
                f'unsupported whole-motor case manifest: '
                f'{manifest.get("formatVersion")!r}'
            )
        if manifest.get('motorId') != motorId or manifest.get('parentCaseId') != parentCaseId:
            raise ValueError(
                f'whole-motor index and manifest disagree for {groupKey!r}'
            )

        components = []
        componentPaths = []
        seenComponentIds = set()
        for componentEntry in manifest.get('components', ()):
            componentPath = caseRoot / componentEntry['path']
            component = loadThermalComponentCase(componentPath)
            if component.caseId != componentEntry.get('caseId'):
                raise ValueError(
                    f'whole-motor component reference expected '
                    f'{componentEntry.get("caseId")!r}, loaded {component.caseId!r}'
                )
            if component.caseId in seenComponentIds:
                raise ValueError(
                    f'duplicate component caseId in whole-motor case: '
                    f'{component.caseId!r}'
                )
            seenComponentIds.add(component.caseId)
            components.append(component)
            componentPaths.append(componentPath)
        if not components:
            raise ValueError(f'whole-motor case {groupKey!r} has no components')
        loadedCases.append(
            wholeMotorCase(
                motorId=motorId,
                parentCaseId=parentCaseId,
                components=tuple(components),
                componentPaths=tuple(componentPaths),
                interfaces=tuple(manifest.get('interfaces', ())),
                externalPatches=tuple(manifest.get('externalPatches', ())),
                inputPolicy=dict(manifest.get('inputPolicy', {})),
                summary=dict(manifest.get('summary', {})),
                metadata={**entry.get('metadata', {}), **manifest.get('metadata', {})},
            )
        )
    return wholeMotorDataset(
        sourcePath=indexPath,
        cases=tuple(loadedCases),
        metadata=dict(index.get('metadata', {})),
    )


def saveWholeMotorDataset(
    cases: Sequence[thermalComponentCase],
    directory: str | Path,
    *,
    componentDatasetDirectory: str | Path,
    componentCasePaths: Mapping[str, str | Path] | None = None,
    motorId: str,
    parentCaseId: str,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    append: bool = False,
) -> Path:
    resolvedCases = tuple(cases)
    if not resolvedCases:
        raise ValueError('whole-motor dataset requires at least one component case')
    root = Path(directory)
    caseRoot = root / 'cases' / parentCaseId
    manifestPath = caseRoot / 'manifest.json'
    indexPath = root / 'dataset.json'
    if not overwrite and manifestPath.exists():
        raise FileExistsError(f'whole-motor dataset already exists: {root}')
    if not overwrite and not append and indexPath.exists():
        raise FileExistsError(f'whole-motor dataset already exists: {root}')

    regionToCaseId = {
        str(case.metadata['regionUuid']): case.caseId for case in resolvedCases
    }
    componentEntries = []
    interfaceGroups: dict[str, dict[str, Any]] = {}
    externalPatches = []
    for case in resolvedCases:
        patches = tuple(case.metadata.get('surfacePatches', ()))
        interfacePatchIds = []
        resultDerivedPatchIds = []
        externalPatchIds = []
        for patch in patches:
            record = _patchRecord(case, patch, regionToCaseId)
            if patch.get('boundaryInputType') == 'resultHeatFlux':
                resultDerivedPatchIds.append(int(patch['patchId']))
            interfaceUuid = str(patch.get('interfaceUuid', ''))
            if interfaceUuid:
                interfacePatchIds.append(int(patch['patchId']))
                group = interfaceGroups.setdefault(
                    interfaceUuid,
                    {
                        'interfaceId': interfaceUuid,
                        'interfaceName': str(patch.get('interfaceName', interfaceUuid)),
                        'contactConductanceWPerM2K': float(
                            patch.get('contactConductanceWPerM2K', 0.0)
                        ),
                        'sides': [],
                    },
                )
                group['sides'].append(record)
            else:
                externalPatchIds.append(int(patch['patchId']))
                externalPatches.append(record)
        componentPath = Path(componentDatasetDirectory) / (
            Path(componentCasePaths[case.caseId])
            if componentCasePaths is not None
            else Path('cases') / case.caseId
        )
        componentEntries.append(
            {
                'caseId': case.caseId,
                'componentType': case.componentType,
                'componentInstance': case.metadata.get('componentInstance', case.componentType),
                'geometryId': case.geometryId,
                'regionUuid': case.metadata['regionUuid'],
                'meshUuid': case.metadata['meshUuid'],
                'materialUuid': case.metadata['materialUuid'],
                'path': Path(os.path.relpath(componentPath, start=caseRoot)).as_posix(),
                'interfacePatchIds': sorted(set(interfacePatchIds)),
                'externalPatchIds': sorted(set(externalPatchIds)),
                'resultDerivedPatchIds': sorted(set(resultDerivedPatchIds)),
                'wholeMotorInputMaskPatchIds': sorted(set(resultDerivedPatchIds)),
                'absoluteTemperatureAnchorAvailable': bool(
                    case.metadata['absoluteTemperatureAnchorAvailable']
                ),
                'configuredHeatSourcePowerW': float(
                    case.metadata['configuredHeatSourcePowerW']
                ),
            }
        )

    interfaces = []
    for interfaceId, interface in sorted(interfaceGroups.items()):
        sides = interface['sides']
        if len(sides) > 2:
            raise ValueError(
                f'whole-motor interface {interfaceId!r} has more than two FEM sides'
            )
        heatFlowBalance = float(sum(side['expectedHeatFlowRateW'] for side in sides))
        heatFlowMagnitude = float(
            sum(abs(side['expectedHeatFlowRateW']) for side in sides)
        )
        interfaces.append(
            {
                **interface,
                'isComplete': len(sides) == 2,
                'resultTargetHeatFlowBalanceW': heatFlowBalance,
                'resultTargetRelativeImbalance': (
                    heatFlowBalance / heatFlowMagnitude
                    if heatFlowMagnitude > np.finfo(np.float64).eps
                    else 0.0
                ),
            }
        )

    totalSourcePower = float(
        sum(case.metadata['configuredHeatSourcePowerW'] for case in resolvedCases)
    )
    totalBoundaryPower = float(
        sum(case.metadata['expectedBoundaryHeatFlowRateW'] for case in resolvedCases)
    )
    manifest = {
        'formatVersion': WHOLE_MOTOR_FORMAT_VERSION,
        'motorId': motorId,
        'parentCaseId': parentCaseId,
        'components': componentEntries,
        'interfaces': interfaces,
        'externalPatches': externalPatches,
        'inputPolicy': {
            'componentFields': 'reuse component-case arrays',
            'externalConfiguredBoundaries': 'input',
            'resultDerivedBoundaryFields': 'masked from end-to-end whole-motor inputs',
            'internalInterfaceHeatFlowAndTemperature': 'supervision targets only',
        },
        'wholeMotorInputMask': {
            'meaning': 'mask component boundary samples whose patchId is listed in wholeMotorInputMaskPatchIds',
            'replacement': 'future interface coupling, network boundary model, or explicitly supplied physical boundary input',
        },
        'summary': {
            'componentCount': len(resolvedCases),
            'interfaceCount': len(interfaces),
            'completeInterfaceCount': sum(interface['isComplete'] for interface in interfaces),
            'externalPatchCount': len(externalPatches),
            'configuredHeatSourcePowerW': totalSourcePower,
            'expectedBoundaryHeatFlowRateW': totalBoundaryPower,
            'wholeMotorEnergyResidualW': totalSourcePower + totalBoundaryPower,
        },
        'metadata': _jsonReady(metadata or {}),
    }
    caseRoot.mkdir(parents=True, exist_ok=True)
    manifestPath.write_text(json.dumps(_jsonReady(manifest), indent=2), encoding='utf-8')
    root.mkdir(parents=True, exist_ok=True)
    entry = {
        'motorId': motorId,
        'parentCaseId': parentCaseId,
        'path': (Path('cases') / parentCaseId).as_posix(),
        'metadata': _jsonReady(metadata or {}),
    }
    if append and indexPath.exists():
        index = json.loads(indexPath.read_text(encoding='utf-8'))
        if index.get('formatVersion') != WHOLE_MOTOR_FORMAT_VERSION:
            raise ValueError(f'unsupported whole-motor dataset: {index.get("formatVersion")!r}')
        entries = list(index.get('cases', ()))
        matches = [
            position
            for position, current in enumerate(entries)
            if current.get('motorId') == motorId
            and current.get('parentCaseId') == parentCaseId
        ]
        if matches and not overwrite:
            raise ValueError(
                f'duplicate whole-motor case: motorId={motorId!r}, parentCaseId={parentCaseId!r}'
            )
        for position in reversed(matches):
            entries.pop(position)
        entries.append(entry)
        index['cases'] = entries
    else:
        index = {
            'formatVersion': WHOLE_MOTOR_FORMAT_VERSION,
            'viewType': 'wholeMotor',
            'cases': [entry],
            'metadata': _jsonReady(metadata or {}),
        }
    indexPath.write_text(json.dumps(index, indent=2), encoding='utf-8')
    return indexPath
