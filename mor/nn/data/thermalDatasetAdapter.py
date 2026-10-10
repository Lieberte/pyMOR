from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .motorFemGeometry import (
    matchIntegratedPower,
    sampleByGroupWithConservedWeight,
    sampleWithConservedWeight,
)
from .motorFemBoundary import (
    CONVECTION_BOUNDARY_TYPE,
    FIXED_TEMPERATURE_BOUNDARY_TYPE,
    PRESCRIBED_HEAT_FLUX_BOUNDARY_TYPE,
    RESULT_HEAT_FLUX_BOUNDARY_TYPE,
)
from .motorWholeDataset import (
    WHOLE_MOTOR_FORMAT_VERSION,
    loadWholeMotorDataset,
)
from .thermalComponentCase import (
    loadThermalComponentCase,
    thermalBoundarySamples,
    thermalComponentCase,
    thermalTemperatureSamples,
    thermalVolumeSamples,
)


CANONICAL_THERMAL_BOUNDARY_TYPE_NAMES = {
    RESULT_HEAT_FLUX_BOUNDARY_TYPE: 'resultHeatFlux',
    CONVECTION_BOUNDARY_TYPE: 'convection',
    FIXED_TEMPERATURE_BOUNDARY_TYPE: 'fixedTemperature',
    PRESCRIBED_HEAT_FLUX_BOUNDARY_TYPE: 'prescribedHeatFlux',
}


@dataclass(frozen=True)
class thermalDatasetGroup:
    motorId: str | None
    parentCaseId: str
    caseIds: tuple[str, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class thermalDatasetView:
    sourcePath: Path
    viewType: str
    cases: tuple[thermalComponentCase, ...]
    groups: tuple[thermalDatasetGroup, ...]
    boundaryTypeNames: Mapping[int, str]
    metadata: Mapping[str, Any]


def _normalizedBoundaryName(value: str) -> str:
    normalized = re.sub(r'[^a-z0-9]+', '', str(value).strip().lower())
    aliases = {
        'resultheatflux': 'resultHeatFlux',
        'heatflux': 'resultHeatFlux',
        'flux': 'resultHeatFlux',
        'neumann': 'resultHeatFlux',
        'convection': 'convection',
        'convective': 'convection',
        'robin': 'convection',
        'htc': 'convection',
        'fixedtemperature': 'fixedTemperature',
        'constanttemperature': 'fixedTemperature',
        'constt': 'fixedTemperature',
        'dirichlet': 'fixedTemperature',
        'prescribedheatflux': 'prescribedHeatFlux',
        'prescribedpower': 'prescribedHeatFlux',
        'surfaceheatflux': 'prescribedHeatFlux',
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(f'unsupported thermal boundary type name: {value!r}') from error


def _boundaryTypeNames(index: Mapping[str, Any]) -> dict[int, str]:
    raw = index.get('featureSchema', {}).get('boundaryTypeNames', {})
    return {int(identifier): str(name) for identifier, name in raw.items()}


def _canonicalBoundaryIds(
    case: thermalComponentCase,
    boundaryTypeNames: Mapping[int, str],
) -> tuple[np.ndarray, dict[int, int]]:
    usedIds = set(int(value) for value in np.unique(case.boundary.boundaryTypeIds))
    if not boundaryTypeNames:
        unsupported = usedIds - set(CANONICAL_THERMAL_BOUNDARY_TYPE_NAMES)
        if unsupported:
            raise ValueError(
                f'case {case.caseId!r} uses boundary ids {sorted(unsupported)} '
                'without names describing their physical meaning'
            )
        identity = {identifier: identifier for identifier in usedIds}
        return case.boundary.boundaryTypeIds.copy(), identity

    missing = usedIds - set(boundaryTypeNames)
    if missing:
        raise ValueError(
            f'case {case.caseId!r} uses boundary ids {sorted(missing)} that are '
            'missing from dataset featureSchema.boundaryTypeNames'
        )
    canonicalIdByName = {
        name: identifier
        for identifier, name in CANONICAL_THERMAL_BOUNDARY_TYPE_NAMES.items()
    }
    remapping = {
        sourceId: canonicalIdByName[_normalizedBoundaryName(boundaryTypeNames[sourceId])]
        for sourceId in usedIds
    }
    result = np.asarray(
        [remapping[int(value)] for value in case.boundary.boundaryTypeIds],
        dtype=np.int64,
    )
    return result, remapping


def _canonicalCase(
    case: thermalComponentCase,
    *,
    boundaryTypeNames: Mapping[int, str],
    sourcePath: Path,
    motorId: str | None = None,
    parentCaseId: str | None = None,
) -> thermalComponentCase:
    canonicalIds, remapping = _canonicalBoundaryIds(case, boundaryTypeNames)
    metadata = dict(case.metadata)
    if motorId is not None:
        metadata['motorId'] = motorId
    if parentCaseId is not None:
        metadata['parentCaseId'] = parentCaseId
    metadata['sourceDataset'] = str(sourcePath.resolve())
    patches = []
    for patch in metadata.get('surfacePatches', ()):
        updated = dict(patch)
        if 'boundaryTypeId' in updated:
            sourceId = int(updated['boundaryTypeId'])
            if sourceId not in remapping:
                raise ValueError(
                    f'case {case.caseId!r} patch uses undeclared boundary id '
                    f'{sourceId}'
                )
            updated['boundaryTypeId'] = remapping[sourceId]
        patches.append(updated)
    if patches:
        metadata['surfacePatches'] = patches

    operating = dict(case.operatingFeatures)
    if 'speedRpm' not in operating:
        try:
            operating['speedRpm'] = float(parentCaseId)
        except (TypeError, ValueError):
            operating['speedRpm'] = 0.0
    anchored = bool(
        np.any(canonicalIds == FIXED_TEMPERATURE_BOUNDARY_TYPE)
        or np.any(case.boundary.htcWPerM2K > 0.0)
    )
    operating.setdefault('absoluteTemperatureAnchor', float(anchored))
    boundary = thermalBoundarySamples(
        coordinatesMeters=case.boundary.coordinatesMeters,
        normals=case.boundary.normals,
        htcWPerM2K=case.boundary.htcWPerM2K,
        ambientTemperatureK=case.boundary.ambientTemperatureK,
        weightsM2=case.boundary.weightsM2,
        boundaryTypeIds=canonicalIds,
        heatFluxWPerM2=case.boundary.heatFluxWPerM2,
        patchIds=case.boundary.patchIds,
        extraFeatures=case.boundary.extraFeatures,
    )
    return replace(
        case,
        boundary=boundary,
        operatingFeatures=operating,
        metadata=metadata,
    )


def _componentIndexForCase(casePath: Path) -> tuple[Path, Mapping[str, Any]] | None:
    resolvedCasePath = casePath.resolve()
    for parent in (casePath.parent, *casePath.parents):
        indexPath = parent / 'dataset.json'
        if not indexPath.is_file():
            continue
        index = json.loads(indexPath.read_text(encoding='utf-8'))
        if index.get('formatVersion') == WHOLE_MOTOR_FORMAT_VERSION:
            continue
        for entry in index.get('cases', ()):
            if (parent / entry['path']).resolve() == resolvedCasePath:
                return indexPath, index
    return None


def _loadComponentDataset(indexPath: Path) -> thermalDatasetView:
    index = json.loads(indexPath.read_text(encoding='utf-8'))
    if index.get('formatVersion') == WHOLE_MOTOR_FORMAT_VERSION:
        raise ValueError(f'expected component dataset, found whole-motor view: {indexPath}')
    names = _boundaryTypeNames(index)
    datasetMetadata = dict(index.get('metadata', {}))
    cases = []
    grouped: dict[tuple[str | None, str], list[thermalComponentCase]] = {}
    for entry in index.get('cases', ()):
        raw = loadThermalComponentCase(indexPath.parent / entry['path'])
        rawMotorId = raw.metadata.get('motorId') or datasetMetadata.get('motorId')
        motorId = str(rawMotorId).strip() if rawMotorId not in (None, '') else None
        rawParentId = (
            raw.metadata.get('parentCaseId')
            or datasetMetadata.get('parentCaseId')
            or raw.caseId
        )
        parentCaseId = str(rawParentId).strip()
        case = _canonicalCase(
            raw,
            boundaryTypeNames=names,
            sourcePath=indexPath,
            motorId=motorId,
            parentCaseId=parentCaseId,
        )
        cases.append(case)
        grouped.setdefault((motorId, parentCaseId), []).append(case)
    groups = tuple(
        thermalDatasetGroup(
            motorId=motorId,
            parentCaseId=parentCaseId,
            caseIds=tuple(case.caseId for case in groupCases),
            metadata=dict(datasetMetadata),
        )
        for (motorId, parentCaseId), groupCases in grouped.items()
    )
    return thermalDatasetView(
        sourcePath=indexPath,
        viewType='components',
        cases=tuple(cases),
        groups=groups,
        boundaryTypeNames=dict(CANONICAL_THERMAL_BOUNDARY_TYPE_NAMES),
        metadata=datasetMetadata,
    )


def _loadWholeMotorView(indexPath: Path) -> thermalDatasetView:
    whole = loadWholeMotorDataset(indexPath)
    cases = []
    groups = []
    for group in whole.cases:
        groupCases = []
        for raw, casePath in zip(group.components, group.componentPaths):
            componentIndex = _componentIndexForCase(casePath)
            names = _boundaryTypeNames(componentIndex[1]) if componentIndex else {}
            case = _canonicalCase(
                raw,
                boundaryTypeNames=names,
                sourcePath=indexPath,
                motorId=group.motorId,
                parentCaseId=group.parentCaseId,
            )
            cases.append(case)
            groupCases.append(case)
        groups.append(
            thermalDatasetGroup(
                motorId=group.motorId,
                parentCaseId=group.parentCaseId,
                caseIds=tuple(case.caseId for case in groupCases),
                metadata=dict(group.metadata),
            )
        )
    return thermalDatasetView(
        sourcePath=indexPath,
        viewType='wholeMotor',
        cases=tuple(cases),
        groups=tuple(groups),
        boundaryTypeNames=dict(CANONICAL_THERMAL_BOUNDARY_TYPE_NAMES),
        metadata=dict(whole.metadata),
    )


def loadThermalDatasetAuto(
    source: str | Path,
    *,
    view: str = 'components',
) -> thermalDatasetView:
    sourcePath = Path(source)
    if (
        sourcePath.is_dir()
        and sourcePath.name.lower() == 'cases'
        and (sourcePath.parent / 'dataset.json').is_file()
    ):
        sourcePath = sourcePath.parent
    normalizedView = str(view).strip().lower().replace('-', '')
    if normalizedView not in {'auto', 'components', 'component', 'wholemotor'}:
        raise ValueError("view must be 'auto', 'components', or 'wholeMotor'")

    if sourcePath.is_file():
        indexPath = sourcePath
    else:
        viewsPath = sourcePath / 'views.json'
        if viewsPath.is_file():
            views = json.loads(viewsPath.read_text(encoding='utf-8')).get('views', {})
            if normalizedView == 'wholemotor':
                key = 'wholeMotor' if 'wholeMotor' in views else 'pointWholeMotor'
            else:
                key = 'component' if 'component' in views else 'point'
            indexPath = sourcePath / views[key]
        elif normalizedView == 'wholemotor' and (sourcePath / 'whole_motor' / 'dataset.json').is_file():
            indexPath = sourcePath / 'whole_motor' / 'dataset.json'
        else:
            indexPath = sourcePath / 'dataset.json'
    if not indexPath.is_file():
        raise FileNotFoundError(f'thermal dataset index does not exist: {indexPath}')
    index = json.loads(indexPath.read_text(encoding='utf-8'))
    if index.get('formatVersion') == WHOLE_MOTOR_FORMAT_VERSION:
        return _loadWholeMotorView(indexPath)
    if normalizedView == 'wholemotor':
        sibling = indexPath.parent / 'whole_motor' / 'dataset.json'
        if sibling.is_file():
            return _loadWholeMotorView(sibling)
        raise FileNotFoundError(f'whole-motor view does not exist beside {indexPath}')
    return _loadComponentDataset(indexPath)


def loadThermalTrainingCases(
    sources: Sequence[str | Path] | str | Path,
    *,
    view: str = 'components',
) -> tuple[thermalComponentCase, ...]:
    resolvedSources = (
        (sources,)
        if isinstance(sources, (str, Path))
        else tuple(sources)
    )
    return tuple(
        case
        for source in resolvedSources
        for case in loadThermalDatasetAuto(source, view=view).cases
    )


def splitThermalCasesByMotorId(
    cases: Sequence[thermalComponentCase],
    *,
    validationMotorIds: Sequence[str],
) -> tuple[tuple[thermalComponentCase, ...], tuple[thermalComponentCase, ...]]:
    resolvedCases = tuple(cases)
    validationIds = {str(value).strip() for value in validationMotorIds}
    if not validationIds or any(not value for value in validationIds):
        raise ValueError('validationMotorIds must contain at least one non-empty id')
    motorIds = []
    for case in resolvedCases:
        motorId = case.metadata.get('motorId')
        if motorId in (None, ''):
            raise ValueError(
                f'case {case.caseId!r} does not define metadata.motorId and '
                'cannot participate in a cross-motor split'
            )
        motorIds.append(str(motorId))
    missing = validationIds - set(motorIds)
    if missing:
        raise ValueError(f'validation motor ids are missing from cases: {sorted(missing)}')
    trainCases = tuple(
        case
        for case, motorId in zip(resolvedCases, motorIds)
        if motorId not in validationIds
    )
    validationCases = tuple(
        case for case, motorId in zip(resolvedCases, motorIds) if motorId in validationIds
    )
    if not trainCases:
        raise ValueError('cross-motor split leaves no training cases')
    return trainCases, validationCases


def subsampleThermalComponentCase(
    case: thermalComponentCase,
    *,
    maximumVolumeSamples: int | None = None,
    maximumBoundarySamples: int | None = None,
    maximumQuerySamples: int | None = None,
    randomSeed: int = 0,
) -> thermalComponentCase:
    volumeSelection, volumeWeights = sampleWithConservedWeight(
        case.volume.weightsM3, maximumVolumeSamples, randomSeed
    )
    volumeHeatSource = case.volume.heatSourceWPerM3[volumeSelection].copy()
    matchIntegratedPower(
        volumeHeatSource,
        volumeWeights,
        case.volume.totalHeatSourcePowerW,
    )
    boundarySelection, boundaryWeights = sampleByGroupWithConservedWeight(
        case.boundary.weightsM2,
        case.boundary.patchIds,
        maximumBoundarySamples,
        randomSeed + 1,
    )
    boundaryHeatFlux = case.boundary.heatFluxWPerM2[boundarySelection].copy()
    selectedPatchIds = case.boundary.patchIds[boundarySelection]
    for patchId in np.unique(case.boundary.patchIds):
        fullMask = case.boundary.patchIds == patchId
        selectedMask = selectedPatchIds == patchId
        selectedPositions = np.flatnonzero(selectedMask)
        selectedHeatFlux = boundaryHeatFlux[selectedPositions].copy()
        matchIntegratedPower(
            selectedHeatFlux,
            boundaryWeights[selectedPositions],
            float(
                np.dot(
                    case.boundary.heatFluxWPerM2[fullMask],
                    case.boundary.weightsM2[fullMask],
                )
            ),
        )
        boundaryHeatFlux[selectedPositions] = selectedHeatFlux
    querySelection, queryWeights = sampleWithConservedWeight(
        case.temperature.weightsM3, maximumQuerySamples, randomSeed + 2
    )
    return replace(
        case,
        volume=thermalVolumeSamples(
            coordinatesMeters=case.volume.coordinatesMeters[volumeSelection],
            conductivityWPerMK=case.volume.conductivityWPerMK[volumeSelection],
            densityKgPerM3=case.volume.densityKgPerM3[volumeSelection],
            specificHeatJPerKgK=case.volume.specificHeatJPerKgK[volumeSelection],
            heatSourceWPerM3=volumeHeatSource,
            weightsM3=volumeWeights,
            extraFeatures=case.volume.extraFeatures[volumeSelection],
        ),
        boundary=thermalBoundarySamples(
            coordinatesMeters=case.boundary.coordinatesMeters[boundarySelection],
            normals=case.boundary.normals[boundarySelection],
            htcWPerM2K=case.boundary.htcWPerM2K[boundarySelection],
            ambientTemperatureK=case.boundary.ambientTemperatureK[boundarySelection],
            weightsM2=boundaryWeights,
            boundaryTypeIds=case.boundary.boundaryTypeIds[boundarySelection],
            heatFluxWPerM2=boundaryHeatFlux,
            patchIds=selectedPatchIds,
            extraFeatures=case.boundary.extraFeatures[boundarySelection],
        ),
        temperature=thermalTemperatureSamples(
            coordinatesMeters=case.temperature.coordinatesMeters[querySelection],
            temperatureK=case.temperature.temperatureK[querySelection],
            weightsM3=queryWeights,
        ),
        metadata={
            **case.metadata,
            'trainingSubsample': {
                'originalVolumeSampleCount': len(case.volume.weightsM3),
                'volumeSampleCount': len(volumeSelection),
                'originalBoundarySampleCount': len(case.boundary.weightsM2),
                'boundarySampleCount': len(boundarySelection),
                'originalQuerySampleCount': len(case.temperature.weightsM3),
                'querySampleCount': len(querySelection),
                'randomSeed': int(randomSeed),
            },
        },
    )


def loadSubsampledThermalDataset(
    source: str | Path,
    *,
    analysisType: str | None = None,
    componentTypes: Sequence[str] | None = None,
    maximumVolumeSamples: int | None = None,
    maximumBoundarySamples: int | None = None,
    maximumQuerySamples: int | None = None,
    randomSeed: int = 0,
) -> thermalDatasetView:
    """Load and immediately subsample component cases one at a time.

    Unlike :func:`loadThermalDatasetAuto`, this keeps only the sampled arrays in
    memory. It is intended for full-resolution point datasets whose aggregate
    field files are substantially larger than memory.
    """

    sourcePath = Path(source)
    if sourcePath.is_file():
        indexPath = sourcePath
    else:
        viewsPath = sourcePath / 'views.json'
        if viewsPath.is_file():
            views = json.loads(viewsPath.read_text(encoding='utf-8')).get('views', {})
            analysisKey = (
                None
                if analysisType is None
                else f'point{str(analysisType).strip().capitalize()}'
            )
            key = (
                analysisKey
                if analysisKey in views
                else ('component' if 'component' in views else 'point')
            )
            indexPath = sourcePath / views[key]
        else:
            indexPath = sourcePath / 'dataset.json'
    if not indexPath.is_file():
        raise FileNotFoundError(f'thermal dataset index does not exist: {indexPath}')
    index = json.loads(indexPath.read_text(encoding='utf-8'))
    if index.get('formatVersion') == WHOLE_MOTOR_FORMAT_VERSION:
        raise ValueError('streaming subsampling requires a component dataset')

    selectedComponents = (
        None
        if componentTypes is None
        else frozenset(str(value) for value in componentTypes)
    )
    requestedAnalysis = None if analysisType is None else str(analysisType)
    names = _boundaryTypeNames(index)
    datasetMetadata = dict(index.get('metadata', {}))
    cases = []
    grouped: dict[tuple[str | None, str], list[thermalComponentCase]] = {}
    hierarchyFields = (
        'systemType',
        'assemblyType',
        'assemblyId',
        'componentInstance',
        'analysisType',
        'temporalSampleType',
        'snapshotTimeS',
    )
    for entryIndex, entry in enumerate(index.get('cases', ())):
        entryAnalysis = str(
            entry.get('analysisType', datasetMetadata.get('analysisType', 'steady'))
        )
        if requestedAnalysis is not None and entryAnalysis != requestedAnalysis:
            continue
        entryComponent = str(entry.get('componentType', '')).strip()
        if selectedComponents is not None and entryComponent not in selectedComponents:
            continue

        raw = loadThermalComponentCase(indexPath.parent / entry['path'])
        rawMotorId = (
            raw.metadata.get('motorId')
            or entry.get('motorId')
            or datasetMetadata.get('motorId')
        )
        motorId = str(rawMotorId).strip() if rawMotorId not in (None, '') else None
        rawParentId = (
            raw.metadata.get('parentCaseId')
            or entry.get('parentCaseId')
            or datasetMetadata.get('parentCaseId')
            or raw.caseId
        )
        parentCaseId = str(rawParentId).strip()
        case = _canonicalCase(
            raw,
            boundaryTypeNames=names,
            sourcePath=indexPath,
            motorId=motorId,
            parentCaseId=parentCaseId,
        )
        hierarchy = {
            field: entry[field]
            for field in hierarchyFields
            if entry.get(field) is not None
        }
        case = replace(case, metadata={**case.metadata, **hierarchy})
        case = subsampleThermalComponentCase(
            case,
            maximumVolumeSamples=maximumVolumeSamples,
            maximumBoundarySamples=maximumBoundarySamples,
            maximumQuerySamples=maximumQuerySamples,
            randomSeed=randomSeed + entryIndex,
        )
        cases.append(case)
        grouped.setdefault((motorId, parentCaseId), []).append(case)

    groups = tuple(
        thermalDatasetGroup(
            motorId=motorId,
            parentCaseId=parentCaseId,
            caseIds=tuple(case.caseId for case in groupCases),
            metadata=dict(datasetMetadata),
        )
        for (motorId, parentCaseId), groupCases in grouped.items()
    )
    return thermalDatasetView(
        sourcePath=indexPath,
        viewType='components',
        cases=tuple(cases),
        groups=groups,
        boundaryTypeNames=dict(CANONICAL_THERMAL_BOUNDARY_TYPE_NAMES),
        metadata=datasetMetadata,
    )
