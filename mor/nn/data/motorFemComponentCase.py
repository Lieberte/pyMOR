from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .thermalComponentCase import (
    thermalBoundarySamples,
    thermalComponentCase,
    thermalTemperatureSamples,
    thermalVolumeSamples,
)


# Preserve existing helper imports while shared implementations live in dedicated modules.
from .motorFemMetadata import (
    activeFemRegions,
    regionName as _regionName,
    normalizedComponentName as _normalizedComponentName,
    componentTypeName as _componentTypeName,
    resolveFemSystem as _resolveFemSystem,
    findRegion as _findRegion,
    thermalAnalysisMetadata as _thermalAnalysisMetadata,
)
from .motorFemGeometry import (
    reshapeConnectivity as _reshapeConnectivity,
    tetrahedronGeometry as _tetrahedronGeometry,
    lumpedNodeVolumes as _lumpedNodeVolumes,
    boundaryNodeGeometry as _boundaryNodeGeometry,
    sampleWithConservedWeight as _sampleWithConservedWeight,
    sampleByGroupWithConservedWeight as _sampleByGroupWithConservedWeight,
    matchIntegratedPower as _matchIntegratedPower,
)
from .motorFemBoundary import (
    fixedTableValue as _fixedTableValue,
    patchBoundaryInput as _patchBoundaryInput,
    networkBoundaryInputs as _networkBoundaryInputs,
    surfacePatchSamples as _surfacePatchSamples,
    CONVECTION_BOUNDARY_TYPE,
    FIXED_TEMPERATURE_BOUNDARY_TYPE,
    PRESCRIBED_HEAT_FLUX_BOUNDARY_TYPE,
    RESULT_HEAT_FLUX_BOUNDARY_TYPE,
)
from .motorFemIO import (
    loadH5py as _h5py,
    singleValue as _singleValue,
    fixedIsotropicMaterialProperty as _fixedIsotropicMaterialProperty,
    fixedMaterialConductivityTensor as _fixedMaterialConductivityTensor,
    decodedStrings as _decodedStrings,
    overviewHeatFlowRates as _overviewHeatFlowRates,
    connectionEndpoint as _connectionEndpoint,
    connectionHeatFlowRates as _connectionHeatFlowRates,
    interfaceMetadataByPatch as _interfaceMetadataByPatch,
    resultFile as _resultFile,
    fixedHeatSourceValue as _fixedHeatSourceValue,
    elementHeatSources as _elementHeatSources,
)


def motorFemComponentNames(caseDirectory: str | Path) -> tuple[str, ...]:
    controlPath = Path(caseDirectory) / 'controlData.json'
    control = json.loads(controlPath.read_text(encoding='utf-8'))
    _, femSystem, _ = _resolveFemSystem(control)
    names = tuple(
        _normalizedComponentName(_regionName(region))
        for region in activeFemRegions(femSystem).values()
    )
    if not names:
        raise ValueError('3D FEM system does not contain any regions')
    if len(set(names)) != len(names):
        raise ValueError(f'normalized FEM component names must be unique: {names}')
    return names


def buildMotorFemComponentCase(
    caseDirectory: str | Path,
    component: str,
    *,
    maximumVolumeSamples: int | None = None,
    maximumBoundarySamples: int | None = None,
    maximumQuerySamples: int | None = None,
    randomSeed: int = 0,
    resultTime: float | None = None,
    resultFile: str | Path | None = None,
    speedRpm: float | None = None,
    motorId: str | None = None,
    parentCaseId: str | None = None,
) -> thermalComponentCase:
    """Convert one solved motor FEM region into an independent supervised component case.

    Explicit fixed-temperature and convection boundaries are preserved from the input
    model. Remaining surfaces are represented by result-derived Neumann samples whose
    area integral reproduces the exported boundary heat-flow sum exactly.
    """

    h5py = _h5py()
    caseRoot = Path(caseDirectory)
    controlJsonPath = caseRoot / 'controlData.json'
    controlH5Path = caseRoot / 'controlData.h5'
    if not controlJsonPath.is_file() or not controlH5Path.is_file():
        raise FileNotFoundError(f'case directory does not contain controlData.json/h5: {caseRoot}')

    control = json.loads(controlJsonPath.read_text(encoding='utf-8'))
    analysisMetadata = _thermalAnalysisMetadata(control, snapshotTimeS=resultTime)
    femSystemUuid, femSystem, femSchema = _resolveFemSystem(control)
    regionUuid, region = _findRegion(femSystem, component)
    componentName = _normalizedComponentName(_regionName(region))
    componentType = _componentTypeName(componentName)
    meshUuid = str(region['meshUuid'])
    materialUuid = str(region['materialUuid'])
    meshPath = caseRoot / 'mesh' / f'{meshUuid}.samesh'
    resultPath = _resultFile(caseRoot, resultTime, resultFile)

    with h5py.File(meshPath, 'r') as meshFile:
        nodes = np.asarray(meshFile['nodes'], dtype=np.float64).reshape(-1, 3)
        tetrahedra = _reshapeConnectivity(
            meshFile['connectivityVolume'], 4, 'connectivityVolume'
        )
        surfaceTriangles = _reshapeConnectivity(
            meshFile['connectivitySurface'], 3, 'connectivitySurface'
        )
        if np.any(tetrahedra < 0) or np.any(tetrahedra >= nodes.shape[0]):
            raise ValueError('volume connectivity contains invalid node indices')
        if np.any(surfaceTriangles < 0) or np.any(surfaceTriangles >= nodes.shape[0]):
            raise ValueError('surface connectivity contains invalid node indices')
        volumeCoordinates, volumeWeights = _tetrahedronGeometry(nodes, tetrahedra)
        heatSources = _elementHeatSources(meshFile, region, volumeWeights)
        componentCenter = np.average(volumeCoordinates, axis=0, weights=volumeWeights)
        boundaryNodes, _, boundaryNodeWeights = _boundaryNodeGeometry(
            nodes, surfaceTriangles, componentCenter
        )
        (
            patchSourceNodes,
            patchBoundaryNormals,
            patchBoundaryWeights,
            patchIds,
            patchMetadata,
        ) = _surfacePatchSamples(
            meshFile,
            region,
            nodes,
            surfaceTriangles,
            componentCenter,
            _networkBoundaryInputs(femSystem, regionUuid),
        )

    resultBase = (
        f'thermalComponents/thermal3dFem/{femSystemUuid}/{regionUuid}/3dResults/results'
        if femSchema == 'nestedFem3DThermal'
        else f'thermalComponents/femRegion/{regionUuid}/results'
    )
    with h5py.File(resultPath, 'r') as resultFile:
        for field in ('temperature', 'heatFlowRate'):
            if f'{resultBase}/{field}' not in resultFile:
                raise ValueError(
                    f'active FEM component {componentName!r} ({regionUuid}) is missing '
                    f'{field} in solved result {resultPath}'
                )
        temperatures = np.asarray(
            resultFile[f'{resultBase}/temperature'], dtype=np.float64
        ).reshape(-1)
        nodalHeatFlow = np.asarray(
            resultFile[f'{resultBase}/heatFlowRate'], dtype=np.float64
        ).reshape(-1)
        overviewBase = (
            f'thermalComponents/thermal3dFem/{femSystemUuid}/{regionUuid}'
            '/overviewResults'
            if femSchema == 'nestedFem3DThermal'
            else f'thermalComponents/femRegion/{regionUuid}/overviewResults'
        )
        overviewPath = f'{overviewBase}/results/heatFlowRate'
        overviewBoundaryPower = (
            float(np.asarray(resultFile[overviewPath], dtype=np.float64).sum())
            if overviewPath in resultFile
            else None
        )
        overviewHeatFlowRates = _overviewHeatFlowRates(resultFile, overviewBase)
        connectionHeatFlowRates = _connectionHeatFlowRates(resultFile, regionUuid)
    if temperatures.size != nodes.shape[0] or nodalHeatFlow.size != nodes.shape[0]:
        raise ValueError('3D FEM result arrays must contain one value per mesh node')
    if not np.all(np.isfinite(temperatures)) or not np.all(np.isfinite(nodalHeatFlow)):
        raise ValueError('3D FEM result arrays must contain finite values')

    nodeVolumeWeights = _lumpedNodeVolumes(nodes.shape[0], tetrahedra, volumeWeights)
    rawBoundaryHeatFlow = nodalHeatFlow[boundaryNodes]
    rawBoundaryPower = float(rawBoundaryHeatFlow.sum())
    boundaryHeatFlowScale = 1.0
    if overviewBoundaryPower is not None and abs(rawBoundaryPower) > np.finfo(float).eps:
        scaleCandidate = -overviewBoundaryPower / rawBoundaryPower
        if 0.25 <= scaleCandidate <= 4.0:
            boundaryHeatFlowScale = scaleCandidate
    nodalBoundaryAreas = np.zeros(nodes.shape[0], dtype=np.float64)
    nodalBoundaryAreas[boundaryNodes] = boundaryNodeWeights
    rawNodalHeatFlux = nodalHeatFlow / np.where(
        nodalBoundaryAreas > 0.0, nodalBoundaryAreas, 1.0
    )
    patchBoundaryHeatFlux = rawNodalHeatFlux[patchSourceNodes] * boundaryHeatFlowScale
    interfaceMetadata = _interfaceMetadataByPatch(femSystem, regionUuid)
    for patch in patchMetadata:
        patchId = patch['patchId']
        mask = patchIds == patchId
        rawPatchPower = float(
            np.dot(rawNodalHeatFlux[patchSourceNodes[mask]], patchBoundaryWeights[mask])
        )
        globalScaledPatchPower = float(
            np.dot(patchBoundaryHeatFlux[mask], patchBoundaryWeights[mask])
        )
        expectedOverviewPower = overviewHeatFlowRates.get(patch['patchName'])
        connectionHeatFlowRate = connectionHeatFlowRates.get(patch['patchUuid'])
        if patch['boundaryInputType'] == 'prescribedHeatFlux':
            expectedPower = float(patch['prescribedHeatFlowRateW'])
            resultHeatFlowSource = 'prescribedInput'
        elif connectionHeatFlowRate is not None:
            expectedPower = float(connectionHeatFlowRate)
            resultHeatFlowSource = 'connectionTable'
        elif expectedOverviewPower is not None:
            expectedPower = -float(expectedOverviewPower)
            resultHeatFlowSource = 'overview'
        else:
            expectedPower = globalScaledPatchPower
            resultHeatFlowSource = 'nodalBoundary'
        calibratedPatchHeatFlux = patchBoundaryHeatFlux[mask].copy()
        _matchIntegratedPower(
            calibratedPatchHeatFlux, patchBoundaryWeights[mask], expectedPower
        )
        patchBoundaryHeatFlux[mask] = calibratedPatchHeatFlux
        reconstructedPower = float(
            np.dot(patchBoundaryHeatFlux[mask], patchBoundaryWeights[mask])
        )
        patch.update(interfaceMetadata.get(patch['patchUuid'], {}))
        patch.update(
            {
                'rawNodalHeatFlowRateW': rawPatchPower,
                'globalScaledHeatFlowRateW': globalScaledPatchPower,
                'expectedHeatFlowRateW': expectedPower,
                'reconstructedHeatFlowRateW': reconstructedPower,
                'heatFlowErrorW': reconstructedPower - expectedPower,
                'heatFlowRelativeError': (
                    (reconstructedPower - expectedPower) / abs(expectedPower)
                    if abs(expectedPower) > np.finfo(float).eps
                    else 0.0
                ),
                'overviewHeatFlowRateW': expectedOverviewPower,
                'connectionHeatFlowRateW': connectionHeatFlowRate,
                'resultHeatFlowSource': resultHeatFlowSource,
                'localHeatFlowScale': (
                    reconstructedPower / rawPatchPower
                    if abs(rawPatchPower) > np.finfo(float).eps
                    else 1.0
                ),
            }
        )

    boundaryTypeIds = np.full(
        patchSourceNodes.size, RESULT_HEAT_FLUX_BOUNDARY_TYPE, dtype=np.int64
    )
    boundaryHtc = np.zeros(patchSourceNodes.size, dtype=np.float64)
    boundaryAmbientTemperature = np.zeros(patchSourceNodes.size, dtype=np.float64)
    boundaryInputHeatFlux = patchBoundaryHeatFlux.copy()
    for patch in patchMetadata:
        mask = patchIds == patch['patchId']
        boundaryTypeIds[mask] = int(patch['boundaryTypeId'])
        boundaryHtc[mask] = float(patch['htcWPerM2K'])
        boundaryAmbientTemperature[mask] = float(patch['ambientTemperatureK'])
        if patch['boundaryInputType'] == 'prescribedHeatFlux':
            boundaryInputHeatFlux[mask] = float(patch['prescribedHeatFluxWPerM2'])
        elif patch['boundaryInputType'] != 'resultHeatFlux':
            boundaryInputHeatFlux[mask] = 0.0

    with h5py.File(controlH5Path, 'r') as controlFile:
        density = _fixedIsotropicMaterialProperty(
            controlFile, materialUuid, 'density'
        )
        specificHeat = _fixedIsotropicMaterialProperty(
            controlFile, materialUuid, 'specificHeatCapacity'
        )
        conductivity = _fixedMaterialConductivityTensor(controlFile, materialUuid)

    _, heatSourceGroupIds = np.unique(heatSources, return_inverse=True)
    volumeSelection, selectedVolumeWeights = _sampleByGroupWithConservedWeight(
        volumeWeights, heatSourceGroupIds, maximumVolumeSamples, randomSeed
    )
    boundarySelection, selectedBoundaryWeights = _sampleByGroupWithConservedWeight(
        patchBoundaryWeights, patchIds, maximumBoundarySamples, randomSeed + 1
    )
    selectedResultHeatFlux = patchBoundaryHeatFlux[boundarySelection].copy()
    selectedPatchIds = patchIds[boundarySelection]
    exportedBoundaryPower = float(
        np.dot(patchBoundaryHeatFlux, patchBoundaryWeights)
    )
    for patch in patchMetadata:
        patchId = patch['patchId']
        fullMask = patchIds == patchId
        selectedMask = selectedPatchIds == patchId
        targetPower = float(
            np.dot(patchBoundaryHeatFlux[fullMask], patchBoundaryWeights[fullMask])
        )
        sampledPatchHeatFlux = selectedResultHeatFlux[selectedMask].copy()
        _matchIntegratedPower(
            sampledPatchHeatFlux,
            selectedBoundaryWeights[selectedMask],
            targetPower,
        )
        selectedResultHeatFlux[selectedMask] = sampledPatchHeatFlux
        sampledPower = float(
            np.dot(
                selectedResultHeatFlux[selectedMask],
                selectedBoundaryWeights[selectedMask],
            )
        )
        patch['sampledBoundarySampleCount'] = int(np.count_nonzero(selectedMask))
        patch['sampledSurfaceAreaM2'] = float(selectedBoundaryWeights[selectedMask].sum())
        patch['sampledReconstructedHeatFlowRateW'] = sampledPower
        patch['sampledHeatFlowErrorW'] = sampledPower - targetPower
    selectedBoundaryTypes = boundaryTypeIds[boundarySelection]
    selectedBoundaryHeatFlux = boundaryInputHeatFlux[boundarySelection].copy()
    resultFluxMask = selectedBoundaryTypes == RESULT_HEAT_FLUX_BOUNDARY_TYPE
    selectedBoundaryHeatFlux[resultFluxMask] = selectedResultHeatFlux[resultFluxMask]
    prescribedFluxMask = (
        selectedBoundaryTypes == PRESCRIBED_HEAT_FLUX_BOUNDARY_TYPE
    )
    for patch in patchMetadata:
        selectedMask = selectedPatchIds == int(patch['patchId'])
        patch['sampledBoundaryInputHeatFlowRateW'] = float(
            np.dot(
                selectedBoundaryHeatFlux[selectedMask],
                selectedBoundaryWeights[selectedMask],
            )
        )
    queryCandidates = np.flatnonzero(nodeVolumeWeights > 0.0)
    querySelection, selectedQueryWeights = _sampleWithConservedWeight(
        nodeVolumeWeights[queryCandidates], maximumQuerySamples, randomSeed + 2
    )
    selectedQueryNodes = queryCandidates[querySelection]

    sourceCaseName = caseRoot.name
    resolvedSpeedRpm = None
    if speedRpm is not None:
        resolvedSpeedRpm = float(speedRpm)
        if not np.isfinite(resolvedSpeedRpm):
            raise ValueError('speedRpm must be finite')
    elif parentCaseId is None:
        try:
            resolvedSpeedRpm = float(sourceCaseName)
        except ValueError as error:
            raise ValueError(
                f'case directory name must be a numeric speed in rpm when speedRpm '
                f'and parentCaseId are not provided: {sourceCaseName!r}'
            ) from error

    resolvedParentCaseId = (
        str(parentCaseId).strip() if parentCaseId is not None else f'{resolvedSpeedRpm:g}rpm'
    )
    if not resolvedParentCaseId or any(character in resolvedParentCaseId for character in '/\\'):
        raise ValueError('parentCaseId must be non-empty and cannot contain path separators')

    configuredPower = float(np.dot(heatSources, volumeWeights))
    resultDerivedBoundaryPower = float(
        np.dot(
            selectedBoundaryHeatFlux[~prescribedFluxMask],
            selectedBoundaryWeights[~prescribedFluxMask],
        )
    )
    prescribedBoundaryPower = float(
        np.dot(
            selectedBoundaryHeatFlux[prescribedFluxMask],
            selectedBoundaryWeights[prescribedFluxMask],
        )
    )
    expectedBoundaryPower = (
        -overviewBoundaryPower
        if overviewBoundaryPower is not None
        else exportedBoundaryPower
    )
    hybridBoundaryPower = resultDerivedBoundaryPower + prescribedBoundaryPower
    steadyEnergyResidual = configuredPower + hybridBoundaryPower
    steadyEnergyScale = max(
        abs(configuredPower), abs(hybridBoundaryPower), np.finfo(np.float64).eps
    )
    return thermalComponentCase(
        caseId=f'{resolvedParentCaseId}-{componentName}',
        geometryId=f'{componentName}-{meshUuid}',
        componentType=componentType,
        volume=thermalVolumeSamples(
            coordinatesMeters=volumeCoordinates[volumeSelection],
            conductivityWPerMK=conductivity,
            densityKgPerM3=np.full(volumeSelection.size, density),
            specificHeatJPerKgK=np.full(volumeSelection.size, specificHeat),
            heatSourceWPerM3=heatSources[volumeSelection],
            weightsM3=selectedVolumeWeights,
        ),
        boundary=thermalBoundarySamples(
            coordinatesMeters=nodes[patchSourceNodes[boundarySelection]],
            normals=patchBoundaryNormals[boundarySelection],
            htcWPerM2K=boundaryHtc[boundarySelection],
            ambientTemperatureK=boundaryAmbientTemperature[boundarySelection],
            weightsM2=selectedBoundaryWeights,
            boundaryTypeIds=selectedBoundaryTypes,
            heatFluxWPerM2=selectedBoundaryHeatFlux,
            patchIds=selectedPatchIds,
        ),
        temperature=thermalTemperatureSamples(
            coordinatesMeters=nodes[selectedQueryNodes],
            temperatureK=temperatures[selectedQueryNodes],
            weightsM3=selectedQueryWeights,
        ),
        operatingFeatures={
            **(
                {'speedRpm': resolvedSpeedRpm}
                if resolvedSpeedRpm is not None
                else {}
            ),
            'absoluteTemperatureAnchor': float(
                any(patch['absoluteTemperatureAnchor'] for patch in patchMetadata)
            ),
            **(
                {'snapshotTimeS': analysisMetadata['snapshotTimeS']}
                if 'snapshotTimeS' in analysisMetadata
                else {}
            ),
        },
        metadata={
            **analysisMetadata,
            'motorId': str(motorId).strip() if motorId is not None else None,
            'systemType': 'electric_machine',
            'assemblyType': 'motor',
            'assemblyId': str(motorId).strip() if motorId is not None else None,
            'parentCaseId': resolvedParentCaseId,
            'sourceCase': sourceCaseName,
            'resultTime': resultTime,
            'resultFile': str(resultPath),
            'femSystemUuid': femSystemUuid,
            'femSchema': femSchema,
            'regionUuid': regionUuid,
            'regionName': _regionName(region),
            'componentInstance': componentName,
            'meshUuid': meshUuid,
            'materialUuid': materialUuid,
            'conductivityTensorWPerMK': conductivity,
            'fullNodeCount': int(nodes.shape[0]),
            'fullTetrahedronCount': int(tetrahedra.shape[0]),
            'fullBoundaryNodeCount': int(boundaryNodes.size),
            'fullBoundarySampleCount': int(patchSourceNodes.size),
            'surfacePatchCount': int(len(patchMetadata)),
            'sampledBoundaryNodeCount': int(boundarySelection.size),
            'configuredHeatSourcePowerW': configuredPower,
            'exportedBoundaryHeatFlowRateW': exportedBoundaryPower,
            'expectedBoundaryHeatFlowRateW': expectedBoundaryPower,
            'boundaryHeatFlowErrorW': exportedBoundaryPower - expectedBoundaryPower,
            'rawExportedNodalBoundaryHeatFlowRateW': rawBoundaryPower,
            'overviewBoundaryHeatFlowRateW': overviewBoundaryPower,
            'boundaryHeatFlowScale': boundaryHeatFlowScale,
            # Keep the result-derived and explicit-input contributions separate.
            # ``boundary.totalHeatFlowRateW`` is the hybrid boundary input used by
            # the learner, while ``expectedBoundaryHeatFlowRateW`` refers to the
            # solved-result overview target and therefore must not be compared to
            # the hybrid value when a FEM-network power input is present.
            'resultDerivedBoundaryHeatFlowRateW': resultDerivedBoundaryPower,
            'prescribedBoundaryHeatFlowRateW': prescribedBoundaryPower,
            'hybridBoundaryInputHeatFlowRateW': hybridBoundaryPower,
            'connectionBoundaryHeatFlowRateW': float(
                sum(connectionHeatFlowRates.values())
            ),
            'connectionBoundaryPatchCount': int(len(connectionHeatFlowRates)),
            'steadyEnergyResidualW': steadyEnergyResidual,
            'steadyEnergyRelativeResidual': abs(steadyEnergyResidual) / steadyEnergyScale,
            'prescribedSurfaceHeatSourcePowerW': float(
                sum(
                    patch.get('prescribedHeatFlowRateW', 0.0)
                    for patch in patchMetadata
                )
            ),
            'absoluteTemperatureAnchorAvailable': bool(
                any(patch['absoluteTemperatureAnchor'] for patch in patchMetadata)
            ),
            'boundaryRepresentation': (
                'hybrid explicit convection/fixed-temperature inputs plus '
                'result-derived heat flux on remaining surfaces'
            ),
            'volumeRepresentation': 'pointwise volumetric heat source density with volume weights',
            'surfacePatches': patchMetadata,
            'resultDerivedBoundary': bool(
                any(
                    patch['boundaryInputType'] == 'resultHeatFlux'
                    for patch in patchMetadata
                )
            ),
        },
    )


def formatMotorFemBoundaryHeatFlowReport(case: thermalComponentCase) -> str:
    """Format per-surface area and heat-flow conservation checks for one FEM case."""

    patches = case.metadata.get('surfacePatches', [])
    lines = [
        f'case={case.caseId} component={case.componentType}',
        'patch | inputType | area[m2] | resultExpected[W] | '
        'resultReconstructed[W] | datasetInput[W]',
    ]
    for patch in patches:
        lines.append(
            '{name} | {inputType} | {area:.9g} | {expected:.9g} | '
            '{reconstructed:.9g} | {datasetInput:.9g}'.format(
                name=patch['patchName'],
                inputType=patch['boundaryInputType'],
                area=float(patch['surfaceAreaM2']),
                expected=float(patch['expectedHeatFlowRateW']),
                reconstructed=float(patch['sampledReconstructedHeatFlowRateW']),
                datasetInput=float(patch['sampledBoundaryInputHeatFlowRateW']),
            )
        )
    lines.append(
        'RESULT TOTAL | area={area:.9g} m2 | expected={expected:.9g} W | '
        'reconstructed={reconstructed:.9g} W | error={error:.3e} W'.format(
            area=float(case.boundary.weightsM2.sum()),
            expected=float(case.metadata['expectedBoundaryHeatFlowRateW']),
            reconstructed=float(case.metadata['exportedBoundaryHeatFlowRateW']),
            error=(
                float(case.metadata['exportedBoundaryHeatFlowRateW'])
                - float(case.metadata['expectedBoundaryHeatFlowRateW'])
            ),
        )
    )
    lines.append(
        'DATASET INPUT TOTAL | hybrid={hybrid:.9g} W | prescribedSurfaceSource='
        '{prescribed:.9g} W'.format(
            hybrid=float(case.metadata['hybridBoundaryInputHeatFlowRateW']),
            prescribed=float(case.metadata['prescribedSurfaceHeatSourcePowerW']),
        )
    )
    lines.append(
        'VOLUME SOURCE | integrated={:.9g} W'.format(
            case.volume.totalHeatSourcePowerW
        )
    )
    return '\n'.join(lines)


def buildMotorFemComponentCases(
    datasetDirectory: str | Path,
    component: str,
    *,
    speeds: Sequence[str | int] | None = None,
    maximumVolumeSamples: int | None = None,
    maximumBoundarySamples: int | None = None,
    maximumQuerySamples: int | None = None,
    randomSeed: int = 0,
) -> tuple[thermalComponentCase, ...]:
    root = Path(datasetDirectory)
    numericDirectories = tuple(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.replace('.', '', 1).isdigit()
    )
    selectedSpeeds = (
        tuple(str(value) for value in speeds)
        if speeds is not None
        else tuple(
            path.name
            for path in sorted(numericDirectories, key=lambda item: float(item.name))
        )
    )
    if not selectedSpeeds:
        raise ValueError(f'no speed case directories were selected below {root}')
    return tuple(
        buildMotorFemComponentCase(
            root / speed,
            component,
            maximumVolumeSamples=maximumVolumeSamples,
            maximumBoundarySamples=maximumBoundarySamples,
            maximumQuerySamples=maximumQuerySamples,
            randomSeed=randomSeed,
        )
        for speed in selectedSpeeds
    )
