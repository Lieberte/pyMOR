from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .motorFemComponentCase import (
    buildMotorFemComponentCase,
    motorFemComponentNames,
)
from .motorFemMetadata import (
    findRegion,
    componentTypeName,
    normalizedComponentName,
    regionName,
    resolveFemSystem,
)
from .motorFemGeometry import (
    boundaryNodeGeometry,
    lumpedNodeVolumes,
    reshapeConnectivity,
    tetrahedronGeometry,
)
from .motorFemBoundary import (
    surfacePatchSamples,
)
from .motorFemIO import (
    elementHeatSources,
    loadH5py,
)
from .motorWholeDataset import saveWholeMotorDataset


POINT_FORMAT_VERSION = 'motor-full-point-dataset-v1'
GRAPH_FORMAT_VERSION = 'motor-full-graph-dataset-v1'
FULL_DATASET_FORMAT_VERSION = 'motor-full-dual-dataset-v1'


def _geometryClassificationTask(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    componentTypes = sorted(
        {
            str(entry['componentType']).strip()
            for entry in entries
            if str(entry.get('componentType', '')).strip()
        }
    )
    return {
        'taskType': 'multiclassClassification',
        'targetField': 'componentType',
        'classes': componentTypes,
    }


def _temporalIndexFields(componentCase) -> dict[str, Any]:
    metadata = componentCase.metadata
    fields = {
        'analysisType': metadata.get('analysisType', 'unknown'),
        'temporalSampleType': metadata.get('temporalSampleType', 'unknownSnapshot'),
    }
    if metadata.get('snapshotTimeS') is not None:
        fields['snapshotTimeS'] = float(metadata['snapshotTimeS'])
    return fields


def _assemblyIndexFields(componentCase) -> dict[str, str]:
    metadata = componentCase.metadata
    return {
        'systemType': str(metadata.get('systemType', 'electric_machine')),
        'assemblyType': str(metadata.get('assemblyType', 'motor')),
        'assemblyId': str(metadata.get('assemblyId') or metadata['motorId']),
    }


def _identifier(value: str, name: str) -> str:
    normalized = re.sub(r'[^A-Za-z0-9_-]+', '-', str(value).strip()).strip('-_')
    if not normalized:
        raise ValueError(f'{name} cannot be normalized to an empty identifier')
    return normalized


def _numericCaseDirectories(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in root.iterdir()
                if path.is_dir() and path.name.replace('.', '', 1).isdigit()
            ),
            key=lambda path: float(path.name),
        )
    )


def _jsonReady(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonReady(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonReady(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'unsupported JSON value: {type(value).__name__}')


def _writeJson(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonReady(value), indent=2), encoding='utf-8')


def _meshData(caseDirectory: Path, component: str):
    h5py = loadH5py()
    control = json.loads((caseDirectory / 'controlData.json').read_text(encoding='utf-8'))
    femSystemUuid, femSystem, femSchema = resolveFemSystem(control)
    regionUuid, region = findRegion(femSystem, component)
    meshUuid = str(region['meshUuid'])
    with h5py.File(caseDirectory / 'mesh' / f'{meshUuid}.samesh', 'r') as meshFile:
        nodes = np.asarray(meshFile['nodes'], dtype=np.float64).reshape(-1, 3)
        tetrahedra = reshapeConnectivity(
            meshFile['connectivityVolume'], 4, 'connectivityVolume'
        ).astype(np.int32, copy=False)
        surfaceTriangles = reshapeConnectivity(
            meshFile['connectivitySurface'], 3, 'connectivitySurface'
        ).astype(np.int32, copy=False)
        elementCenters, elementVolumes = tetrahedronGeometry(nodes, tetrahedra)
        elementHeatSource = elementHeatSources(meshFile, region, elementVolumes)
        componentCenter = np.average(elementCenters, axis=0, weights=elementVolumes)
        boundaryNodes, _, _ = boundaryNodeGeometry(
            nodes, surfaceTriangles, componentCenter
        )
        (
            boundarySampleNodeIndices,
            boundaryNormals,
            boundaryWeights,
            boundaryPatchIds,
            _,
        ) = surfacePatchSamples(
            meshFile,
            region,
            nodes,
            surfaceTriangles,
            componentCenter,
        )
    nodeWeights = lumpedNodeVolumes(nodes.shape[0], tetrahedra, elementVolumes)
    return {
        'femSystemUuid': str(femSystemUuid),
        'femSchema': femSchema,
        'regionUuid': str(regionUuid),
        'regionName': regionName(region),
        'componentInstance': normalizedComponentName(regionName(region)),
        'componentType': componentTypeName(regionName(region)),
        'meshUuid': meshUuid,
        'nodesMeters': nodes,
        'tetrahedra': tetrahedra,
        'surfaceTriangles': surfaceTriangles,
        'elementCentersMeters': elementCenters,
        'elementVolumesM3': elementVolumes,
        'elementHeatSourceWPerM3': elementHeatSource,
        'nodeWeightsM3': nodeWeights,
        'boundaryNodes': boundaryNodes.astype(np.int32, copy=False),
        'boundarySampleNodeIndices': boundarySampleNodeIndices.astype(np.int32, copy=False),
        'boundaryNormals': boundaryNormals,
        'boundaryWeightsM2': boundaryWeights,
        'boundaryPatchIds': boundaryPatchIds.astype(np.int32, copy=False),
    }


def _validateAlignment(componentCase, meshData: dict[str, Any]) -> None:
    np.testing.assert_allclose(
        componentCase.volume.coordinatesMeters,
        meshData['elementCentersMeters'],
    )
    np.testing.assert_allclose(
        componentCase.volume.weightsM3,
        meshData['elementVolumesM3'],
    )
    np.testing.assert_allclose(
        componentCase.boundary.coordinatesMeters,
        meshData['nodesMeters'][meshData['boundarySampleNodeIndices']],
    )
    np.testing.assert_allclose(
        componentCase.boundary.weightsM2,
        meshData['boundaryWeightsM2'],
    )
    np.testing.assert_array_equal(
        componentCase.boundary.patchIds,
        meshData['boundaryPatchIds'],
    )
    positiveNodes = np.flatnonzero(meshData['nodeWeightsM3'] > 0.0)
    np.testing.assert_array_equal(positiveNodes, np.arange(meshData['nodesMeters'].shape[0]))
    np.testing.assert_allclose(
        componentCase.temperature.coordinatesMeters,
        meshData['nodesMeters'],
    )


def _nodeHeatSource(meshData: dict[str, Any]) -> np.ndarray:
    tetrahedra = meshData['tetrahedra']
    elementVolumes = meshData['elementVolumesM3']
    elementSources = meshData['elementHeatSourceWPerM3']
    nodeIntegral = np.zeros(meshData['nodesMeters'].shape[0], dtype=np.float64)
    localIntegral = elementSources * elementVolumes / 4.0
    for localNode in range(4):
        np.add.at(nodeIntegral, tetrahedra[:, localNode], localIntegral)
    return np.divide(
        nodeIntegral,
        meshData['nodeWeightsM3'],
        out=np.zeros_like(nodeIntegral),
        where=meshData['nodeWeightsM3'] > 0.0,
    )


def _caseFields(componentCase, meshData: dict[str, Any]):
    conductivity = componentCase.volume.conductivityWPerMK[0]
    if not np.allclose(componentCase.volume.conductivityWPerMK, conductivity):
        raise NotImplementedError('spatially varying conductivity is not yet supported')
    density = componentCase.volume.densityKgPerM3[0]
    specificHeat = componentCase.volume.specificHeatJPerKgK[0]
    return {
        'conductivityTensorWPerMK': conductivity,
        'densityKgPerM3': np.asarray([density]),
        'specificHeatJPerKgK': np.asarray([specificHeat]),
        'elementHeatSourceWPerM3': componentCase.volume.heatSourceWPerM3,
        'nodeHeatSourceWPerM3': _nodeHeatSource(meshData),
        'boundaryHtcWPerM2K': componentCase.boundary.htcWPerM2K,
        'boundaryReferenceTemperatureK': componentCase.boundary.ambientTemperatureK,
        'boundaryTypeIds': componentCase.boundary.boundaryTypeIds.astype(
            np.int32, copy=False
        ),
        'boundaryHeatFluxWPerM2': componentCase.boundary.heatFluxWPerM2,
        'nodeTemperatureK': componentCase.temperature.temperatureK,
    }


def _saveSelfContainedCase(
    root: Path,
    componentCase,
    meshData: dict[str, Any],
    *,
    formatVersion: str,
) -> dict[str, Any]:
    caseRoot = (
        root
        / 'motors'
        / str(componentCase.metadata['motorId'])
        / 'cases'
        / componentCase.caseId
    )
    caseRoot.mkdir(parents=True, exist_ok=True)
    dataPath = caseRoot / 'data.npz'
    if formatVersion == GRAPH_FORMAT_VERSION:
        geometry = {
            'nodesMeters': meshData['nodesMeters'],
            'tetrahedra': meshData['tetrahedra'],
            'surfaceTriangles': meshData['surfaceTriangles'],
            'elementCentersMeters': meshData['elementCentersMeters'],
            'elementVolumesM3': meshData['elementVolumesM3'],
            'nodeWeightsM3': meshData['nodeWeightsM3'],
            'boundaryNodes': meshData['boundaryNodes'],
            'boundarySampleNodeIndices': meshData['boundarySampleNodeIndices'],
            'boundaryNormals': meshData['boundaryNormals'],
            'boundaryWeightsM2': meshData['boundaryWeightsM2'],
            'boundaryPatchIds': meshData['boundaryPatchIds'],
        }
    else:
        geometry = {
            'volumeCoordinatesMeters': meshData['elementCentersMeters'],
            'volumeWeightsM3': meshData['elementVolumesM3'],
            'boundaryCoordinatesMeters': meshData['nodesMeters'][
                meshData['boundarySampleNodeIndices']
            ],
            'boundaryNormals': meshData['boundaryNormals'],
            'boundaryWeightsM2': meshData['boundaryWeightsM2'],
            'boundaryPatchIds': meshData['boundaryPatchIds'],
            'queryCoordinatesMeters': meshData['nodesMeters'],
            'queryWeightsM3': meshData['nodeWeightsM3'],
        }
    np.savez_compressed(dataPath, **geometry, **_caseFields(componentCase, meshData))
    manifest = {
        'formatVersion': formatVersion,
        'caseId': componentCase.caseId,
        'motorId': componentCase.metadata['motorId'],
        'parentCaseId': componentCase.metadata['parentCaseId'],
        **_assemblyIndexFields(componentCase),
        **_temporalIndexFields(componentCase),
        'componentType': componentCase.componentType,
        'componentInstance': componentCase.metadata['componentInstance'],
        'geometryId': componentCase.geometryId,
        'meshUuid': meshData['meshUuid'],
        'regionUuid': meshData['regionUuid'],
        'dataFile': dataPath.name,
        'selfContained': True,
        'connectivityStored': formatVersion == GRAPH_FORMAT_VERSION,
        'operatingFeatures': componentCase.operatingFeatures,
        'metadata': componentCase.metadata,
    }
    _writeJson(caseRoot / 'manifest.json', manifest)
    return {
        'caseId': componentCase.caseId,
        'motorId': componentCase.metadata['motorId'],
        'parentCaseId': componentCase.metadata['parentCaseId'],
        **_assemblyIndexFields(componentCase),
        **_temporalIndexFields(componentCase),
        'componentType': componentCase.componentType,
        'componentInstance': componentCase.metadata['componentInstance'],
        'geometryId': componentCase.geometryId,
        'meshUuid': meshData['meshUuid'],
        'path': (
            Path('motors')
            / str(componentCase.metadata['motorId'])
            / 'cases'
            / componentCase.caseId
        ).as_posix(),
        'selfContained': True,
        'connectivityStored': formatVersion == GRAPH_FORMAT_VERSION,
    }


def buildFullResolutionMotorDatasetFromDirectories(
    sourceDirectory: str | Path,
    outputDirectory: str | Path,
    *,
    motorId: str,
    caseIds: Sequence[str] | None = None,
    components: Sequence[str] | None = None,
) -> Path:
    sourceRoot = Path(sourceDirectory)
    available = _numericCaseDirectories(sourceRoot)
    selectedNames = (
        tuple(str(value) for value in caseIds)
        if caseIds is not None
        else tuple(path.name for path in available)
    )
    byName = {path.name: path for path in available}
    missing = tuple(name for name in selectedNames if name not in byName)
    if missing:
        raise FileNotFoundError(f'motor case directories are missing: {missing}')
    outputRoot = Path(outputDirectory)
    if outputRoot.exists() and any(outputRoot.iterdir()):
        raise FileExistsError(f'full-resolution motor dataset output is not empty: {outputRoot}')
    outputRoot.mkdir(parents=True, exist_ok=True)
    pointRoot = outputRoot / 'point'
    graphRoot = outputRoot / 'graph'
    resolvedMotorId = _identifier(motorId, 'motorId')
    pointEntries = []
    graphEntries = []
    groupedCases = []

    for caseName in selectedNames:
        caseRoot = byName[caseName]
        selectedComponents = (
            tuple(components) if components is not None else motorFemComponentNames(caseRoot)
        )
        componentCases = []
        for component in selectedComponents:
            componentCase = buildMotorFemComponentCase(
                caseRoot,
                component,
                motorId=resolvedMotorId,
                parentCaseId=caseName,
            )
            meshData = _meshData(caseRoot, component)
            _validateAlignment(componentCase, meshData)
            pointEntries.append(
                _saveSelfContainedCase(
                    pointRoot,
                    componentCase,
                    meshData,
                    formatVersion=POINT_FORMAT_VERSION,
                )
            )
            graphEntries.append(
                _saveSelfContainedCase(
                    graphRoot,
                    componentCase,
                    meshData,
                    formatVersion=GRAPH_FORMAT_VERSION,
                )
            )
            componentCases.append(componentCase)
        groupedCases.append((caseName, tuple(componentCases)))

    sharedMetadata = {
        'motorId': resolvedMotorId,
        'systemType': 'electric_machine',
        'systemTypes': ['electric_machine'],
        'assemblyType': 'motor',
        'assemblyTypes': ['motor'],
        'assemblyId': resolvedMotorId,
        'assemblyIds': [resolvedMotorId],
        'sourceDirectory': str(sourceRoot.resolve()),
        'parentCaseIds': selectedNames,
        'fullResolution': True,
        'randomSubsampling': False,
        'selfContainedCases': True,
        'geometryAssetsSharedByMeshUuid': False,
        'analysisTypes': sorted(
            {
                case.metadata.get('analysisType', 'unknown')
                for _, cases in groupedCases
                for case in cases
            }
        ),
    }
    _writeJson(
        pointRoot / 'dataset.json',
        {
            'formatVersion': POINT_FORMAT_VERSION,
            'viewType': 'fullPoint',
            'connectivityStored': False,
            'selfContainedCases': True,
            'metadata': sharedMetadata,
            'tasks': {
                'geometryComponentClassification': _geometryClassificationTask(
                    pointEntries
                ),
            },
            'cases': pointEntries,
        },
    )
    _writeJson(
        graphRoot / 'dataset.json',
        {
            'formatVersion': GRAPH_FORMAT_VERSION,
            'viewType': 'fullGraph',
            'connectivityStored': True,
            'selfContainedCases': True,
            'metadata': sharedMetadata,
            'tasks': {
                'geometryComponentClassification': _geometryClassificationTask(
                    graphEntries
                ),
            },
            'cases': graphEntries,
        },
    )
    for viewRoot in (pointRoot, graphRoot):
        entries = pointEntries if viewRoot == pointRoot else graphEntries
        casePaths = {entry['caseId']: entry['path'] for entry in entries}
        for caseIndex, (caseName, componentCases) in enumerate(groupedCases):
            saveWholeMotorDataset(
                componentCases,
                viewRoot / 'whole_motor',
                componentDatasetDirectory=viewRoot,
                componentCasePaths=casePaths,
                motorId=resolvedMotorId,
                parentCaseId=caseName,
                metadata=sharedMetadata,
                append=caseIndex > 0,
            )
    viewsPath = outputRoot / 'views.json'
    _writeJson(
        viewsPath,
        {
            'formatVersion': FULL_DATASET_FORMAT_VERSION,
            'motorId': resolvedMotorId,
            'parentCaseIds': selectedNames,
            'fullResolution': True,
            'analysisTypes': sharedMetadata['analysisTypes'],
            'views': {
                'point': 'point/dataset.json',
                'pointWholeMotor': 'point/whole_motor/dataset.json',
                'graph': 'graph/dataset.json',
                'graphWholeMotor': 'graph/whole_motor/dataset.json',
            },
        },
    )
    from .motorTransientDataset import writeMotorThermalAnalysisIndexes

    writeMotorThermalAnalysisIndexes(outputRoot)
    return viewsPath
