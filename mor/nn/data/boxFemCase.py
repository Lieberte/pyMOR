from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .thermalComponentCase import (
    thermalBoundarySamples,
    thermalComponentCase,
    thermalTemperatureSamples,
    thermalVolumeSamples,
)


FIXED_TEMPERATURE_BOUNDARY_TYPE = 1


def _h5py():
    try:
        import h5py
    except ImportError as error:
        raise ImportError('box FEM conversion requires h5py') from error
    return h5py


def _singleFemRegion(control: dict[str, Any]):
    systems = control['rootSystem']['thermalComponents']['fem3DThermals']
    if len(systems) != 1:
        raise ValueError(f'expected one FEM system, found {len(systems)}')
    systemUuid, system = next(iter(systems.items()))
    regions = system['femRegions']
    if len(regions) != 1:
        raise ValueError(f'expected one FEM region, found {len(regions)}')
    regionUuid, region = next(iter(regions.items()))
    return str(systemUuid), system, str(regionUuid), region


def _resultBase(systemUuid: str, regionUuid: str) -> str:
    return f'thermalComponents/thermal3dFem/{systemUuid}/{regionUuid}'


def _meshResultPath(root: Path, base: str) -> Path:
    for path in sorted((root / 'results').rglob('*.h5')):
        with _h5py().File(path, 'r') as file:
            if f'{base}/3dResults/locationIDs/nodes' in file:
                return path
    raise FileNotFoundError('no result file contains embedded FEM mesh location IDs')


def _solutionResultPath(root: Path, resultTime: float) -> Path:
    name = f'{float(resultTime):.6f}.h5'
    matches = tuple((root / 'results').rglob(name))
    if len(matches) != 1:
        raise ValueError(f'expected one result file named {name}, found {len(matches)}')
    return matches[0]


def _tetraGeometry(nodes: np.ndarray, tetrahedra10: np.ndarray):
    cornerNodes = tetrahedra10[:, :4]
    points = nodes[cornerNodes]
    sixVolumes = np.abs(
        np.einsum(
            'ij,ij->i',
            np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
            points[:, 3] - points[:, 0],
        )
    )
    volumes = sixVolumes / 6.0
    if np.any(volumes <= np.finfo(np.float64).eps):
        raise ValueError('embedded quadratic tetrahedra contain degenerate corner geometry')
    return points.mean(axis=1), volumes


def _materialValue(controlFile: Any, materialUuid: str, name: str) -> float:
    values = np.asarray(
        controlFile[f'{materialUuid}/{name}/isotropic/fixedValue'], dtype=np.float64
    ).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]):
        raise ValueError(f'material property {name!r} must contain one finite value')
    return float(values[0])


def _mappedElementPower(controlFile: Any, elementCount: int) -> np.ndarray:
    mappingRoot = controlFile.get('mappingData')
    candidates = []
    if mappingRoot is not None:
        for group in mappingRoot.values():
            for dataset in group.values():
                values = np.asarray(dataset, dtype=np.float64).reshape(-1)
                if values.size == elementCount:
                    candidates.append(values)
    if len(candidates) != 1:
        raise ValueError(
            f'expected one mapped element-power array of length {elementCount}, '
            f'found {len(candidates)}'
        )
    if not np.all(np.isfinite(candidates[0])):
        raise ValueError('mapped element power must contain finite values')
    return candidates[0]


def _fixedTemperatureByPatch(region: dict[str, Any]) -> dict[str, float]:
    result = {}
    for parentPatch in region.get('patches', {}).values():
        for patch in parentPatch.get('children', {}).values():
            constT = patch.get('boundaryConditions', {}).get('constT')
            if constT is None:
                continue
            table = constT.get('constT1D', {})
            if table.get('type') != 'fixedValue':
                raise NotImplementedError('only fixed-temperature box boundaries are supported')
            result[str(patch.get('name'))] = float(table['fixedValue'])
    return result


def _faceName(
    triangleCornerCoordinates: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> str:
    span = np.maximum(maximum - minimum, 1.0)
    tolerance = 1e-8 * float(np.max(span))
    checks = (
        (0, minimum[0], 'left'),
        (0, maximum[0], 'right'),
        (1, minimum[1], 'front'),
        (1, maximum[1], 'back'),
        (2, minimum[2], 'bottom'),
        (2, maximum[2], 'up'),
    )
    for axis, value, name in checks:
        if np.all(np.abs(triangleCornerCoordinates[:, axis] - value) <= tolerance):
            return name
    raise ValueError('could not classify a box boundary triangle onto an axis-aligned face')


def _boxBoundarySamples(
    nodes: np.ndarray,
    boundaryTriangles6: np.ndarray,
    fixedTemperatures: dict[str, float],
):
    minimum = nodes.min(axis=0)
    maximum = nodes.max(axis=0)
    faceOrder = ('left', 'right', 'front', 'back', 'bottom', 'up')
    classified = {name: [] for name in faceOrder}
    for triangleIndex, triangle in enumerate(boundaryTriangles6):
        face = _faceName(nodes[triangle[:3]], minimum, maximum)
        classified[face].append(triangleIndex)

    coordinates = []
    normals = []
    weights = []
    temperatures = []
    patchIds = []
    metadata = []
    faceNormals = {
        'left': np.array([-1.0, 0.0, 0.0]),
        'right': np.array([1.0, 0.0, 0.0]),
        'front': np.array([0.0, -1.0, 0.0]),
        'back': np.array([0.0, 1.0, 0.0]),
        'bottom': np.array([0.0, 0.0, -1.0]),
        'up': np.array([0.0, 0.0, 1.0]),
    }
    for patchId, face in enumerate(faceOrder):
        triangleIndices = np.asarray(classified[face], dtype=np.int64)
        triangles = boundaryTriangles6[triangleIndices]
        corners = nodes[triangles[:, :3]]
        triangleAreas = 0.5 * np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
            axis=1,
        )
        nodalAreas = np.zeros(nodes.shape[0], dtype=np.float64)
        for localNode in range(6):
            np.add.at(nodalAreas, triangles[:, localNode], triangleAreas / 6.0)
        faceNodes = np.flatnonzero(nodalAreas > 0.0)
        faceTemperature = fixedTemperatures[face]
        coordinates.append(nodes[faceNodes])
        normals.append(np.repeat(faceNormals[face][None], faceNodes.size, axis=0))
        weights.append(nodalAreas[faceNodes])
        temperatures.append(np.full(faceNodes.size, faceTemperature))
        patchIds.append(np.full(faceNodes.size, patchId, dtype=np.int64))
        metadata.append(
            {
                'patchId': patchId,
                'patchName': face,
                'surfaceTriangleCount': int(triangleIndices.size),
                'boundarySampleCount': int(faceNodes.size),
                'surfaceAreaM2': float(triangleAreas.sum()),
                'fixedTemperatureK': faceTemperature,
            }
        )
    return (
        np.vstack(coordinates),
        np.vstack(normals),
        np.concatenate(weights),
        np.concatenate(temperatures),
        np.concatenate(patchIds),
        metadata,
    )


def _nodeVolumeAndHeatSource(
    nodeCount: int,
    tetrahedra10: np.ndarray,
    volumes: np.ndarray,
    elementHeatSourceWPerM3: np.ndarray,
):
    weights = np.zeros(nodeCount, dtype=np.float64)
    sourceIntegral = np.zeros(nodeCount, dtype=np.float64)
    for localNode in range(10):
        indices = tetrahedra10[:, localNode]
        localWeights = volumes / 10.0
        np.add.at(weights, indices, localWeights)
        np.add.at(sourceIntegral, indices, elementHeatSourceWPerM3 * localWeights)
    source = np.divide(sourceIntegral, weights, out=np.zeros_like(weights), where=weights > 0.0)
    return weights, source


def _meshEdges(tetrahedra10: np.ndarray) -> np.ndarray:
    localPairs = np.asarray(tuple(combinations(range(10), 2)), dtype=np.int64)
    first = tetrahedra10[:, localPairs[:, 0]].reshape(-1)
    second = tetrahedra10[:, localPairs[:, 1]].reshape(-1)
    undirected = np.column_stack((np.minimum(first, second), np.maximum(first, second)))
    undirected = np.unique(undirected, axis=0)
    directed = np.vstack((undirected, undirected[:, ::-1]))
    return directed.T


@dataclass(frozen=True)
class boxFemCase:
    componentCase: thermalComponentCase
    nodesMeters: np.ndarray
    tetrahedra10: np.ndarray
    boundaryTriangles6: np.ndarray
    edgeIndex: np.ndarray
    nodeWeightsM3: np.ndarray
    nodeHeatSourceWPerM3: np.ndarray
    fixedTemperatureK: np.ndarray
    temperatureK: np.ndarray


def buildBoxFemCase(
    caseDirectory: str | Path,
    *,
    resultTime: float = 1.0,
    maximumVolumeSamples: int | None = 1024,
    randomSeed: int = 0,
) -> boxFemCase:
    h5py = _h5py()
    root = Path(caseDirectory)
    control = json.loads((root / 'controlData.json').read_text(encoding='utf-8'))
    systemUuid, _, regionUuid, region = _singleFemRegion(control)
    base = _resultBase(systemUuid, regionUuid)
    meshResultPath = _meshResultPath(root, base)
    solutionPath = _solutionResultPath(root, resultTime)

    with h5py.File(meshResultPath, 'r') as meshResult:
        meshBase = f'{base}/3dResults/locationIDs'
        nodes = np.asarray(meshResult[f'{meshBase}/nodes'], dtype=np.float64).reshape(-1, 3)
        connectivity = np.asarray(meshResult[f'{meshBase}/connectivity'], dtype=np.int64)
        topology = meshResult[f'{meshBase}/connectivity'].attrs.get('topologyType')
        if isinstance(topology, bytes):
            topology = topology.decode('utf-8')
        if topology != 'TETRAHEDRON_10' or connectivity.size % 10 != 0:
            raise ValueError(f'expected TETRAHEDRON_10 connectivity, found {topology!r}')
        tetrahedra10 = connectivity.reshape(-1, 10)
        boundaryValues = np.asarray(
            meshResult[f'{meshBase}/boundaryNodeIndex'], dtype=np.int64
        )
        if boundaryValues.size % 6 != 0:
            raise ValueError('expected quadratic TRIANGLE_6 boundary connectivity')
        boundaryTriangles6 = boundaryValues.reshape(-1, 6)

    volumeCoordinates, volumeWeights = _tetraGeometry(nodes, tetrahedra10)
    with h5py.File(root / 'controlData.h5', 'r') as controlFile:
        materialUuid = str(region['materialUuid'])
        density = _materialValue(controlFile, materialUuid, 'density')
        specificHeat = _materialValue(controlFile, materialUuid, 'specificHeatCapacity')
        conductivity = _materialValue(controlFile, materialUuid, 'thermalConductivity')
        elementPower = _mappedElementPower(controlFile, tetrahedra10.shape[0])
    elementHeatSource = elementPower / volumeWeights

    fixedTemperatures = _fixedTemperatureByPatch(region)
    (
        boundaryCoordinates,
        boundaryNormals,
        boundaryWeights,
        boundaryTemperatures,
        patchIds,
        patchMetadata,
    ) = _boxBoundarySamples(nodes, boundaryTriangles6, fixedTemperatures)

    with h5py.File(solutionPath, 'r') as solutionFile:
        temperature = np.asarray(
            solutionFile[f'{base}/3dResults/results/temperature'], dtype=np.float64
        ).reshape(-1)
    if temperature.size != nodes.shape[0] or not np.all(np.isfinite(temperature)):
        raise ValueError('solution temperature must contain one finite value per node')

    selection = np.arange(volumeWeights.size)
    selectedWeights = volumeWeights.copy()
    if maximumVolumeSamples is not None and int(maximumVolumeSamples) < selection.size:
        maximum = int(maximumVolumeSamples)
        if maximum <= 0:
            raise ValueError('maximumVolumeSamples must be positive')
        generator = np.random.default_rng(int(randomSeed))
        selection = np.sort(generator.choice(selection.size, size=maximum, replace=False))
        selectedWeights = volumeWeights[selection].copy()
        selectedWeights *= volumeWeights.sum() / selectedWeights.sum()
        sampledPower = float(np.dot(elementHeatSource[selection], selectedWeights))
        if abs(sampledPower) > np.finfo(float).eps:
            elementHeatSource = elementHeatSource.copy()
            elementHeatSource[selection] *= elementPower.sum() / sampledPower

    nodeWeights, nodeHeatSource = _nodeVolumeAndHeatSource(
        nodes.shape[0], tetrahedra10, volumeWeights, elementPower / volumeWeights
    )
    fixedTemperaturePerNode = np.full(nodes.shape[0], np.nan, dtype=np.float64)
    minimum = nodes.min(axis=0)
    maximum = nodes.max(axis=0)
    tolerance = 1e-8 * max(float(np.max(maximum - minimum)), 1.0)
    faceValues = (
        (0, minimum[0], 'left'),
        (0, maximum[0], 'right'),
        (1, minimum[1], 'front'),
        (1, maximum[1], 'back'),
        (2, minimum[2], 'bottom'),
        (2, maximum[2], 'up'),
    )
    for axis, value, face in faceValues:
        mask = np.abs(nodes[:, axis] - value) <= tolerance
        fixedTemperaturePerNode[mask] = fixedTemperatures[face]

    componentCase = thermalComponentCase(
        caseId=f'box-{float(resultTime):g}s',
        geometryId='box-cubic-0.1m',
        componentType='box',
        volume=thermalVolumeSamples(
            coordinatesMeters=volumeCoordinates[selection],
            conductivityWPerMK=np.full(selection.size, conductivity),
            densityKgPerM3=np.full(selection.size, density),
            specificHeatJPerKgK=np.full(selection.size, specificHeat),
            heatSourceWPerM3=elementHeatSource[selection],
            weightsM3=selectedWeights,
        ),
        boundary=thermalBoundarySamples(
            coordinatesMeters=boundaryCoordinates,
            normals=boundaryNormals,
            htcWPerM2K=np.zeros(boundaryCoordinates.shape[0]),
            ambientTemperatureK=boundaryTemperatures,
            weightsM2=boundaryWeights,
            boundaryTypeIds=np.full(
                boundaryCoordinates.shape[0],
                FIXED_TEMPERATURE_BOUNDARY_TYPE,
                dtype=np.int64,
            ),
            heatFluxWPerM2=np.zeros(boundaryCoordinates.shape[0]),
            patchIds=patchIds,
        ),
        temperature=thermalTemperatureSamples(
            coordinatesMeters=nodes,
            temperatureK=temperature,
            weightsM3=nodeWeights,
        ),
        metadata={
            'resultTimeS': float(resultTime),
            'femSystemUuid': systemUuid,
            'regionUuid': regionUuid,
            'regionName': str(region.get('name', '')),
            'topologyType': 'TETRAHEDRON_10',
            'nodeCount': int(nodes.shape[0]),
            'elementCount': int(tetrahedra10.shape[0]),
            'boundaryTriangleCount': int(boundaryTriangles6.shape[0]),
            'surfacePatches': patchMetadata,
            'mappedHeatSourcePowerW': float(elementPower.sum()),
            'sampledHeatSourcePowerW': float(
                np.dot(elementHeatSource[selection], selectedWeights)
            ),
        },
    )
    return boxFemCase(
        componentCase=componentCase,
        nodesMeters=nodes,
        tetrahedra10=tetrahedra10,
        boundaryTriangles6=boundaryTriangles6,
        edgeIndex=_meshEdges(tetrahedra10),
        nodeWeightsM3=nodeWeights,
        nodeHeatSourceWPerM3=nodeHeatSource,
        fixedTemperatureK=fixedTemperaturePerNode,
        temperatureK=temperature,
    )
