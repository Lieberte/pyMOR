from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .thermalComponentCase import (
    thermalBoundarySamples,
    thermalComponentCase,
    thermalTemperatureSamples,
    thermalVolumeSamples,
)


RESULT_HEAT_FLUX_BOUNDARY_TYPE = 0
CONVECTION_BOUNDARY_TYPE = 1
FIXED_TEMPERATURE_BOUNDARY_TYPE = 2


def _fixedTableValue(table: dict[str, Any], name: str) -> float:
    if table.get('type') != 'fixedValue':
        raise NotImplementedError(f'only fixed-value {name} is currently supported')
    value = float(table.get('fixedValue', 0.0))
    if not np.isfinite(value):
        raise ValueError(f'{name} fixedValue must be finite')
    return value


def _patchBoundaryInput(patch: dict[str, Any]) -> dict[str, Any]:
    conditions = patch.get('boundaryConditions') or {}
    convective = conditions.get('convective')
    if convective is not None:
        htc = _fixedTableValue(convective.get('htc1D', {}), 'convective HTC')
        referenceTemperature = _fixedTableValue(
            convective.get('refT1D', {}), 'convective reference temperature'
        )
        scaleFactor = float(convective.get('scaleFactor', 1.0))
        if not np.isfinite(scaleFactor) or scaleFactor < 0.0:
            raise ValueError('convective scaleFactor must be non-negative and finite')
        return {
            'boundaryInputType': 'convection',
            'boundaryTypeId': CONVECTION_BOUNDARY_TYPE,
            'htcWPerM2K': htc * scaleFactor,
            'ambientTemperatureK': referenceTemperature,
            'absoluteTemperatureAnchor': htc * scaleFactor > 0.0,
        }
    constantTemperature = conditions.get('constT')
    if constantTemperature is not None:
        table = next(
            (
                value
                for key, value in constantTemperature.items()
                if key.lower().endswith('1d') and isinstance(value, dict)
            ),
            constantTemperature,
        )
        temperature = _fixedTableValue(table, 'fixed boundary temperature')
        return {
            'boundaryInputType': 'fixedTemperature',
            'boundaryTypeId': FIXED_TEMPERATURE_BOUNDARY_TYPE,
            'htcWPerM2K': 0.0,
            'ambientTemperatureK': temperature,
            'absoluteTemperatureAnchor': True,
        }
    return {
        'boundaryInputType': 'resultHeatFlux',
        'boundaryTypeId': RESULT_HEAT_FLUX_BOUNDARY_TYPE,
        'htcWPerM2K': 0.0,
        'ambientTemperatureK': 0.0,
        'absoluteTemperatureAnchor': False,
    }


def _h5py():
    try:
        import h5py
    except ImportError as error:
        raise ImportError(
            "motor FEM HDF5 conversion requires h5py; install the 'motor' optional dependency"
        ) from error
    return h5py


def _singleValue(dataset: Any, name: str) -> float:
    values = np.asarray(dataset, dtype=np.float64).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]):
        raise ValueError(f'{name} must contain one finite value')
    return float(values[0])


def _fixedIsotropicMaterialProperty(controlFile: Any, materialUuid: str, name: str) -> float:
    path = f'{materialUuid}/{name}/isotropic/fixedValue'
    if path not in controlFile:
        raise KeyError(f'material property is missing from controlData.h5: {path}')
    return _singleValue(controlFile[path], path)


def _fixedMaterialConductivityTensor(controlFile: Any, materialUuid: str) -> np.ndarray:
    base = f'{materialUuid}/thermalConductivity'
    componentNames = ('XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ')
    componentPaths = tuple(f'{base}/anisotropic/{name}/fixedValue' for name in componentNames)
    if all(path in controlFile for path in componentPaths):
        values = np.asarray(
            [_singleValue(controlFile[path], path) for path in componentPaths],
            dtype=np.float64,
        ).reshape(3, 3)
        if np.any(np.abs(values) > np.finfo(np.float64).eps):
            if not np.allclose(values, values.T):
                raise ValueError('anisotropic thermal conductivity tensor must be symmetric')
            if np.any(np.linalg.eigvalsh(values) <= 0.0):
                raise ValueError(
                    'anisotropic thermal conductivity tensor must be positive definite'
                )
            return values

    principalPaths = tuple(
        f'{base}/orthotropic/propertyX{axis}/fixedValue' for axis in (1, 2, 3)
    )
    directionPaths = tuple(f'{base}/orthotropic/x{axis}' for axis in (1, 2, 3))
    if all(path in controlFile for path in (*principalPaths, *directionPaths)):
        principalValues = np.asarray(
            [_singleValue(controlFile[path], path) for path in principalPaths],
            dtype=np.float64,
        )
        if np.all(principalValues > 0.0):
            directions = np.column_stack(
                [
                    np.asarray(controlFile[path], dtype=np.float64).reshape(-1)
                    for path in directionPaths
                ]
            )
            if directions.shape != (3, 3) or not np.all(np.isfinite(directions)):
                raise ValueError('orthotropic conductivity directions must form a 3x3 matrix')
            if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-8):
                raise ValueError('orthotropic conductivity directions must be orthonormal')
            return directions @ np.diag(principalValues) @ directions.T

    isotropic = _fixedIsotropicMaterialProperty(
        controlFile, materialUuid, 'thermalConductivity'
    )
    if isotropic <= 0.0:
        raise ValueError('isotropic thermal conductivity must be positive')
    return np.eye(3) * isotropic


def _reshapeConnectivity(values: Any, width: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64).reshape(-1)
    if array.size == 0 or array.size % width != 0:
        raise ValueError(f'{name} must contain a non-empty flattened width-{width} array')
    return array.reshape(-1, width)


def _tetrahedronGeometry(
    nodes: np.ndarray, connectivity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points = nodes[connectivity]
    signedSixVolumes = np.einsum(
        'ij,ij->i',
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
        points[:, 3] - points[:, 0],
    )
    volumes = np.abs(signedSixVolumes) / 6.0
    if np.any(volumes <= np.finfo(np.float64).eps):
        raise ValueError('volume connectivity contains degenerate tetrahedra')
    return points.mean(axis=1), volumes


def _lumpedNodeVolumes(
    nodeCount: int, connectivity: np.ndarray, volumes: np.ndarray
) -> np.ndarray:
    weights = np.zeros(nodeCount, dtype=np.float64)
    np.add.at(weights, connectivity.reshape(-1), np.repeat(volumes / 4.0, 4))
    return weights


def _boundaryNodeGeometry(
    nodes: np.ndarray, surfaceConnectivity: np.ndarray, componentCenter: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = nodes[surfaceConnectivity]
    areaVectors = 0.5 * np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    centers = points.mean(axis=1)
    inward = np.einsum('ij,ij->i', areaVectors, centers - componentCenter) < 0.0
    areaVectors[inward] *= -1.0
    areas = np.linalg.norm(areaVectors, axis=1)
    if np.any(areas <= np.finfo(np.float64).eps):
        raise ValueError('surface connectivity contains degenerate triangles')

    nodeAreas = np.zeros(nodes.shape[0], dtype=np.float64)
    nodeAreaVectors = np.zeros_like(nodes, dtype=np.float64)
    for localNode in range(3):
        indices = surfaceConnectivity[:, localNode]
        np.add.at(nodeAreas, indices, areas / 3.0)
        np.add.at(nodeAreaVectors, indices, areaVectors / 3.0)

    boundaryNodes = np.flatnonzero(nodeAreas > 0.0)
    normalVectors = nodeAreaVectors[boundaryNodes]
    normalLengths = np.linalg.norm(normalVectors, axis=1)
    if np.any(normalLengths <= np.finfo(np.float64).eps):
        raise ValueError('could not construct non-zero area-weighted boundary normals')
    normals = normalVectors / normalLengths[:, None]
    return boundaryNodes, normals, nodeAreas[boundaryNodes]


def _surfacePatchSamples(
    meshFile: Any,
    region: dict[str, Any],
    nodes: np.ndarray,
    surfaceConnectivity: np.ndarray,
    componentCenter: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    sourceNodes = []
    normals = []
    weights = []
    patchIds = []
    patchMetadata = []
    triangleAssignments = np.zeros(surfaceConnectivity.shape[0], dtype=np.int64)

    def appendPatch(
        triangleIndices: np.ndarray,
        *,
        patchUuid: str,
        patchName: str,
        parentPatchUuid: str,
        parentPatchName: str,
        patch: dict[str, Any],
    ) -> None:
        patchId = len(patchMetadata)
        patchTriangles = surfaceConnectivity[triangleIndices]
        patchNodes, patchNormals, patchWeights = _boundaryNodeGeometry(
            nodes, patchTriangles, componentCenter
        )
        sourceNodes.append(patchNodes)
        normals.append(patchNormals)
        weights.append(patchWeights)
        patchIds.append(np.full(patchNodes.size, patchId, dtype=np.int64))
        patchMetadata.append(
            {
                'patchId': patchId,
                'patchUuid': patchUuid,
                'patchName': patchName,
                'parentPatchUuid': parentPatchUuid,
                'parentPatchName': parentPatchName,
                'normalBoundaryEnabled': bool(patch.get('normalBoundaryEnabled', False)),
                'networkBoundaryEnabled': bool(patch.get('networkBoundaryEnabled', False)),
                'tccEnabled': bool(patch.get('tccEnabled', False)),
                'surfaceTriangleCount': int(triangleIndices.size),
                'boundarySampleCount': int(patchNodes.size),
                'surfaceAreaM2': float(patchWeights.sum()),
                **_patchBoundaryInput(patch),
            }
        )

    for parentPatchUuid, parentPatch in region.get('patches', {}).items():
        for childUuid, patch in parentPatch.get('children', {}).items():
            datasetPath = f'patches/{parentPatchUuid}/{childUuid}'
            if datasetPath not in meshFile:
                raise KeyError(f'surface patch is missing from mesh: {datasetPath}')
            triangleIndices = np.asarray(meshFile[datasetPath], dtype=np.int64).reshape(-1)
            if (
                triangleIndices.size == 0
                or np.any(triangleIndices < 0)
                or np.any(triangleIndices >= surfaceConnectivity.shape[0])
            ):
                raise ValueError(f'surface patch {childUuid!r} contains invalid triangle indices')
            triangleAssignments[triangleIndices] += 1
            appendPatch(
                triangleIndices,
                patchUuid=str(patch.get('uuid', childUuid)),
                patchName=str(patch.get('name', childUuid)),
                parentPatchUuid=str(parentPatchUuid),
                parentPatchName=str(parentPatch.get('name', parentPatchUuid)),
                patch=patch,
            )

    if not patchMetadata:
        appendPatch(
            np.arange(surfaceConnectivity.shape[0]),
            patchUuid='all-surfaces',
            patchName='all-surfaces',
            parentPatchUuid='all-surfaces',
            parentPatchName='all-surfaces',
            patch={},
        )
        triangleAssignments[:] = 1
    elif np.any(triangleAssignments == 0):
        appendPatch(
            np.flatnonzero(triangleAssignments == 0),
            patchUuid='unassigned-surfaces',
            patchName='unassigned-surfaces',
            parentPatchUuid='unassigned-surfaces',
            parentPatchName='unassigned-surfaces',
            patch={},
        )
        triangleAssignments[triangleAssignments == 0] = 1

    if np.any(triangleAssignments != 1):
        raise ValueError('surface triangles must belong to exactly one surface patch')
    return (
        np.concatenate(sourceNodes),
        np.vstack(normals),
        np.concatenate(weights),
        np.concatenate(patchIds),
        patchMetadata,
    )


def _sampleWithConservedWeight(
    weights: np.ndarray, maximumCount: int | None, randomSeed: int
) -> tuple[np.ndarray, np.ndarray]:
    count = weights.size
    if maximumCount is None or int(maximumCount) >= count:
        return np.arange(count), weights.copy()
    maximum = int(maximumCount)
    if maximum <= 0:
        raise ValueError('sample limits must be positive')
    generator = np.random.default_rng(int(randomSeed))
    indices = np.sort(generator.choice(count, size=maximum, replace=False))
    selectedWeights = weights[indices].copy()
    selectedWeights *= weights.sum() / selectedWeights.sum()
    return indices, selectedWeights


def _sampleByGroupWithConservedWeight(
    weights: np.ndarray,
    groupIds: np.ndarray,
    maximumCount: int | None,
    randomSeed: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = weights.size
    if maximumCount is None or int(maximumCount) >= count:
        return np.arange(count), weights.copy()
    maximum = int(maximumCount)
    uniqueGroupIds, groupCounts = np.unique(groupIds, return_counts=True)
    if maximum < uniqueGroupIds.size:
        raise ValueError(
            'maximum sample count must be at least the number of conserved groups'
        )

    allocations = np.ones(uniqueGroupIds.size, dtype=np.int64)
    remaining = maximum - uniqueGroupIds.size
    available = groupCounts - 1
    if remaining > 0 and available.sum() > 0:
        exact = remaining * available / available.sum()
        extra = np.minimum(np.floor(exact).astype(np.int64), available)
        allocations += extra
        remaining -= int(extra.sum())
        priorities = np.argsort(-(exact - np.floor(exact)))
        while remaining > 0:
            changed = False
            for position in priorities:
                if allocations[position] < groupCounts[position]:
                    allocations[position] += 1
                    remaining -= 1
                    changed = True
                    if remaining == 0:
                        break
            if not changed:
                break

    generator = np.random.default_rng(int(randomSeed))
    selections = []
    selectedWeights = []
    for groupId, allocation in zip(uniqueGroupIds, allocations):
        candidates = np.flatnonzero(groupIds == groupId)
        chosen = np.sort(generator.choice(candidates, size=int(allocation), replace=False))
        patchWeights = weights[chosen].copy()
        patchWeights *= weights[candidates].sum() / patchWeights.sum()
        selections.append(chosen)
        selectedWeights.append(patchWeights)
    selection = np.concatenate(selections)
    order = np.argsort(selection)
    return selection[order], np.concatenate(selectedWeights)[order]


def _matchIntegratedPower(
    heatFluxWPerM2: np.ndarray,
    weightsM2: np.ndarray,
    targetPowerW: float,
) -> float:
    currentPower = float(np.dot(heatFluxWPerM2, weightsM2))
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(targetPowerW))
    if abs(currentPower) > tolerance:
        heatFluxWPerM2 *= targetPowerW / currentPower
    else:
        heatFluxWPerM2 += (targetPowerW - currentPower) / weightsM2.sum()
    return currentPower


def _decodedStrings(values: Any) -> tuple[str, ...]:
    return tuple(
        value.decode('utf-8') if isinstance(value, bytes) else str(value)
        for value in np.asarray(values).reshape(-1)
    )


def _overviewHeatFlowRates(resultFile: Any, overviewBase: str) -> dict[str, float]:
    identifiersPath = f'{overviewBase}/locationIDs/heatFlowRate'
    valuesPath = f'{overviewBase}/results/heatFlowRate'
    if identifiersPath not in resultFile or valuesPath not in resultFile:
        return {}
    identifiers = _decodedStrings(resultFile[identifiersPath])
    values = np.asarray(resultFile[valuesPath], dtype=np.float64).reshape(-1)
    if len(identifiers) != values.size or not np.all(np.isfinite(values)):
        raise ValueError('overview heat-flow identifiers and values must have matching sizes')
    return {identifier: float(value) for identifier, value in zip(identifiers, values)}


def _interfaceMetadataByPatch(
    femSystem: dict[str, Any], regionUuid: str
) -> dict[str, dict[str, Any]]:
    result = {}
    for interfaceUuid, interface in femSystem.get('tccInterfaces', {}).items():
        for sideName, oppositeSideName in (
            ('contactBody1', 'contactBody2'),
            ('contactBody2', 'contactBody1'),
        ):
            side = interface.get(sideName, {})
            if str(side.get('femUuid', '')) != regionUuid:
                continue
            opposite = interface.get(oppositeSideName, {})
            result[str(side.get('patchUuid', ''))] = {
                'interfaceUuid': str(interfaceUuid),
                'interfaceName': str(interface.get('name', interfaceUuid)),
                'oppositeRegionUuid': str(opposite.get('femUuid', '')),
                'oppositePatchUuid': str(opposite.get('patchUuid', '')),
                'contactConductanceWPerM2K': float(
                    interface.get('tccSetting', {}).get('conductance', 0.0)
                ),
                'searchRadiusM': float(interface.get('searchRadius', 0.0)),
            }
    return result


def _regionName(region: dict[str, Any]) -> str:
    return str(region.get('name', '')).strip()


def _normalizedComponentName(name: str) -> str:
    normalized = str(name).strip()
    if normalized.lower().endswith('_mm'):
        normalized = normalized[:-3]
    normalized = re.sub(r'[^A-Za-z0-9_-]+', '_', normalized).strip('_')
    if not normalized:
        raise ValueError('component name cannot be normalized to an empty identifier')
    return normalized.lower()


def _componentTypeName(componentName: str) -> str:
    """Return the semantic class while keeping the FEM region name as its instance ID."""
    normalized = _normalizedComponentName(componentName)
    if re.fullmatch(r'disk(?:[_-]?\d+)?', normalized):
        return 'magnetic_isolation_plate'
    return normalized


def _findRegion(femSystem: dict[str, Any], component: str) -> tuple[str, dict[str, Any]]:
    requested = str(component).strip().lower()
    matches = []
    for regionUuid, region in femSystem.get('femRegions', {}).items():
        name = _regionName(region)
        aliases = {str(regionUuid).lower(), name.lower(), _normalizedComponentName(name)}
        if requested in aliases:
            matches.append((str(regionUuid), region))
    if len(matches) != 1:
        available = sorted(
            _regionName(region) for region in femSystem.get('femRegions', {}).values()
        )
        raise KeyError(
            f'component {component!r} did not match exactly one FEM region; '
            f'available={available}'
        )
    return matches[0]


def motorFemComponentNames(caseDirectory: str | Path) -> tuple[str, ...]:
    controlPath = Path(caseDirectory) / 'controlData.json'
    control = json.loads(controlPath.read_text(encoding='utf-8'))
    femSystems = control['rootSystem']['thermalComponents']['fem3DThermals']
    if len(femSystems) != 1:
        raise ValueError(f'expected exactly one 3D FEM system, found {len(femSystems)}')
    femSystem = next(iter(femSystems.values()))
    names = tuple(
        _normalizedComponentName(_regionName(region))
        for region in femSystem.get('femRegions', {}).values()
    )
    if not names:
        raise ValueError('3D FEM system does not contain any regions')
    if len(set(names)) != len(names):
        raise ValueError(f'normalized FEM component names must be unique: {names}')
    return names


def _resultFile(
    caseDirectory: Path,
    resultTime: float | None = None,
    resultFile: str | Path | None = None,
) -> Path:
    if resultTime is not None and resultFile is not None:
        raise ValueError('resultTime and resultFile cannot both be provided')
    if resultFile is not None:
        path = Path(resultFile)
        if not path.is_absolute():
            path = caseDirectory / path
        if not path.is_file():
            raise FileNotFoundError(f'result HDF5 file does not exist: {path}')
        return path
    if resultTime is not None:
        resultName = f'{float(resultTime):.6f}.h5'
        matches = tuple((caseDirectory / 'results').rglob(resultName))
        if len(matches) != 1:
            raise ValueError(
                f'expected one result HDF5 file named {resultName!r} below '
                f'{caseDirectory / "results"}, found {len(matches)}'
            )
        return matches[0]
    files = tuple((caseDirectory / 'results').rglob('*.h5'))
    if len(files) != 1:
        raise ValueError(
            f'expected one result HDF5 file below {caseDirectory / "results"}, '
            f'found {len(files)}'
        )
    return files[0]


def _fixedHeatSourceValue(group: dict[str, Any]) -> tuple[str, float] | None:
    heatSource = group.get('volumetricGroups', {}).get('heatSource')
    if heatSource is None:
        return None
    table = heatSource.get('heatSource1D', {})
    if table.get('type') != 'fixedValue':
        raise NotImplementedError('only fixed-value steady heat sources are currently supported')
    value = float(table.get('fixedValue', 0.0))
    if not np.isfinite(value):
        raise ValueError('heat-source fixedValue must be finite')
    return str(heatSource.get('inputVariable', 'power')), value


def _elementHeatSources(
    meshFile: Any,
    region: dict[str, Any],
    tetrahedronVolumes: np.ndarray,
) -> np.ndarray:
    sources = np.zeros_like(tetrahedronVolumes)
    for volumePatch in region.get('volumePatches', {}).values():
        for childUuid, group in volumePatch.get('children', {}).items():
            source = _fixedHeatSourceValue(group)
            if source is None:
                continue
            datasetPath = f'volumes/{childUuid}'
            if datasetPath not in meshFile:
                raise KeyError(f'volume group is missing from mesh: {datasetPath}')
            elements = np.asarray(meshFile[datasetPath], dtype=np.int64).reshape(-1)
            if elements.size == 0 or np.any(elements < 0) or np.any(elements >= sources.size):
                raise ValueError(f'volume group {childUuid!r} contains invalid element indices')
            inputVariable, value = source
            if inputVariable == 'power':
                groupVolume = tetrahedronVolumes[elements].sum()
                sources[elements] += value / groupVolume
            elif inputVariable in ('volumetricHeatSource', 'heatSource'):
                sources[elements] += value
            else:
                raise NotImplementedError(
                    f'unsupported heat-source input variable: {inputVariable!r}'
                )
    return sources


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
    femSystems = control['rootSystem']['thermalComponents']['fem3DThermals']
    if len(femSystems) != 1:
        raise ValueError(f'expected exactly one 3D FEM system, found {len(femSystems)}')
    femSystemUuid, femSystem = next(iter(femSystems.items()))
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
        )

    resultBase = (
        f'thermalComponents/thermal3dFem/{femSystemUuid}/{regionUuid}/3dResults/results'
    )
    with h5py.File(resultPath, 'r') as resultFile:
        temperatures = np.asarray(
            resultFile[f'{resultBase}/temperature'], dtype=np.float64
        ).reshape(-1)
        nodalHeatFlow = np.asarray(
            resultFile[f'{resultBase}/heatFlowRate'], dtype=np.float64
        ).reshape(-1)
        overviewBase = (
            f'thermalComponents/thermal3dFem/{femSystemUuid}/{regionUuid}'
            '/overviewResults'
        )
        overviewPath = f'{overviewBase}/results/heatFlowRate'
        overviewBoundaryPower = (
            float(np.asarray(resultFile[overviewPath], dtype=np.float64).sum())
            if overviewPath in resultFile
            else None
        )
        overviewHeatFlowRates = _overviewHeatFlowRates(resultFile, overviewBase)
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
        expectedPower = (
            -float(expectedOverviewPower)
            if expectedOverviewPower is not None
            else globalScaledPatchPower
        )
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
        if patch['boundaryInputType'] != 'resultHeatFlux':
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
    expectedBoundaryPower = (
        -overviewBoundaryPower
        if overviewBoundaryPower is not None
        else exportedBoundaryPower
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
        },
        metadata={
            'motorId': str(motorId).strip() if motorId is not None else None,
            'parentCaseId': resolvedParentCaseId,
            'sourceCase': sourceCaseName,
            'resultTime': resultTime,
            'resultFile': str(resultPath),
            'femSystemUuid': femSystemUuid,
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
            'prescribedBoundaryHeatFlowRateW': float(
                np.dot(selectedBoundaryHeatFlux, selectedBoundaryWeights)
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
        'patch | area[m2] | expected[W] | reconstructed[W] | sampled[W] | error[W]',
    ]
    for patch in patches:
        lines.append(
            '{name} | {area:.9g} | {expected:.9g} | {reconstructed:.9g} | '
            '{sampled:.9g} | {error:.3e}'.format(
                name=patch['patchName'],
                area=float(patch['surfaceAreaM2']),
                expected=float(patch['expectedHeatFlowRateW']),
                reconstructed=float(patch['reconstructedHeatFlowRateW']),
                sampled=float(patch['sampledReconstructedHeatFlowRateW']),
                error=float(patch['sampledHeatFlowErrorW']),
            )
        )
    lines.append(
        'TOTAL | {area:.9g} | {expected:.9g} | {reconstructed:.9g} | '
        '{sampled:.9g} | {error:.3e}'.format(
            area=float(case.boundary.weightsM2.sum()),
            expected=float(case.metadata['expectedBoundaryHeatFlowRateW']),
            reconstructed=float(case.metadata['exportedBoundaryHeatFlowRateW']),
            sampled=case.boundary.totalHeatFlowRateW,
            error=(
                case.boundary.totalHeatFlowRateW
                - float(case.metadata['expectedBoundaryHeatFlowRateW'])
            ),
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
