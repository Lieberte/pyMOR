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
PRESCRIBED_HEAT_FLUX_BOUNDARY_TYPE = 3


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
    heatFlux = conditions.get('heatFlux')
    if heatFlux is not None:
        inputVariable = str(heatFlux.get('inputVariable', 'heatFlux'))
        if inputVariable not in ('heatFlux', 'power'):
            raise NotImplementedError(
                f'unsupported prescribed heat-flux input variable: {inputVariable!r}'
            )
        value = _fixedTableValue(
            heatFlux.get('heatFlux1D', {}), 'prescribed heat flux'
        )
        return {
            'boundaryInputType': 'prescribedHeatFlux',
            'boundaryTypeId': PRESCRIBED_HEAT_FLUX_BOUNDARY_TYPE,
            'htcWPerM2K': 0.0,
            'ambientTemperatureK': 0.0,
            'absoluteTemperatureAnchor': False,
            'prescribedHeatFluxInputVariable': inputVariable,
            'prescribedHeatFluxValue': value,
        }
    return {
        'boundaryInputType': 'resultHeatFlux',
        'boundaryTypeId': RESULT_HEAT_FLUX_BOUNDARY_TYPE,
        'htcWPerM2K': 0.0,
        'ambientTemperatureK': 0.0,
        'absoluteTemperatureAnchor': False,
    }


def _networkBoundaryInputs(
    femSystem: dict[str, Any], regionUuid: str
) -> dict[str, dict[str, Any]]:
    """Return explicit heat-flux inputs declared on FEM-network interfaces."""
    overrides: dict[str, dict[str, Any]] = {}
    for interfaceUuid, interface in femSystem.get('femNetworkInterfaces', {}).items():
        if not interface.get('isActive', True):
            continue
        fem = interface.get('fem', {})
        if str(fem.get('femUuid', '')) != str(regionUuid):
            continue
        if not bool(fem.get('isHeatFluxInput', False)):
            continue
        heatFlux = fem.get('heatFlux') or {}
        inputVariable = str(heatFlux.get('inputVariable', 'heatFlux'))
        if inputVariable not in ('heatFlux', 'power'):
            raise NotImplementedError(
                f'unsupported FEM-network heat-flux input variable: {inputVariable!r}'
            )
        value = _fixedTableValue(
            heatFlux.get('heatFlux1D', {}), 'FEM-network prescribed heat flux'
        )
        patchUuid = str(fem.get('patchUuid', '')).strip()
        if not patchUuid:
            raise ValueError(
                f'FEM-network interface {interfaceUuid!r} has no FEM patch UUID'
            )
        if patchUuid in overrides:
            raise ValueError(
                f'multiple FEM-network heat-flux inputs target patch {patchUuid!r}'
            )
        overrides[patchUuid] = {
            'boundaryInputType': 'prescribedHeatFlux',
            'boundaryTypeId': PRESCRIBED_HEAT_FLUX_BOUNDARY_TYPE,
            'htcWPerM2K': 0.0,
            'ambientTemperatureK': 0.0,
            'absoluteTemperatureAnchor': False,
            'prescribedHeatFluxInputVariable': inputVariable,
            'prescribedHeatFluxValue': value,
            'networkInterfaceUuid': str(interfaceUuid),
            'networkBoundaryEnabled': True,
        }
    return overrides


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
    cancelled = normalLengths <= np.finfo(np.float64).eps
    for position in np.flatnonzero(cancelled):
        node = boundaryNodes[position]
        adjacent = np.flatnonzero(np.any(surfaceConnectivity == node, axis=1))
        if adjacent.size:
            candidate = adjacent[np.argmax(areas[adjacent])]
            normalVectors[position] = areaVectors[candidate]
            normalLengths[position] = areas[candidate]
    if np.any(normalLengths <= np.finfo(np.float64).eps):
        raise ValueError('could not construct non-zero boundary normals')
    normals = normalVectors / normalLengths[:, None]
    return boundaryNodes, normals, nodeAreas[boundaryNodes]


def _surfacePatchSamples(
    meshFile: Any,
    region: dict[str, Any],
    nodes: np.ndarray,
    surfaceConnectivity: np.ndarray,
    componentCenter: np.ndarray,
    boundaryInputOverrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    sourceNodes = []
    normals = []
    weights = []
    patchIds = []
    patchMetadata = []
    triangleAssignments = np.zeros(surfaceConnectivity.shape[0], dtype=np.int64)
    overrides = boundaryInputOverrides or {}
    appliedOverrides: set[str] = set()

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
        boundaryInput = _patchBoundaryInput(patch)
        if patchUuid in overrides:
            boundaryInput = {**boundaryInput, **overrides[patchUuid]}
            appliedOverrides.add(patchUuid)
        area = float(patchWeights.sum())
        if boundaryInput.get('boundaryInputType') == 'prescribedHeatFlux':
            value = float(boundaryInput['prescribedHeatFluxValue'])
            inputVariable = boundaryInput['prescribedHeatFluxInputVariable']
            prescribedFlux = value / area if inputVariable == 'power' else value
            boundaryInput.update(
                {
                    'prescribedHeatFluxWPerM2': prescribedFlux,
                    'prescribedHeatFlowRateW': prescribedFlux * area,
                    'prescribedSurfaceHeatSourcePowerW': prescribedFlux * area,
                }
            )
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
                'surfaceAreaM2': area,
                **boundaryInput,
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
    missingOverrides = set(overrides) - appliedOverrides
    if missingOverrides:
        raise KeyError(
            f'FEM-network heat-flux inputs reference unknown patches: {sorted(missingOverrides)}'
        )
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


def _connectionEndpoint(value: Any) -> tuple[str, str]:
    decoded = (
        bytes(value).decode('utf-8', errors='replace')
        if isinstance(value, (bytes, np.bytes_))
        else str(value)
    )
    componentUuid, separator, remainder = decoded.partition('@')
    patchUuid = remainder.split('&', 1)[0] if separator else ''
    return componentUuid, patchUuid


def _connectionHeatFlowRates(resultFile: Any, regionUuid: str) -> dict[str, float]:
    """Return signed solver-connection heat flow for each patch of one FEM region.

    The result connection table stores ``left endpoint, right endpoint, power``
    triples. Its sign convention is positive from left to right, so heat entering
    the left component is negative and heat entering the right component is
    positive. This is the same sign convention used by the component PDE balance,
    ``volume source + boundary heat flow = 0``.
    """

    path = 'connections/heatFlowRate'
    if path not in resultFile:
        return {}
    values = np.asarray(resultFile[path]).reshape(-1)
    if values.size % 3:
        raise ValueError(
            'connections/heatFlowRate must contain endpoint pairs and heat-flow values'
        )
    result: dict[str, float] = {}
    for leftRaw, rightRaw, powerRaw in values.reshape(-1, 3):
        leftComponent, leftPatch = _connectionEndpoint(leftRaw)
        rightComponent, rightPatch = _connectionEndpoint(rightRaw)
        decodedPower = (
            bytes(powerRaw).decode('utf-8', errors='replace')
            if isinstance(powerRaw, (bytes, np.bytes_))
            else str(powerRaw)
        )
        power = float(decodedPower)
        if not np.isfinite(power):
            raise ValueError('connections/heatFlowRate must contain finite values')
        if leftComponent == str(regionUuid) and leftPatch:
            result[leftPatch] = result.get(leftPatch, 0.0) - power
        if rightComponent == str(regionUuid) and rightPatch:
            result[rightPatch] = result.get(rightPatch, 0.0) + power
    return result


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


_COMPONENT_INSTANCE_ALIASES = {
    '转子铁芯': 'rotor_core',
    '端板': 'end_plate',
    '端板1': 'end_plate_1',
    '端板2': 'end_plate_2',
    '轴承外圈': 'bearing_outer_ring',
    '轴承内圈': 'bearing_inner_ring',
    '轴承支架': 'bearing_support',
    '转轴': 'shaft',
    '球': 'bearing_ball',
    '绕组槽部': 'winding_slot',
    '绕组皇冠端': 'winding_crown_end',
    '绕组焊接端': 'winding_weld_end',
    '定子铁芯': 'stator',
    '轴': 'shaft',
    '隔磁板': 'magnetic_isolation_plate',
    '隔磁板1': 'magnetic_isolation_plate_1',
    '隔磁板2': 'magnetic_isolation_plate_2',
    '油箱': 'oil_tank',
    '机箱': 'housing',
    '轴承': 'bearing',
    '磁钢': 'magnets',
    '端盖': 'end_cap',
    '阻尼条': 'damper_bar',
    '行星轮': 'planet_gear',
    '支架': 'carrier',
    '支架轴承': 'carrier_bearing',
    '齿圈': 'ring_gear',
    '外壳': 'housing',
}

_NUMBERED_COMPONENT_INSTANCE_ALIASES = {
    '端板': 'end_plate',
    '轴承外圈': 'bearing_outer_ring',
    '隔磁板': 'magnetic_isolation_plate',
    '轴承': 'bearing',
    '端盖': 'end_cap',
    '阻尼条': 'damper_bar',
    '行星轮': 'planet_gear',
    '支架轴承': 'carrier_bearing',
}


def _normalizedComponentName(name: str) -> str:
    normalized = str(name).strip()
    normalized = re.sub(r'\.(?:stp|step|igs|iges)$', '', normalized, flags=re.IGNORECASE)
    if normalized.lower().endswith('_mm'):
        normalized = normalized[:-3]
    compact = re.sub(r'\s+', '', normalized)
    if compact in _COMPONENT_INSTANCE_ALIASES:
        return _COMPONENT_INSTANCE_ALIASES[compact]
    for prefix, alias in sorted(
        _NUMBERED_COMPONENT_INSTANCE_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        match = re.fullmatch(re.escape(prefix) + r'([0-9]+)', compact)
        if match is not None:
            return f'{alias}_{match.group(1)}'
    normalized = re.sub(r'[^A-Za-z0-9_-]+', '_', normalized).strip('_')
    if not normalized:
        codepoints = '_'.join(f'u{ord(character):x}' for character in compact)
        if not codepoints:
            raise ValueError('component name cannot be normalized to an empty identifier')
        return f'component_{codepoints}'
    return normalized.lower()


def _componentTypeName(componentName: str) -> str:
    """Return the semantic class while keeping the FEM region name as its instance ID."""
    normalized = _normalizedComponentName(componentName)
    if re.fullmatch(r'disk(?:[_-]?\d+)?', normalized):
        return 'magnetic_isolation_plate'
    if normalized.startswith('magnetic_isolation_plate'):
        return 'magnetic_isolation_plate'
    if 'shaft' in normalized or re.fullmatch(r'zhou[_-]?\d+', normalized):
        return 'shaft'
    if re.fullmatch(r'(?:cap(?:s|[_-]?\d+)?|end[_-]?cap(?:[_-]?\d+)?)', normalized):
        return 'end_cap'
    if 'insulating_paper' in normalized or normalized.startswith('insulation'):
        return 'insulation'
    if 'bearing' in normalized or re.fullmatch(r'b[_-]?\d+', normalized):
        return 'bearing'
    if 'magnet' in normalized:
        return 'magnets'
    if normalized.startswith('stator') or normalized.startswith('st_fe'):
        return 'stator'
    if (
        normalized in {'shell', 'pan'}
        or 'shell' in normalized
        or normalized.startswith('housing')
    ):
        return 'housing'
    if 'coil' in normalized or 'winding' in normalized:
        return 'winding'
    if (
        normalized == 'rotor'
        or normalized.endswith('_rotor')
        or normalized.startswith('rotor_fe')
        or normalized.startswith('rt_fe')
    ):
        return 'rotor_core'
    if 'sleeve' in normalized:
        return 'sleeve'
    if normalized.startswith('end_plate'):
        return 'end_plate'
    if normalized == 'oil_tank':
        return 'oil_tank'
    if normalized.startswith('damper_bar'):
        return 'damper_bar'
    if normalized.startswith('planet_gear'):
        return 'planet_gear'
    if normalized.startswith('ring_gear'):
        return 'ring_gear'
    if normalized == 'carrier':
        return 'carrier'
    if 'hub' in normalized:
        return 'wheel_hub'
    if 'sprayring' in normalized or 'spray_ring' in normalized:
        return 'cooling_ring'
    if 'rotorplate' in normalized or 'rotor_plate' in normalized:
        return 'end_plate'
    if normalized == 'fem':
        return 'gear'
    if 'gear' in normalized or re.fullmatch(r'g[_-]?\d+(?:[_-]\d+)?', normalized):
        return 'gear'
    return normalized


def _resolveFemSystem(control: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    candidates = []

    def visit(system: dict[str, Any]) -> None:
        thermal = system.get('thermalComponents', {})
        for femSystemUuid, femSystem in thermal.get('fem3DThermals', {}).items():
            if femSystem.get('femRegions'):
                candidates.append(
                    (
                        str(femSystemUuid),
                        {
                            **femSystem,
                            'femNetworkInterfaces': thermal.get(
                                'femNetworkInterfaces', {}
                            ),
                        },
                        'nestedFem3DThermal',
                    )
                )
        directRegions = thermal.get('femRegions')
        if directRegions:
            candidates.append(
                (
                    str(system.get('uuid', 'rootSystem')),
                    {
                        'femRegions': directRegions,
                        'tccInterfaces': thermal.get('femFemTccInterfaces', {}),
                        'femNetworkInterfaces': thermal.get('femNetworkInterfaces', {}),
                    },
                    'directFemRegions',
                )
            )
        for child in system.get('children', {}).values():
            if isinstance(child, dict):
                visit(child)

    visit(control['rootSystem'])
    if not candidates:
        raise KeyError('control data contains neither fem3DThermals nor direct femRegions')
    if len(candidates) != 1:
        raise ValueError(f'expected exactly one 3D FEM system, found {len(candidates)}')
    return candidates[0]


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
    _, femSystem, _ = _resolveFemSystem(control)
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


def _thermalAnalysisMetadata(
    control: dict[str, Any],
    *,
    snapshotTimeS: float | None = None,
) -> dict[str, Any]:
    timeDomain = control.get('solverSetting', {}).get('timeDomain', {})
    rawTransient = timeDomain.get('isTransient')
    if not isinstance(rawTransient, (bool, np.bool_)):
        return {
            'analysisType': 'unknown',
            'isTransient': None,
            'temporalSampleType': 'unknownSnapshot',
            'timeSeriesStored': False,
        }
    isTransient = bool(rawTransient)
    metadata: dict[str, Any] = {
        'analysisType': 'transient' if isTransient else 'steady',
        'isTransient': isTransient,
        'temporalSampleType': 'transientSnapshot' if isTransient else 'steadySolution',
        'timeSeriesStored': False,
    }
    if not isTransient:
        return metadata
    for sourceName, targetName in (
        ('deltaTime', 'sourceTimeStepS'),
        ('endTime', 'sourceEndTimeS'),
    ):
        rawValue = timeDomain.get(sourceName)
        if rawValue is None:
            continue
        value = float(rawValue)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f'transient solver {sourceName} must be positive and finite')
        metadata[targetName] = value
    if snapshotTimeS is not None:
        value = float(snapshotTimeS)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError('transient snapshotTimeS must be non-negative and finite')
        metadata['snapshotTimeS'] = value
    return metadata


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
