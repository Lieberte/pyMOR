from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from mor.models import thermalBoundaryField

from .meshIr import meshIr


def _scalarValues(values: Any, count: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != count or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must contain {count} finite scalar values')
    return array


def _positiveScale(value: float, name: str) -> float:
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f'{name} must be positive and finite')
    return scale


@dataclass(frozen=True)
class thermalFieldMappingResult:
    boundaryName: str
    targetNodeIndices: np.ndarray
    values: np.ndarray
    sourceIndices: np.ndarray
    distances: np.ndarray
    method: str

    @property
    def mappedMask(self) -> np.ndarray:
        return np.isfinite(self.values)

    @property
    def coverage(self) -> float:
        return float(np.mean(self.mappedMask)) if self.values.size else 1.0

    @property
    def unmappedNodeIndices(self) -> np.ndarray:
        return self.targetNodeIndices[~self.mappedMask]

    @property
    def maximumDistance(self) -> float:
        mappedDistances = self.distances[self.mappedMask]
        return float(np.max(mappedDistances)) if mappedDistances.size else float('nan')

    @property
    def meanDistance(self) -> float:
        mappedDistances = self.distances[self.mappedMask]
        return float(np.mean(mappedDistances)) if mappedDistances.size else float('nan')

    def diagnostics(self) -> dict[str, Any]:
        return {
            'boundaryName': self.boundaryName,
            'method': self.method,
            'targetNodeCount': int(self.targetNodeIndices.size),
            'mappedNodeCount': int(np.count_nonzero(self.mappedMask)),
            'coverage': self.coverage,
            'maximumDistance': self.maximumDistance,
            'meanDistance': self.meanDistance,
            'unmappedNodeIndices': self.unmappedNodeIndices.tolist(),
        }

    def requireComplete(self) -> 'thermalFieldMappingResult':
        if self.coverage < 1.0:
            raise ValueError(
                f'{self.method} mapping covers {np.count_nonzero(self.mappedMask)}/'
                f'{self.targetNodeIndices.size} nodes on boundary {self.boundaryName!r}; '
                f'unmapped global node indices: {self.unmappedNodeIndices.tolist()}'
            )
        return self

    def asBoundaryField(self) -> thermalBoundaryField:
        self.requireComplete()
        return thermalBoundaryField(self.values, location='node')


def _boundaryNodes(mesh: meshIr, boundaryName: str) -> np.ndarray:
    if boundaryName not in mesh.boundaryNodeIndices:
        raise KeyError(f'boundaryNodeIndices does not contain {boundaryName!r}')
    indices = np.asarray(mesh.boundaryNodeIndices[boundaryName], dtype=int)
    if indices.size == 0:
        raise ValueError(f'boundary {boundaryName!r} has no nodes')
    return indices


def mapBoundaryFieldByNodeIds(
    mesh: meshIr,
    boundaryName: str,
    sourceNodeIds: Any,
    sourceValues: Any,
    *,
    nodeIdData: str = 'originalNodeId',
    requireComplete: bool = True,
) -> thermalFieldMappingResult:
    targetNodes = _boundaryNodes(mesh, boundaryName)
    if nodeIdData not in mesh.pointData:
        raise KeyError(
            f'pointData does not contain {nodeIdData!r}; provide the solver/original node labels '
            'explicitly when converting the mesh'
        )
    meshNodeIds = np.asarray(mesh.pointData[nodeIdData]).reshape(-1)
    sourceIds = np.asarray(sourceNodeIds).reshape(-1)
    values = _scalarValues(sourceValues, sourceIds.size, 'sourceValues')
    if np.unique(sourceIds).size != sourceIds.size:
        raise ValueError('sourceNodeIds must be unique')
    lookup = {nodeId.item() if isinstance(nodeId, np.generic) else nodeId: index for index, nodeId in enumerate(sourceIds)}
    mappedValues = np.full(targetNodes.size, np.nan, dtype=float)
    sourceIndices = np.full(targetNodes.size, -1, dtype=int)
    for localIndex, nodeId in enumerate(meshNodeIds[targetNodes]):
        key = nodeId.item() if isinstance(nodeId, np.generic) else nodeId
        sourceIndex = lookup.get(key)
        if sourceIndex is not None:
            mappedValues[localIndex] = values[sourceIndex]
            sourceIndices[localIndex] = sourceIndex
    result = thermalFieldMappingResult(
        boundaryName=boundaryName,
        targetNodeIndices=targetNodes.copy(),
        values=mappedValues,
        sourceIndices=sourceIndices,
        distances=np.zeros(targetNodes.size, dtype=float),
        method='nodeId',
    )
    return result.requireComplete() if requireComplete else result


def mapBoundaryFieldByCoordinates(
    mesh: meshIr,
    boundaryName: str,
    sourceCoordinates: Any,
    sourceValues: Any,
    *,
    maxDistance: float,
    method: str = 'nearest',
    neighbors: int = 4,
    power: float = 2.0,
    meshLengthScaleToMeters: float = 1.0,
    sourceLengthScaleToMeters: float = 1.0,
    requireComplete: bool = True,
) -> thermalFieldMappingResult:
    targetNodes = _boundaryNodes(mesh, boundaryName)
    sourcePoints = np.asarray(sourceCoordinates, dtype=float)
    if sourcePoints.ndim != 2 or sourcePoints.shape[1] != mesh.nodes.shape[1] or sourcePoints.shape[0] == 0:
        raise ValueError(
            f'sourceCoordinates must have shape (n, {mesh.nodes.shape[1]}) with n > 0'
        )
    if not np.all(np.isfinite(sourcePoints)):
        raise ValueError('sourceCoordinates must contain finite values')
    values = _scalarValues(sourceValues, sourcePoints.shape[0], 'sourceValues')
    maximumDistance = float(maxDistance)
    if not np.isfinite(maximumDistance) or maximumDistance <= 0.0:
        raise ValueError('maxDistance must be positive and finite in meters')
    normalizedMethod = str(method).lower()
    if normalizedMethod not in ('nearest', 'idw'):
        raise ValueError("coordinate mapping method must be 'nearest' or 'idw'")
    neighbors = int(neighbors)
    power = float(power)
    if neighbors <= 0:
        raise ValueError('neighbors must be positive')
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError('power must be positive and finite')
    meshScale = _positiveScale(meshLengthScaleToMeters, 'meshLengthScaleToMeters')
    sourceScale = _positiveScale(sourceLengthScaleToMeters, 'sourceLengthScaleToMeters')
    targetsMeters = np.asarray(mesh.nodes[targetNodes], dtype=float) * meshScale
    sourcesMeters = sourcePoints * sourceScale
    tree = cKDTree(sourcesMeters)
    queryCount = 1 if normalizedMethod == 'nearest' else min(neighbors, sourcePoints.shape[0])
    distances, sourceIndices = tree.query(targetsMeters, k=queryCount)
    nearestDistances = distances if queryCount == 1 else distances[:, 0]
    mapped = nearestDistances <= maximumDistance
    mappedValues = np.full(targetNodes.size, np.nan, dtype=float)
    representativeSources = np.full(targetNodes.size, -1, dtype=int)
    if normalizedMethod == 'nearest':
        sourceIndices = np.asarray(sourceIndices, dtype=int)
        mappedValues[mapped] = values[sourceIndices[mapped]]
        representativeSources[mapped] = sourceIndices[mapped]
    else:
        distances = np.asarray(distances, dtype=float).reshape(targetNodes.size, queryCount)
        sourceIndices = np.asarray(sourceIndices, dtype=int).reshape(targetNodes.size, queryCount)
        for targetIndex in np.flatnonzero(mapped):
            localDistances = distances[targetIndex]
            localSources = sourceIndices[targetIndex]
            exact = np.flatnonzero(localDistances <= np.finfo(float).eps)
            if exact.size:
                mappedValues[targetIndex] = values[localSources[int(exact[0])]]
            else:
                weights = 1.0 / np.power(localDistances, power)
                mappedValues[targetIndex] = float(np.dot(weights, values[localSources]) / np.sum(weights))
            representativeSources[targetIndex] = int(localSources[0])
    result = thermalFieldMappingResult(
        boundaryName=boundaryName,
        targetNodeIndices=targetNodes.copy(),
        values=mappedValues,
        sourceIndices=representativeSources,
        distances=np.asarray(nearestDistances, dtype=float),
        method=normalizedMethod,
    )
    return result.requireComplete() if requireComplete else result
