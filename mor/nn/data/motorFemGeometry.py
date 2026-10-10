from __future__ import annotations

from typing import Any

import numpy as np


def reshapeConnectivity(values: Any, width: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64).reshape(-1)
    if array.size == 0 or array.size % width != 0:
        raise ValueError(f'{name} must contain a non-empty flattened width-{width} array')
    return array.reshape(-1, width)


def tetrahedronGeometry(
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


def lumpedNodeVolumes(
    nodeCount: int, connectivity: np.ndarray, volumes: np.ndarray
) -> np.ndarray:
    return np.bincount(connectivity.reshape(-1), weights=np.repeat(volumes / 4.0, 4),
                       minlength=nodeCount)


def boundaryNodeGeometry(
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

    indices = surfaceConnectivity.reshape(-1)
    nodeAreas = np.bincount(indices, weights=np.repeat(areas / 3.0, 3), minlength=nodes.shape[0])
    nodeAreaVectors = np.column_stack([
        np.bincount(indices, weights=np.repeat(areaVectors[:, axis] / 3.0, 3), minlength=nodes.shape[0])
        for axis in range(3)
    ])

    boundaryNodes = np.flatnonzero(nodeAreas > 0.0)
    normalVectors = nodeAreaVectors[boundaryNodes]
    normalLengths = np.linalg.norm(normalVectors, axis=1)
    cancelled = normalLengths <= np.finfo(np.float64).eps
    if np.any(cancelled):
        repeatedAreas = np.repeat(areas, 3)
        largestAreas = np.zeros(nodes.shape[0], dtype=np.float64)
        np.maximum.at(largestAreas, indices, repeatedAreas)
        largestFaces = np.full(nodes.shape[0], areas.size, dtype=np.int64)
        candidates = repeatedAreas == largestAreas[indices]
        np.minimum.at(largestFaces, indices[candidates], np.repeat(np.arange(areas.size), 3)[candidates])
        selectedFaces = largestFaces[boundaryNodes[cancelled]]
        normalVectors[cancelled] = areaVectors[selectedFaces]
        normalLengths[cancelled] = areas[selectedFaces]
    if np.any(normalLengths <= np.finfo(np.float64).eps):
        raise ValueError('could not construct non-zero boundary normals')
    normals = normalVectors / normalLengths[:, None]
    return boundaryNodes, normals, nodeAreas[boundaryNodes]


def sampleWithConservedWeight(
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


def sampleByGroupWithConservedWeight(
    weights: np.ndarray,
    groupIds: np.ndarray,
    maximumCount: int | None,
    randomSeed: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = weights.size
    if maximumCount is None or int(maximumCount) >= count:
        return np.arange(count), weights.copy()
    maximum = int(maximumCount)
    groupedIndices = np.argsort(groupIds, kind='stable')
    orderedGroupIds = groupIds[groupedIndices]
    groupOffsets = np.concatenate(([0], np.flatnonzero(orderedGroupIds[1:] != orderedGroupIds[:-1]) + 1, [count]))
    uniqueGroupIds = orderedGroupIds[groupOffsets[:-1]]
    groupCounts = np.diff(groupOffsets)
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
    for groupIndex, allocation in enumerate(allocations):
        candidates = groupedIndices[groupOffsets[groupIndex]:groupOffsets[groupIndex + 1]]
        chosen = np.sort(generator.choice(candidates, size=int(allocation), replace=False))
        patchWeights = weights[chosen].copy()
        patchWeights *= weights[candidates].sum() / patchWeights.sum()
        selections.append(chosen)
        selectedWeights.append(patchWeights)
    selection = np.concatenate(selections)
    order = np.argsort(selection)
    return selection[order], np.concatenate(selectedWeights)[order]


def matchIntegratedPower(
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
