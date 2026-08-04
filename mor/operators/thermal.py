from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse

from .matrix import matrixOperator
from .operatorsBase import operatorBase


@dataclass(frozen=True)
class thermalOperatorSet:
    conductivity: matrixOperator
    capacity: matrixOperator
    load: matrixOperator
    elementVolumes: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def nodeCount(self) -> int:
        return int(self.conductivity.shape[0])

    @property
    def elementCount(self) -> int:
        return int(self.elementVolumes.size)

    @property
    def K(self) -> matrixOperator:
        return self.conductivity

    @property
    def M(self) -> matrixOperator:
        return self.capacity

    @property
    def Q(self) -> matrixOperator:
        return self.load

    def steadyResidual(self, temperature: Any) -> Any:
        return self.conductivity.apply(temperature) - self.load.toBackendData()

    def transientResidual(self, temperature: Any, temperatureRate: Any) -> Any:
        return (
            self.capacity.apply(temperatureRate)
            + self.conductivity.apply(temperature)
            - self.load.toBackendData()
        )


def _scalarElementValues(value: Any, count: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(count, float(arr), dtype=float)
    flat = arr.reshape(-1)
    if flat.size != count:
        raise ValueError(f'{name} must be scalar or contain {count} values')
    return flat


def _conductivityTensors(value: Any, count: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        tensors = np.repeat((np.eye(3) * float(arr))[None, :, :], count, axis=0)
    elif arr.size == count and arr.ndim <= 2:
        tensors = arr.reshape(count, 1, 1) * np.eye(3)[None, :, :]
    elif arr.shape == (3, 3):
        tensors = np.repeat(arr[None, :, :], count, axis=0)
    elif arr.shape == (count, 3, 3):
        tensors = arr
    else:
        raise ValueError('conductivity must be scalar, per-element scalar, 3x3, or per-element 3x3')
    if not np.all(np.isfinite(tensors)):
        raise ValueError('conductivity contains non-finite values')
    symmetric = 0.5 * (tensors + np.swapaxes(tensors, 1, 2))
    if np.any(np.linalg.eigvalsh(symmetric) <= 0.0):
        raise ValueError('conductivity tensors must be positive definite')
    return symmetric


def _validateTetraMesh(nodes: Any, elements: Any) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(nodes, dtype=float)
    tetra = np.asarray(elements, dtype=int)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('linear tetra thermal assembly requires nodes with shape (n, 3)')
    if tetra.ndim != 2 or tetra.shape[1] < 4:
        raise ValueError('elements must have shape (m, 4) or a higher-order tetra connectivity')
    tetra = tetra[:, :4]
    if tetra.size == 0:
        raise ValueError('at least one tetra element is required')
    if tetra.min() < 0 or tetra.max() >= points.shape[0]:
        raise ValueError('element node index out of range')
    return points, tetra


def _surfaceNodalValues(value: Any, count: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full((count, 3), float(arr), dtype=float)
    if arr.shape == (count,):
        return np.repeat(arr[:, None], 3, axis=1)
    if arr.shape == (count, 3):
        return arr
    raise ValueError(f'{name} must be scalar, contain {count} face values, or have shape ({count}, 3)')


def _triangleTripleShapeIntegrals() -> np.ndarray:
    integrals = np.empty((3, 3, 3), dtype=float)
    factorial = {1: 1.0, 2: 2.0, 3: 6.0}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                _, counts = np.unique([i, j, k], return_counts=True)
                numerator = np.prod([factorial[int(count)] for count in counts])
                integrals[i, j, k] = numerator / 60.0
    return integrals


def _validateTriangleFaces(nodes: np.ndarray, faces: Any, name: str) -> np.ndarray:
    triangles = np.asarray(faces, dtype=int)
    if triangles.ndim != 2 or triangles.shape[1] < 3:
        raise ValueError(f'{name} must have shape (m, 3) or higher-order triangle connectivity')
    triangles = triangles[:, :3]
    if triangles.size > 0 and (triangles.min() < 0 or triangles.max() >= nodes.shape[0]):
        raise ValueError(f'{name} node index out of range')
    return triangles


def assembleLinearTetraThermalOperators(
    nodes: Any,
    elements: Any,
    *,
    conductivity: Any,
    density: Any = 1.0,
    specificHeat: Any = 1.0,
    heatSource: Any = 0.0,
    capacityMode: str = 'consistent',
    convectionFaces: Any | None = None,
    convectionCoefficient: Any | None = None,
    ambientTemperature: Any | None = None,
    heatFluxFaces: Any | None = None,
    heatFlux: Any | None = None,
    contactPrimaryFaces: Any | None = None,
    contactSecondaryFaces: Any | None = None,
    contactCoefficient: Any | None = None,
    assemblyBatchSize: int = 50000,
    backendName: str = 'scipy',
) -> thermalOperatorSet:
    points, tetra = _validateTetraMesh(nodes, elements)
    elementCount = tetra.shape[0]
    conductivityTensors = _conductivityTensors(conductivity, elementCount)
    densityValues = _scalarElementValues(density, elementCount, 'density')
    specificHeatValues = _scalarElementValues(specificHeat, elementCount, 'specificHeat')
    heatSourceValues = _scalarElementValues(heatSource, elementCount, 'heatSource')
    if np.any(densityValues <= 0.0):
        raise ValueError('density must be positive')
    if np.any(specificHeatValues <= 0.0):
        raise ValueError('specificHeat must be positive')
    if capacityMode not in ('consistent', 'lumped'):
        raise ValueError("capacityMode must be 'consistent' or 'lumped'")
    if assemblyBatchSize <= 0:
        raise ValueError('assemblyBatchSize must be positive')

    load = np.zeros(points.shape[0], dtype=float)
    volumes = np.empty(elementCount, dtype=float)
    referenceGradients = np.array(
        [[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    consistentTemplate = np.ones((4, 4), dtype=float) + np.eye(4, dtype=float)
    shape = (points.shape[0], points.shape[0])
    conductivityMatrix = csr_matrix(shape, dtype=float)
    capacityMatrix = csr_matrix(shape, dtype=float)

    for start in range(0, elementCount, assemblyBatchSize):
        stop = min(start + assemblyBatchSize, elementCount)
        nodeIndices = tetra[start:stop]
        coordinates = points[nodeIndices]
        jacobians = np.stack(
            [
                coordinates[:, 1] - coordinates[:, 0],
                coordinates[:, 2] - coordinates[:, 0],
                coordinates[:, 3] - coordinates[:, 0],
            ],
            axis=2,
        )
        determinants = np.linalg.det(jacobians)
        batchVolumes = np.abs(determinants) / 6.0
        degenerate = np.flatnonzero(batchVolumes <= np.finfo(float).eps)
        if degenerate.size:
            raise ValueError(f'degenerate tetra element at index {start + int(degenerate[0])}')
        volumes[start:stop] = batchVolumes
        gradients = np.einsum('ai,eij->eaj', referenceGradients, np.linalg.inv(jacobians))
        localConductivity = batchVolumes[:, None, None] * np.einsum(
            'eai,eij,ebj->eab',
            gradients,
            conductivityTensors[start:stop],
            gradients,
        )
        heatCapacity = densityValues[start:stop] * specificHeatValues[start:stop]
        if capacityMode == 'consistent':
            localCapacity = (
                heatCapacity[:, None, None]
                * batchVolumes[:, None, None]
                * consistentTemplate[None, :, :]
                / 20.0
            )
        else:
            localCapacity = (
                heatCapacity[:, None, None]
                * batchVolumes[:, None, None]
                * np.eye(4, dtype=float)[None, :, :]
                / 4.0
            )
        localLoad = heatSourceValues[start:stop] * batchVolumes / 4.0
        np.add.at(load, nodeIndices.reshape(-1), np.repeat(localLoad, 4))
        rows = np.repeat(nodeIndices, 4, axis=1).reshape(-1)
        cols = np.tile(nodeIndices, (1, 4)).reshape(-1)
        conductivityMatrix = conductivityMatrix + coo_matrix(
            (localConductivity.reshape(-1), (rows, cols)), shape=shape
        ).tocsr()
        capacityMatrix = capacityMatrix + coo_matrix(
            (localCapacity.reshape(-1), (rows, cols)), shape=shape
        ).tocsr()

    if convectionFaces is not None:
        if convectionCoefficient is None or ambientTemperature is None:
            raise ValueError('convectionCoefficient and ambientTemperature are required with convectionFaces')
        triangles = _validateTriangleFaces(points, convectionFaces, 'convectionFaces')
        coefficients = _surfaceNodalValues(
            convectionCoefficient,
            triangles.shape[0],
            'convectionCoefficient',
        )
        ambient = _surfaceNodalValues(
            ambientTemperature,
            triangles.shape[0],
            'ambientTemperature',
        )
        if np.any(coefficients < 0.0):
            raise ValueError('convectionCoefficient must be non-negative')
        coordinates = points[triangles]
        areas = 0.5 * np.linalg.norm(
            np.cross(coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0]),
            axis=1,
        )
        degenerate = np.flatnonzero(areas <= np.finfo(float).eps)
        if degenerate.size:
            raise ValueError(f'degenerate convection face at index {int(degenerate[0])}')
        tripleIntegrals = _triangleTripleShapeIntegrals()
        localBoundary = areas[:, None, None] * np.einsum(
            'ek,ijk->eij',
            coefficients,
            tripleIntegrals,
        )
        localConvectionLoad = areas[:, None] * np.einsum(
            'ej,ek,ijk->ei',
            coefficients,
            ambient,
            tripleIntegrals,
        )
        np.add.at(load, triangles.reshape(-1), localConvectionLoad.reshape(-1))
        boundaryRows = np.repeat(triangles, 3, axis=1).reshape(-1)
        boundaryCols = np.tile(triangles, (1, 3)).reshape(-1)
        conductivityMatrix = conductivityMatrix + coo_matrix(
            (localBoundary.reshape(-1), (boundaryRows, boundaryCols)), shape=shape
        ).tocsr()

    if heatFluxFaces is not None:
        if heatFlux is None:
            raise ValueError('heatFlux is required with heatFluxFaces')
        triangles = _validateTriangleFaces(points, heatFluxFaces, 'heatFluxFaces')
        fluxValues = _surfaceNodalValues(heatFlux, triangles.shape[0], 'heatFlux')
        coordinates = points[triangles]
        areas = 0.5 * np.linalg.norm(
            np.cross(coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0]),
            axis=1,
        )
        degenerate = np.flatnonzero(areas <= np.finfo(float).eps)
        if degenerate.size:
            raise ValueError(f'degenerate heat flux face at index {int(degenerate[0])}')
        boundaryMass = (np.ones((3, 3), dtype=float) + np.eye(3, dtype=float)) / 12.0
        localFluxLoad = areas[:, None] * np.einsum('ej,ij->ei', fluxValues, boundaryMass)
        np.add.at(load, triangles.reshape(-1), localFluxLoad.reshape(-1))

    contactAreas = np.empty(0, dtype=float)
    if contactPrimaryFaces is not None:
        if contactSecondaryFaces is None or contactCoefficient is None:
            raise ValueError(
                'contactSecondaryFaces and contactCoefficient are required with contactPrimaryFaces'
            )
        primary = _validateTriangleFaces(points, contactPrimaryFaces, 'contactPrimaryFaces')
        secondary = _validateTriangleFaces(points, contactSecondaryFaces, 'contactSecondaryFaces')
        if secondary.shape != primary.shape:
            raise ValueError('contact primary and secondary face mappings must have the same shape')
        coefficients = _surfaceNodalValues(contactCoefficient, primary.shape[0], 'contactCoefficient')
        if np.any(coefficients < 0.0):
            raise ValueError('contactCoefficient must be non-negative')
        coordinates = points[primary]
        contactAreas = 0.5 * np.linalg.norm(
            np.cross(coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0]),
            axis=1,
        )
        degenerate = np.flatnonzero(contactAreas <= np.finfo(float).eps)
        if degenerate.size:
            raise ValueError(f'degenerate contact face at index {int(degenerate[0])}')
        localContact = contactAreas[:, None, None] * np.einsum(
            'ek,ijk->eij',
            coefficients,
            _triangleTripleShapeIntegrals(),
        )
        primaryRows = np.repeat(primary, 3, axis=1).reshape(-1)
        primaryCols = np.tile(primary, (1, 3)).reshape(-1)
        secondaryRows = np.repeat(secondary, 3, axis=1).reshape(-1)
        secondaryCols = np.tile(secondary, (1, 3)).reshape(-1)
        values = localContact.reshape(-1)
        conductivityMatrix = conductivityMatrix + coo_matrix(
            (values, (primaryRows, primaryCols)), shape=shape
        ).tocsr()
        conductivityMatrix = conductivityMatrix + coo_matrix(
            (values, (secondaryRows, secondaryCols)), shape=shape
        ).tocsr()
        conductivityMatrix = conductivityMatrix + coo_matrix(
            (-values, (primaryRows, secondaryCols)), shape=shape
        ).tocsr()
        conductivityMatrix = conductivityMatrix + coo_matrix(
            (-values, (secondaryRows, primaryCols)), shape=shape
        ).tocsr()

    return thermalOperatorSet(
        conductivity=matrixOperator(conductivityMatrix, backendName=backendName),
        capacity=matrixOperator(capacityMatrix, backendName=backendName),
        load=matrixOperator(load.reshape(-1, 1), backendName=backendName),
        elementVolumes=volumes,
        metadata={'contactAreas': contactAreas},
    )


def applyDirichletConditions(
    operator: operatorBase | Any,
    load: operatorBase | Any,
    nodeIndices: Any,
    values: Any,
    *,
    backendName: str = 'scipy',
) -> tuple[matrixOperator, matrixOperator]:
    matrixData = operator.toBackendData() if isinstance(operator, operatorBase) else operator
    loadData = load.toBackendData() if isinstance(load, operatorBase) else load
    matrix = matrixData.tocsr().astype(float) if issparse(matrixData) else csr_matrix(np.asarray(matrixData, dtype=float))
    rhs = np.asarray(loadData, dtype=float).reshape(-1).copy()
    rawIndices = np.asarray(nodeIndices, dtype=int).reshape(-1)
    if rawIndices.size == 0:
        return matrixOperator(matrix, backendName=backendName), matrixOperator(rhs.reshape(-1, 1), backendName=backendName)
    prescribed = np.asarray(values, dtype=float)
    if prescribed.ndim == 0:
        prescribed = np.full(rawIndices.size, float(prescribed), dtype=float)
    else:
        prescribed = prescribed.reshape(-1)
    if prescribed.size != rawIndices.size:
        raise ValueError('Dirichlet values must be scalar or match nodeIndices')
    valueByNode: dict[int, float] = {}
    for nodeIndex, value in zip(rawIndices, prescribed):
        node = int(nodeIndex)
        temperature = float(value)
        if node < 0 or node >= matrix.shape[0]:
            raise ValueError('Dirichlet node index out of range')
        if node in valueByNode and not np.isclose(valueByNode[node], temperature):
            raise ValueError(f'conflicting Dirichlet values for node {node}')
        valueByNode[node] = temperature
    indices = np.array(sorted(valueByNode), dtype=int)
    prescribed = np.array([valueByNode[int(index)] for index in indices], dtype=float)
    rhs -= np.asarray(matrix[:, indices] @ prescribed).reshape(-1)
    constrained = matrix.tolil()
    constrained[:, indices] = 0.0
    constrained[indices, :] = 0.0
    for nodeIndex in indices:
        constrained[int(nodeIndex), int(nodeIndex)] = 1.0
    rhs[indices] = prescribed
    return (
        matrixOperator(constrained.tocsr(), backendName=backendName),
        matrixOperator(rhs.reshape(-1, 1), backendName=backendName),
    )
