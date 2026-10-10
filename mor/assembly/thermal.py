from typing import Any
from numbers import Integral

import numpy as np

from mor.backends import backendRegistry
from .sparse import sparseAssembly as _SparseAssembly
from .tetra import thermalElementMatrices
from mor.operators.matrix import matrixOperator
from mor.operators.operatorsBase import operatorBase
from mor.operators.thermalSet import thermalOperatorSet


def _scalarElementValues(value: Any, count: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} must contain finite values')
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
    if not np.all(np.isfinite(points)):
        raise ValueError('nodes must contain finite values')
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
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} must contain finite values')
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
    deviceName: str | None = None,
) -> thermalOperatorSet:
    backend = backendRegistry.get(backendName, **({'device': deviceName} if deviceName is not None else {}))
    def hostData(value):
        if isinstance(value, (list, tuple, int, float, complex, np.ndarray)) or value is None:
            return value
        return backend.array.toNumpy(backend.array.toArray(value))
    nodes, elements = hostData(nodes), hostData(elements)
    conductivity, density, specificHeat, heatSource = map(hostData, (conductivity, density, specificHeat, heatSource))
    convectionFaces, convectionCoefficient, ambientTemperature = map(hostData, (convectionFaces, convectionCoefficient, ambientTemperature))
    heatFluxFaces, heatFlux = map(hostData, (heatFluxFaces, heatFlux))
    contactPrimaryFaces, contactSecondaryFaces, contactCoefficient = map(hostData, (contactPrimaryFaces, contactSecondaryFaces, contactCoefficient))
    def sparseMatrix(values, rows, columns):
        return backend.array.sparseFromTriplets(values, rows, columns, shape)
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
    if not isinstance(assemblyBatchSize, Integral) or isinstance(assemblyBatchSize, bool) or assemblyBatchSize <= 0:
        raise ValueError('assemblyBatchSize must be positive')

    load = np.zeros(points.shape[0], dtype=float)
    volumes = np.empty(elementCount, dtype=float)
    shape = (points.shape[0], points.shape[0])
    conductivityAssembly = _SparseAssembly(shape, backend)
    capacityAssembly = _SparseAssembly(shape, backend)

    for start in range(0, elementCount, assemblyBatchSize):
        stop = min(start + assemblyBatchSize, elementCount)
        nodeIndices = tetra[start:stop]
        coordinates = points[nodeIndices]
        batchVolumes, localConductivity, localCapacity, localLoad = thermalElementMatrices(
            coordinates, conductivityTensors[start:stop], densityValues[start:stop],
            specificHeatValues[start:stop], heatSourceValues[start:stop], capacityMode, start)
        volumes[start:stop] = batchVolumes
        np.add.at(load, nodeIndices.reshape(-1), np.repeat(localLoad, 4))
        rows = np.repeat(nodeIndices, 4, axis=1).reshape(-1)
        cols = np.tile(nodeIndices, (1, 4)).reshape(-1)
        conductivityAssembly.add(sparseMatrix(localConductivity.reshape(-1), rows, cols))
        capacityAssembly.add(sparseMatrix(localCapacity.reshape(-1), rows, cols))

    conductivityMatrix = conductivityAssembly.finish()
    capacityMatrix = capacityAssembly.finish()

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
        conductivityMatrix = conductivityMatrix + sparseMatrix(localBoundary.reshape(-1), boundaryRows, boundaryCols)

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
        conductivityMatrix = conductivityMatrix + sparseMatrix(values, primaryRows, primaryCols)
        conductivityMatrix = conductivityMatrix + sparseMatrix(values, secondaryRows, secondaryCols)
        conductivityMatrix = conductivityMatrix + sparseMatrix(-values, primaryRows, secondaryCols)
        conductivityMatrix = conductivityMatrix + sparseMatrix(-values, secondaryRows, primaryCols)

    return thermalOperatorSet(
        conductivity=matrixOperator(conductivityMatrix, backendName=backendName),
        capacity=matrixOperator(capacityMatrix, backendName=backendName),
        load=matrixOperator(backend.array.toArray(load.reshape(-1, 1)), backendName=backendName),
        elementVolumes=volumes,
        metadata={'contactAreas': contactAreas, 'elementKernelBackend': 'numpy', 'assemblyBackend': backend.name},
    )


def applyDirichletConditions(
    operator: operatorBase | Any,
    load: operatorBase | Any,
    nodeIndices: Any,
    values: Any,
    *,
    backendName: str | None = None,
    deviceName: str | None = None,
) -> tuple[matrixOperator, matrixOperator]:
    name = backendName or (operator.backendName if isinstance(operator, operatorBase) else 'scipy')
    backend = backendRegistry.get(name, **({'device': deviceName} if deviceName is not None else {}))
    for argument in (operator, load):
        if isinstance(argument, operatorBase) and argument.backendName != name:
            raise ValueError('Dirichlet operator and load must match the selected backend')
    matrix = operator.toBackendData() if isinstance(operator, operatorBase) else backend.array.toArray(operator)
    rhs = load.toBackendData() if isinstance(load, operatorBase) else backend.array.toArray(load)
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1] or backend.array.size(rhs) != matrix.shape[0]:
        raise ValueError('Dirichlet matrix must be square and load must contain one value per node')
    def host(value):
        return np.asarray(backend.array.toNumpy(backend.array.toArray(value)))
    rawIndices = host(nodeIndices).reshape(-1)
    prescribed = host(values)
    if not np.all(np.isfinite(rawIndices)) or not np.all(rawIndices == np.floor(rawIndices)):
        raise ValueError('Dirichlet node indices must be finite integers')
    if not np.all(np.isfinite(prescribed)):
        raise ValueError('Dirichlet values must be finite')
    prescribed = (np.full(rawIndices.size, prescribed.item()) if prescribed.ndim == 0 else prescribed.reshape(-1))
    if prescribed.size != rawIndices.size:
        raise ValueError('Dirichlet values must be scalar or match nodeIndices')
    valueByNode = {}
    for nodeIndex, value in zip(rawIndices, prescribed):
        node = int(nodeIndex)
        if node < 0 or node >= matrix.shape[0]:
            raise ValueError('Dirichlet node index out of range')
        if node in valueByNode and not np.isclose(valueByNode[node], value):
            raise ValueError(f'conflicting Dirichlet values for node {node}')
        valueByNode[node] = value
    indices = np.array(sorted(valueByNode), dtype=np.int64)
    prescribed = np.array([valueByNode[index] for index in indices], dtype=prescribed.dtype)
    constrained, shifted = backend.linalg.applyDirichlet(matrix, rhs, indices, prescribed)
    return matrixOperator(constrained, name), matrixOperator(shifted, name)
