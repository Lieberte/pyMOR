from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from mor.models import thermalBoundaryField, thermalContactInterface, thermalConvectionBoundary
from mor.operators import thermalOperatorSet, assembleLinearTetraThermalOperators

from .meshIr import meshIr
from .utils.facetUtils import quadRowToTwoTriangles


convectionBoundary = thermalConvectionBoundary


_tetraTypes = frozenset({'tetra', 'tetra4', 'tetra10'})
_triangleTypes = frozenset({'triangle', 'triangle6', 'triangle10'})
_quadTypes = frozenset({'quad', 'quad8', 'quad9'})


def _tetraBlocks(mesh: meshIr) -> tuple[list[int], np.ndarray]:
    blockIndices = []
    connectivity = []
    for blockIndex, (cellType, cells) in enumerate(mesh.cells or []):
        if cellType not in _tetraTypes:
            continue
        arr = np.asarray(cells, dtype=int)
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError(f'invalid {cellType} connectivity in block {blockIndex}')
        blockIndices.append(blockIndex)
        connectivity.append(arr[:, :4])
    if not connectivity:
        raise NotImplementedError('thermal mesh assembly currently supports linear tetra volume cells')
    return blockIndices, np.concatenate(connectivity, axis=0)


_missing = object()


def _cellProperty(
    mesh: meshIr,
    value: Any,
    name: str,
    blockIndices: list[int],
    *,
    defaultValue: Any = _missing,
) -> Any:
    if isinstance(value, dict):
        if not value:
            if defaultValue is _missing:
                raise ValueError(f'{name} cell-set mapping cannot be empty')
            return defaultValue
        valueArrays = [np.asarray(item, dtype=float) for item in value.values()]
        shapes = {item.shape for item in valueArrays}
        if name == 'conductivity' and shapes.issubset({(), (3, 3)}):
            propertyShape = (3, 3) if (3, 3) in shapes else ()
        elif len(shapes) == 1:
            propertyShape = next(iter(shapes))
        else:
            raise ValueError(f'all {name} cell-set values must have compatible shapes')
        selectedBlocks = []
        assignedBlocks = []
        blockLookup = {blockIndex: selectedIndex for selectedIndex, blockIndex in enumerate(blockIndices)}
        for blockIndex in blockIndices:
            numCells = int(np.asarray(mesh.cells[blockIndex][1]).shape[0])
            if defaultValue is _missing:
                selectedBlocks.append(np.empty((numCells,) + propertyShape, dtype=float))
            else:
                defaultArray = np.asarray(defaultValue, dtype=float)
                selectedBlocks.append(
                    np.broadcast_to(defaultArray, (numCells,) + propertyShape).copy()
                )
            assignedBlocks.append(np.zeros(numCells, dtype=bool))
        for cellSetName, propertyValue in value.items():
            if cellSetName not in mesh.cellSets:
                raise KeyError(f'cellSets does not contain {cellSetName!r} for {name}')
            materialValue = np.asarray(propertyValue, dtype=float)
            if name == 'conductivity' and propertyShape == (3, 3) and materialValue.ndim == 0:
                materialValue = np.eye(3, dtype=float) * float(materialValue)
            if materialValue.shape != propertyShape:
                raise ValueError(f'all {name} cell-set values must have shape {propertyShape}')
            for blockIndex, cellIndices in enumerate(mesh.cellSets[cellSetName]):
                if blockIndex not in blockLookup:
                    continue
                selectedIndex = blockLookup[blockIndex]
                indices = np.asarray(cellIndices, dtype=int)
                if np.any(assignedBlocks[selectedIndex][indices]):
                    raise ValueError(f'overlapping cell sets assign {name} more than once')
                selectedBlocks[selectedIndex][indices] = materialValue
                assignedBlocks[selectedIndex][indices] = True
        missing = sum(int(np.count_nonzero(~assigned)) for assigned in assignedBlocks)
        if missing and defaultValue is _missing:
            raise ValueError(f'{name} cell-set mapping leaves {missing} tetra elements unassigned')
        return np.concatenate(selectedBlocks, axis=0)
    if not isinstance(value, str):
        return value
    if value not in mesh.cellData:
        raise KeyError(f'cellData does not contain {value!r} for {name}')
    selected = [np.asarray(mesh.cellData[value][blockIndex]) for blockIndex in blockIndices]
    return np.concatenate(selected, axis=0)


def _boundaryTriangles(mesh: meshIr, boundaryName: str) -> np.ndarray:
    if boundaryName not in mesh.boundaryFaces:
        raise KeyError(f'boundaryFaces does not contain {boundaryName!r}')
    triangles = []
    for cellType, connectivity in mesh.boundaryFaces[boundaryName]:
        arr = np.asarray(connectivity, dtype=int)
        if cellType in _triangleTypes:
            triangles.append(arr[:, :3])
        elif cellType in _quadTypes:
            triangles.extend(quadRowToTwoTriangles(row[:4]) for row in arr)
        else:
            raise NotImplementedError(f'unsupported thermal boundary face type: {cellType}')
    if not triangles:
        return np.empty((0, 3), dtype=int)
    return np.concatenate(triangles, axis=0)


def _boundaryNodalValues(
    mesh: meshIr,
    boundaryName: str,
    triangles: np.ndarray,
    value: Any,
    name: str,
) -> np.ndarray:
    if not isinstance(value, thermalBoundaryField):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full((triangles.shape[0], 3), float(arr), dtype=float)
        flat = arr.reshape(-1)
        if flat.size != triangles.shape[0]:
            raise ValueError(f'{name} must be scalar or contain {triangles.shape[0]} face values')
        return np.repeat(flat[:, None], 3, axis=1)

    arr = np.asarray(value.values, dtype=float)
    if value.location == 'uniform':
        return np.full((triangles.shape[0], 3), float(arr), dtype=float)
    if value.location == 'face':
        if arr.ndim == 0:
            return np.full((triangles.shape[0], 3), float(arr), dtype=float)
        flat = arr.reshape(-1)
        if flat.size != triangles.shape[0]:
            raise ValueError(f'{name} face field must contain {triangles.shape[0]} values')
        return np.repeat(flat[:, None], 3, axis=1)
    if value.location == 'globalNode':
        if arr.ndim == 0:
            return np.full((triangles.shape[0], 3), float(arr), dtype=float)
        flat = arr.reshape(-1)
        if flat.size != mesh.nodes.shape[0]:
            raise ValueError(f'{name} globalNode field must contain {mesh.nodes.shape[0]} values')
        return flat[triangles]

    nodeIndices = mesh.boundaryNodeIndices.get(boundaryName)
    if nodeIndices is None:
        nodeIndices = np.unique(triangles.reshape(-1))
    if arr.ndim == 0:
        return np.full((triangles.shape[0], 3), float(arr), dtype=float)
    flat = arr.reshape(-1)
    if flat.size != nodeIndices.size:
        raise ValueError(f'{name} node field must contain {nodeIndices.size} values')
    globalValues = np.full(mesh.nodes.shape[0], np.nan, dtype=float)
    globalValues[nodeIndices] = flat
    faceValues = globalValues[triangles]
    if np.any(~np.isfinite(faceValues)):
        raise ValueError(f'{name} node field does not cover every boundary face node')
    return faceValues


def _contactFaceMapping(
    mesh: meshIr,
    name: str,
    interface: thermalContactInterface,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    primaryFaces = _boundaryTriangles(mesh, interface.primaryBoundary)
    if interface.secondaryBoundary not in mesh.boundaryNodeIndices:
        raise KeyError(
            f'boundaryNodeIndices does not contain {interface.secondaryBoundary!r} '
            f'for contact {name!r}'
        )
    primaryNodes = np.unique(primaryFaces.reshape(-1))
    secondaryNodes = np.asarray(mesh.boundaryNodeIndices[interface.secondaryBoundary], dtype=int)
    if primaryNodes.size == 0 or secondaryNodes.size == 0:
        raise ValueError(f'contact {name!r} requires non-empty primary and secondary boundaries')
    distances, localIndices = cKDTree(mesh.nodes[secondaryNodes]).query(mesh.nodes[primaryNodes], k=1)
    outside = distances > interface.maxDistance
    if np.any(outside):
        worst = float(np.max(distances[outside]))
        raise ValueError(
            f'contact {name!r} leaves {int(np.count_nonzero(outside))}/{primaryNodes.size} '
            f'primary nodes outside maxDistance={interface.maxDistance:g} m '
            f'(largest nearest distance {worst:g} m)'
        )
    pairedSecondary = secondaryNodes[np.asarray(localIndices, dtype=int)]
    requireUnique = interface.requireUniqueSecondary or interface.mapping == 'matching'
    if requireUnique and np.unique(pairedSecondary).size != pairedSecondary.size:
        raise ValueError(f'contact {name!r} does not produce a one-to-one secondary node mapping')
    if interface.mapping == 'matching' and primaryNodes.size != secondaryNodes.size:
        raise ValueError(
            f'contact {name!r} matching mapping requires equal boundary node counts '
            f'({primaryNodes.size} != {secondaryNodes.size})'
        )
    if np.any(primaryNodes == pairedSecondary):
        raise ValueError(
            f'contact {name!r} pairs shared global nodes; shared-node interfaces already imply ideal contact'
        )
    globalMapping = np.full(mesh.nodes.shape[0], -1, dtype=int)
    globalMapping[primaryNodes] = pairedSecondary
    secondaryFaces = globalMapping[primaryFaces]
    if np.any(secondaryFaces < 0):
        raise RuntimeError(f'internal contact mapping failure for {name!r}')
    coefficients = _boundaryNodalValues(
        mesh,
        interface.primaryBoundary,
        primaryFaces,
        interface.coefficient,
        f'contacts[{name!r}].coefficient',
    )
    diagnostic = {
        'primaryBoundary': interface.primaryBoundary,
        'secondaryBoundary': interface.secondaryBoundary,
        'mapping': interface.mapping,
        'primaryNodeCount': int(primaryNodes.size),
        'secondaryNodeCount': int(secondaryNodes.size),
        'mappedSecondaryNodeCount': int(np.unique(pairedSecondary).size),
        'coverage': 1.0,
        'maximumDistance': float(np.max(distances)),
        'meanDistance': float(np.mean(distances)),
        'oneToOne': bool(np.unique(pairedSecondary).size == pairedSecondary.size),
    }
    return primaryFaces, secondaryFaces, coefficients, diagnostic


def assembleMeshThermalOperators(
    mesh: meshIr,
    *,
    conductivity: Any,
    density: Any = 1.0,
    specificHeat: Any = 1.0,
    heatSource: Any = 0.0,
    capacityMode: str = 'consistent',
    convection: dict[str, convectionBoundary] | None = None,
    heatFlux: dict[str, Any] | None = None,
    contacts: dict[str, thermalContactInterface] | None = None,
    backendName: str = 'scipy',
) -> thermalOperatorSet:
    blockIndices, elements = _tetraBlocks(mesh)
    resolvedConductivity = _cellProperty(mesh, conductivity, 'conductivity', blockIndices)
    resolvedDensity = _cellProperty(mesh, density, 'density', blockIndices)
    resolvedSpecificHeat = _cellProperty(mesh, specificHeat, 'specificHeat', blockIndices)
    resolvedHeatSource = _cellProperty(
        mesh,
        heatSource,
        'heatSource',
        blockIndices,
        defaultValue=0.0,
    )

    convectionFaces = []
    convectionCoefficients = []
    ambientTemperatures = []
    for boundaryName, condition in (convection or {}).items():
        triangles = _boundaryTriangles(mesh, boundaryName)
        convectionFaces.append(triangles)
        convectionCoefficients.append(
            _boundaryNodalValues(
                mesh,
                boundaryName,
                triangles,
                condition.coefficient,
                f'convection[{boundaryName!r}].coefficient',
            )
        )
        ambientTemperatures.append(
            _boundaryNodalValues(
                mesh,
                boundaryName,
                triangles,
                condition.ambientTemperature,
                f'convection[{boundaryName!r}].ambientTemperature',
            )
        )

    heatFluxFaces = []
    heatFluxValues = []
    for boundaryName, value in (heatFlux or {}).items():
        triangles = _boundaryTriangles(mesh, boundaryName)
        heatFluxFaces.append(triangles)
        heatFluxValues.append(
            _boundaryNodalValues(
                mesh,
                boundaryName,
                triangles,
                value,
                f'heatFlux[{boundaryName!r}]',
            )
        )

    contactPrimaryFaces = []
    contactSecondaryFaces = []
    contactCoefficients = []
    contactDiagnostics: dict[str, dict[str, Any]] = {}
    for name, interface in (contacts or {}).items():
        primary, secondary, coefficient, diagnostic = _contactFaceMapping(mesh, name, interface)
        contactPrimaryFaces.append(primary)
        contactSecondaryFaces.append(secondary)
        contactCoefficients.append(coefficient)
        contactDiagnostics[name] = diagnostic

    result = assembleLinearTetraThermalOperators(
        mesh.nodes,
        elements,
        conductivity=resolvedConductivity,
        density=resolvedDensity,
        specificHeat=resolvedSpecificHeat,
        heatSource=resolvedHeatSource,
        capacityMode=capacityMode,
        convectionFaces=np.concatenate(convectionFaces, axis=0) if convectionFaces else None,
        convectionCoefficient=np.concatenate(convectionCoefficients, axis=0) if convectionCoefficients else None,
        ambientTemperature=np.concatenate(ambientTemperatures, axis=0) if ambientTemperatures else None,
        heatFluxFaces=np.concatenate(heatFluxFaces, axis=0) if heatFluxFaces else None,
        heatFlux=np.concatenate(heatFluxValues, axis=0) if heatFluxValues else None,
        contactPrimaryFaces=np.concatenate(contactPrimaryFaces, axis=0) if contactPrimaryFaces else None,
        contactSecondaryFaces=np.concatenate(contactSecondaryFaces, axis=0) if contactSecondaryFaces else None,
        contactCoefficient=np.concatenate(contactCoefficients, axis=0) if contactCoefficients else None,
        backendName=backendName,
    )
    return thermalOperatorSet(
        conductivity=result.conductivity,
        capacity=result.capacity,
        load=result.load,
        elementVolumes=result.elementVolumes,
        metadata={**result.metadata, 'contacts': contactDiagnostics},
    )
