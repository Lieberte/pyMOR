from pathlib import Path
import numpy as np
import meshio
from ..meshIr import meshIr
from ..utils.facetUtils import appendTrianglesFromSurfaceBlock
from ..utils.facetUtils import extractQuadSubsetBoundaryLines
from ..utils.facetUtils import extractTetraSubsetBoundaryTriangles
from ..utils.facetUtils import extractTriangleSubsetBoundaryLines
from ..utils.facetUtils import mergeTriangleList
from ..utils.facetUtils import topoDim

def _mergeIndexGroups(groups: dict[str, np.ndarray], name: str, idx: np.ndarray) -> None:
    if idx.size == 0:
        return
    if name not in groups:
        groups[name] = idx
        return
    groups[name] = np.unique(np.concatenate([groups[name], idx]))

def _groupByCellType(mesh: meshio.Mesh) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {}
    for cellBlock in mesh.cells:
        grouped.setdefault(cellBlock.type, []).append(np.asarray(cellBlock.data, dtype=int))
    return {k: np.concatenate(v, axis=0) for k, v in grouped.items() if v}

def _iterCellSetBlocks(mesh: meshio.Mesh, groupedCells: dict[str, np.ndarray]):
    for name, typeToIdx in (mesh.cell_sets_dict or {}).items():
        for cellType, idxCell in typeToIdx.items():
            if cellType not in groupedCells:
                continue
            idx = np.asarray(idxCell, dtype=int)
            if idx.size == 0:
                continue
            yield name, cellType, idx, groupedCells[cellType]

def _cellSetNodeIndices(conn: np.ndarray, idxCell: np.ndarray) -> np.ndarray:
    return np.unique(conn[idxCell].ravel())

def _accumulateCellSets(
    mesh: meshio.Mesh,
    groupedCells: dict[str, np.ndarray],
    spatialDim: int,
    *,
    pointSetFilter: frozenset[str] | None = None,
    renameBoundaries: dict[str, str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, list[tuple[str, np.ndarray]]]]:
    boundaryNodeIdx: dict[str, np.ndarray] = {}
    triBuckets: dict[str, list[np.ndarray]] = {}
    lineBuckets: dict[str, list[np.ndarray]] = {}
    for name, cellType, idxCell, conn in _iterCellSetBlocks(mesh, groupedCells):
        if pointSetFilter is not None and name not in pointSetFilter:
            continue
        key = (renameBoundaries or {}).get(name, name)
        _mergeIndexGroups(boundaryNodeIdx, key, _cellSetNodeIndices(conn, idxCell))
        try:
            td = topoDim(cellType)
        except KeyError:
            continue
        if td == spatialDim - 1:
            if spatialDim == 3:
                parts = appendTrianglesFromSurfaceBlock(cellType, conn, idxCell)
                for p in parts:
                    triBuckets.setdefault(key, []).append(p)
            elif spatialDim == 2 and cellType == 'line':
                sub = conn[idxCell]
                lineBuckets.setdefault(key, []).append(sub)
        elif td == spatialDim and spatialDim == 3 and cellType in ('tetra', 'tetra4', 'tetra10'):
            ext = extractTetraSubsetBoundaryTriangles(conn, idxCell)
            if ext.shape[0] > 0:
                triBuckets.setdefault(key, []).append(ext)
        elif td == spatialDim and spatialDim == 2 and cellType in ('triangle', 'triangle6', 'triangle10'):
            ext = extractTriangleSubsetBoundaryLines(conn, idxCell)
            if ext.shape[0] > 0:
                lineBuckets.setdefault(key, []).append(ext)
        elif td == spatialDim and spatialDim == 2 and cellType in ('quad', 'quad8', 'quad9'):
            ext = extractQuadSubsetBoundaryLines(conn, idxCell)
            if ext.shape[0] > 0:
                lineBuckets.setdefault(key, []).append(ext)
    out: dict[str, list[tuple[str, np.ndarray]]] = {}
    for key, chunks in triBuckets.items():
        merged = mergeTriangleList(chunks)
        if merged.shape[0] > 0:
            out.setdefault(key, []).append(('triangle', merged))
    for key, chunks in lineBuckets.items():
        merged = np.concatenate(chunks, axis=0)
        if merged.shape[0] > 0:
            out.setdefault(key, []).append(('line', merged))
    return boundaryNodeIdx, out

def meshIoToIr(
    mesh: meshio.Mesh,
    *,
    pointSetFilter: frozenset[str] | None = None,
    renameBoundaries: dict[str, str] | None = None,
    useCellSetsIfNoPointSets: bool = True,
    fillBoundaryFacesFromCellSets: bool = True,
) -> meshIr:
    nodes = np.asarray(mesh.points, dtype=float)
    spatialDim = int(nodes.shape[1])
    groupedCells = _groupByCellType(mesh)
    pointNodeIdx: dict[str, np.ndarray] = {
        (renameBoundaries or {}).get(name, name): np.asarray(indices, dtype=int)
        for name, indices in (mesh.point_sets or {}).items()
        if pointSetFilter is None or name in pointSetFilter
    }
    cellNodeIdx, cellBoundaryFaces = _accumulateCellSets(
        mesh,
        groupedCells,
        spatialDim,
        pointSetFilter=pointSetFilter,
        renameBoundaries=renameBoundaries,
    )
    boundaryNodeIdx = dict(pointNodeIdx)
    if pointNodeIdx:
        for name, idx in cellNodeIdx.items():
            _mergeIndexGroups(boundaryNodeIdx, name, idx)
    elif useCellSetsIfNoPointSets:
        boundaryNodeIdx = cellNodeIdx
    boundaryNodes: dict[str, np.ndarray] = {name: nodes[idx] for name, idx in boundaryNodeIdx.items()}
    boundaryFaces = cellBoundaryFaces if fillBoundaryFacesFromCellSets else {}
    cells = [(c.type, np.asarray(c.data)) for c in mesh.cells] if len(mesh.cells) > 0 else None
    return meshIr(nodes=nodes, boundaryNodes=boundaryNodes, cells=cells, boundaryFaces=boundaryFaces)

def meshFileToIr(
    path: str | Path,
    *,
    meshioReadKwargs: dict | None = None,
    **kwargs,
) -> meshIr:
    resolved = Path(path).expanduser()
    opts: dict = dict(meshioReadKwargs or {})
    mesh = meshio.read(resolved, **opts)
    return meshIoToIr(mesh, **kwargs)
