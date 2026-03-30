from __future__ import annotations
import numpy as np
import meshio
from ..meshIr import meshIr

def _cellSetsToBoundaryNodes(mesh: meshio.Mesh) -> dict[str, np.ndarray]:
    nodes = np.asarray(mesh.points, dtype=float)
    out: dict[str, np.ndarray] = {}
    csd = mesh.cell_sets_dict or {}
    cellList = mesh.cells
    for name, perBlock in csd.items():
        chunks: list[np.ndarray] = []
        for bi, cellBlock in enumerate(cellList):
            if bi >= len(perBlock):
                continue
            idxCell = np.asarray(perBlock[bi], dtype=int)
            if idxCell.size == 0:
                continue
            conn = np.asarray(cellBlock.data)
            verts = np.unique(conn[idxCell].ravel())
            chunks.append(nodes[verts])
        if not chunks:
            continue
        out[name] = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
    return out

def meshIoToIr(
    mesh: meshio.Mesh,
    *,
    pointSetFilter: frozenset[str] | None = None,
    renameBoundaries: dict[str, str] | None = None,
    useCellSetsIfNoPointSets: bool = True,
) -> meshIr:
    nodes = np.asarray(mesh.points, dtype=float)
    boundaryNodes: dict[str, np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]] = {}
    ps = mesh.point_sets or {}
    for name, indices in ps.items():
        if pointSetFilter is not None and name not in pointSetFilter:
            continue
        key = renameBoundaries[name] if renameBoundaries and name in renameBoundaries else name
        idx = np.asarray(indices, dtype=int)
        boundaryNodes[key] = nodes[idx]
    if not boundaryNodes and useCellSetsIfNoPointSets:
        boundaryNodes = _cellSetsToBoundaryNodes(mesh)
    cells = None
    if mesh.cells:
        cellList = mesh.cells
        cells = [(c.type, np.asarray(c.data)) for c in cellList]
    return meshIr(nodes=nodes, boundaryNodes=boundaryNodes, cells=cells)
