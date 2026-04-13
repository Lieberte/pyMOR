import numpy as np
import meshio
from ..meshIr import meshIr

def _uniqueNodesFromCells(conn: np.ndarray, idxCell: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    idxCell = np.asarray(idxCell, dtype=int)
    if idxCell.size == 0:
        return np.empty((0, nodes.shape[1]), dtype=float)
    conn = np.asarray(conn)
    verts = np.unique(conn[idxCell].ravel())
    return nodes[verts]

def _cellSetsToBoundaryNodes(
    mesh: meshio.Mesh,
    *,
    pointSetFilter: frozenset[str] | None = None,
    renameBoundaries: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    nodes = np.asarray(mesh.points, dtype=float)
    out: dict[str, np.ndarray] = {}
    csd = mesh.cell_sets_dict or {}
    cellList = mesh.cells
    for name, perBlock in csd.items():
        if pointSetFilter is not None and name not in pointSetFilter:
            continue
        chunks: list[np.ndarray] = []
        if isinstance(perBlock, dict) and perBlock:
            vals = list(perBlock.values())
            if vals and isinstance(vals[0], np.ndarray):
                for cellType, idxCell in perBlock.items():
                    idxCell = np.asarray(idxCell, dtype=int)
                    if idxCell.size == 0:
                        continue
                    rows: list[np.ndarray] = []
                    for cellBlock in cellList:
                        if cellBlock.type != cellType:
                            continue
                        rows.append(np.asarray(cellBlock.data))
                    if not rows:
                        continue
                    conn = np.concatenate(rows, axis=0)
                    chunk = _uniqueNodesFromCells(conn, idxCell, nodes)
                    if chunk.shape[0] > 0:
                        chunks.append(chunk)
        elif isinstance(perBlock, (list, tuple)):
            for bi, cellBlock in enumerate(cellList):
                if bi >= len(perBlock):
                    continue
                idxCell = np.asarray(perBlock[bi], dtype=int)
                if idxCell.size == 0:
                    continue
                conn = np.asarray(cellBlock.data)
                chunks.append(_uniqueNodesFromCells(conn, idxCell, nodes))
        if not chunks:
            continue
        key = renameBoundaries[name] if renameBoundaries and name in renameBoundaries else name
        out[key] = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
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
        boundaryNodes = _cellSetsToBoundaryNodes(
            mesh,
            pointSetFilter=pointSetFilter,
            renameBoundaries=renameBoundaries,
        )
    cells = None
    if mesh.cells:
        cellList = mesh.cells
        cells = [(c.type, np.asarray(c.data)) for c in cellList]
    return meshIr(nodes=nodes, boundaryNodes=boundaryNodes, cells=cells)
