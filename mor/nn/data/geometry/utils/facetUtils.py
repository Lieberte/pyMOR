import numpy as np

cellTopologicalDimension = {
    'vertex': 0,
    'line': 1,
    'line3': 1,
    'line4': 1,
    'line5': 1,
    'line6': 1,
    'triangle': 2,
    'triangle6': 2,
    'triangle10': 2,
    'quad': 2,
    'quad8': 2,
    'quad9': 2,
    'tetra': 3,
    'tetra4': 3,
    'tetra10': 3,
    'hexahedron': 3,
    'hexahedron20': 3,
    'wedge': 3,
    'wedge15': 3,
    'pyramid': 3,
    'pyramid14': 3,
}

def topoDim(cellType: str) -> int:
    return cellTopologicalDimension[cellType]

def extractTetraSubsetBoundaryTriangles(conn: np.ndarray, idxCell: np.ndarray) -> np.ndarray:
    conn = np.asarray(conn, dtype=int)
    if conn.ndim == 2 and conn.shape[1] > 4:
        conn = conn[:, :4]
    idxCell = np.asarray(idxCell, dtype=int)
    if idxCell.size == 0:
        return np.empty((0, 3), dtype=int)
    rows = conn[idxCell]
    v0, v1, v2, v3 = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    faces = np.vstack(
        [
            np.column_stack([v0, v1, v2]),
            np.column_stack([v0, v1, v3]),
            np.column_stack([v0, v2, v3]),
            np.column_stack([v1, v2, v3]),
        ]
    )
    facesSorted = np.sort(faces, axis=1)
    _, inv, counts = np.unique(facesSorted, axis=0, return_inverse=True, return_counts=True)
    return faces[counts[inv] == 1]

def _subsetBoundaryEdges(conn: np.ndarray, idxCell: np.ndarray, edgePattern: list[tuple[int, int]]) -> np.ndarray:
    conn = np.asarray(conn, dtype=int)
    idxCell = np.asarray(idxCell, dtype=int)
    if idxCell.size == 0:
        return np.empty((0, 2), dtype=int)
    rows = conn[idxCell]
    chunks = [np.column_stack([rows[:, a], rows[:, b]]) for a, b in edgePattern]
    edges = np.vstack(chunks)
    edgesSorted = np.sort(edges, axis=1)
    _, inv, counts = np.unique(edgesSorted, axis=0, return_inverse=True, return_counts=True)
    return edges[counts[inv] == 1]

def extractTriangleSubsetBoundaryLines(conn: np.ndarray, idxCell: np.ndarray) -> np.ndarray:
    conn = np.asarray(conn, dtype=int)
    if conn.ndim == 2 and conn.shape[1] > 3:
        conn = conn[:, :3]
    return _subsetBoundaryEdges(conn, idxCell, [(0, 1), (1, 2), (2, 0)])

def extractQuadSubsetBoundaryLines(conn: np.ndarray, idxCell: np.ndarray) -> np.ndarray:
    conn = np.asarray(conn, dtype=int)
    if conn.ndim == 2 and conn.shape[1] > 4:
        conn = conn[:, :4]
    return _subsetBoundaryEdges(conn, idxCell, [(0, 1), (1, 2), (2, 3), (3, 0)])

def quadRowToTwoTriangles(row: np.ndarray) -> np.ndarray:
    a, b, c, d = int(row[0]), int(row[1]), int(row[2]), int(row[3])
    return np.array([[a, b, c], [a, c, d]], dtype=int)

def appendTrianglesFromSurfaceBlock(cellType: str, conn: np.ndarray, idxCell: np.ndarray) -> list[np.ndarray]:
    sub = np.asarray(conn, dtype=int)[np.asarray(idxCell, dtype=int)]
    if cellType in ('triangle', 'triangle6', 'triangle10'):
        if sub.ndim != 2 or sub.shape[1] < 3:
            return []
        return [sub[:, :3]]
    if cellType in ('quad', 'quad8', 'quad9'):
        return [quadRowToTwoTriangles(row[:4]) for row in sub]
    return []

def mergeTriangleList(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.empty((0, 3), dtype=int)
    return np.concatenate(chunks, axis=0)

def validateBoundaryFacesDict(
    boundaryFaces: dict[str, list[tuple[str, np.ndarray]]] | None,
    numNodes: int,
) -> dict[str, list[tuple[str, np.ndarray]]]:
    if not boundaryFaces:
        return {}
    out: dict[str, list[tuple[str, np.ndarray]]] = {}
    for name, blocks in boundaryFaces.items():
        cleaned: list[tuple[str, np.ndarray]] = []
        for cellType, conn in blocks:
            arr = np.asarray(conn, dtype=int)
            if arr.size == 0:
                continue
            if arr.min() < 0 or arr.max() >= numNodes:
                raise ValueError(f'boundaryFaces[{name!r}] {cellType} index out of range')
            cleaned.append((cellType, arr))
        if cleaned:
            out[name] = cleaned
    return out

def uniqueRowsStacked(a: np.ndarray, b: np.ndarray, decimals: int = 9) -> np.ndarray:
    if a.size == 0:
        return b
    if b.size == 0:
        return a
    m = np.vstack([a, b])
    r = np.round(m, decimals=decimals)
    _, idx = np.unique(r, axis=0, return_index=True)
    return m[np.sort(idx)]

class facetPatch:
    def __init__(self, nodes: np.ndarray, blocks: list[tuple[str, np.ndarray]]):
        self.dim = int(nodes.shape[1])
        self.nodes = nodes
        triChunks: list[np.ndarray] = []
        lineChunks: list[np.ndarray] = []
        for cellType, conn in blocks:
            c = np.asarray(conn, dtype=int)
            if c.size == 0:
                continue
            if self.dim == 3 and cellType in ('triangle', 'triangle6', 'triangle10') and c.ndim == 2 and c.shape[1] >= 3:
                triChunks.append(c[:, :3])
            elif self.dim == 3 and cellType in ('quad', 'quad8', 'quad9') and c.ndim == 2 and c.shape[1] >= 4:
                for row in c:
                    triChunks.append(quadRowToTwoTriangles(row[:4]))
            elif self.dim == 2 and cellType in ('line', 'line3', 'line4', 'line5', 'line6') and c.ndim == 2 and c.shape[1] >= 2:
                lineChunks.append(c[:, :2])
        self.triangles = mergeTriangleList(triChunks) if triChunks else np.empty((0, 3), dtype=int)
        self.lines = np.concatenate(lineChunks, axis=0) if lineChunks else np.empty((0, 2), dtype=int)
        self._centroids: np.ndarray | None = None
        self._normals: np.ndarray | None = None
        self._areas: np.ndarray | None = None
        self._tree = None
        if self.triangles.shape[0] > 0:
            p0 = nodes[self.triangles[:, 0]]
            p1 = nodes[self.triangles[:, 1]]
            p2 = nodes[self.triangles[:, 2]]
            self._centroids = (p0 + p1 + p2) / 3.0
            e1 = p1 - p0
            e2 = p2 - p0
            nrm = np.cross(e1, e2)
            a = 0.5 * np.linalg.norm(nrm, axis=1, keepdims=True)
            self._areas = a.ravel()
            self._normals = nrm / (2.0 * a + 1e-30)
        elif self.lines.shape[0] > 0:
            p0 = nodes[self.lines[:, 0]]
            p1 = nodes[self.lines[:, 1]]
            self._centroids = 0.5 * (p0 + p1)
            e = p1 - p0
            le = np.linalg.norm(e, axis=1, keepdims=True)
            self._areas = le.ravel()
            nrm = np.concatenate([-e[:, 1:2], e[:, 0:1]], axis=1) / (le + 1e-30)
            center = np.mean(nodes, axis=0, keepdims=True)
            outward = self._centroids - center
            flip = np.sum(nrm * outward, axis=1, keepdims=True) < 0
            self._normals = np.where(flip, -nrm, nrm)

    @property
    def totalArea(self) -> float:
        if self._areas is None:
            return 0.0
        return float(np.sum(self._areas))

    def isEmpty(self) -> bool:
        return self.triangles.shape[0] == 0 and self.lines.shape[0] == 0

    def cornerIndices(self) -> np.ndarray:
        if not self.isEmpty():
            if self.triangles.shape[0] > 0:
                return np.unique(self.triangles.ravel())
            return np.unique(self.lines.ravel())
        return np.empty((0,), dtype=int)

    def cornerCoordinates(self) -> np.ndarray:
        idx = self.cornerIndices()
        if idx.size == 0:
            return np.empty((0, self.dim), dtype=float)
        return self.nodes[idx]

    def _ensureTree(self):
        if self._tree is None and self._centroids is not None and self._centroids.shape[0] > 0:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self._centroids)

    def normalsAtPoints(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if self.isEmpty() or self._normals is None:
            return np.zeros_like(x)
        self._ensureTree()
        if self._tree is None:
            return np.zeros_like(x)
        _, j = self._tree.query(x, k=1)
        j = np.asarray(j, dtype=int).reshape(-1)
        return self._normals[j]

    def samplePoints(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        from .geometryUtils import validateSampleCount
        count = validateSampleCount(n)
        if self.isEmpty():
            return np.empty((0, self.dim), dtype=float), np.empty((0, self.dim), dtype=float)
        if self.triangles.shape[0] > 0:
            w = self._areas / (np.sum(self._areas) + 1e-30)
            pick = np.random.choice(self.triangles.shape[0], size=count, p=w)
            tris = self.triangles[pick]
            p0 = self.nodes[tris[:, 0]]
            p1 = self.nodes[tris[:, 1]]
            p2 = self.nodes[tris[:, 2]]
            r = np.random.rand(count, 2)
            sr = np.sqrt(r[:, 0:1])
            u = 1.0 - sr
            v = sr * (1.0 - r[:, 1:2])
            wv = sr * r[:, 1:2]
            pts = u * p0 + v * p1 + wv * p2
            nrm = self._normals[pick] if self._normals is not None else np.zeros_like(pts)
            return pts, nrm
        p0 = self.nodes[self.lines[:, 0]]
        p1 = self.nodes[self.lines[:, 1]]
        w = self._areas / (np.sum(self._areas) + 1e-30)
        pick = np.random.choice(self.lines.shape[0], size=count, p=w)
        t = np.random.rand(count, 1)
        pts = (1.0 - t) * p0[pick] + t * p1[pick]
        nrm = self._normals[pick] if self._normals is not None else np.zeros_like(pts)
        return pts, nrm
