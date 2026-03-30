import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from .baseGeometry import baseGeometry
from .meshIr import meshIr

class meshGeometry(baseGeometry):
    def __init__(
        self,
        nodes: np.ndarray,
        boundaryNodes: dict[str, np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]] | None = None,
        *,
        insideMode: str = 'all',
        boundaryTol: float | None = None,
        hullTol: float | None = None,
    ):
        super().__init__(dim=nodes.shape[1])
        self.nodes = np.asarray(nodes, dtype=float)
        if self.nodes.ndim != 2:
            raise ValueError('nodes must be 2D')
        self.boundaryNodes = boundaryNodes or {}
        if insideMode not in ('all', 'convexHull'):
            raise ValueError("insideMode must be 'all' or 'convexHull'")
        self.insideMode = insideMode
        char = float(np.max(self.nodes.max(axis=0) - self.nodes.min(axis=0))) if self.nodes.size else 1.0
        self.boundaryTol = float(boundaryTol) if boundaryTol is not None else max(char * 1e-6, 1e-12)
        self.hullTol = float(hullTol) if hullTol is not None else max(char * 1e-9, 1e-12)
        self._hull: ConvexHull | None = None
        self._normShift: np.ndarray | None = None
        self._normScale: np.ndarray | None = None

    @classmethod
    def fromMeshIr(cls, ir: meshIr, **kwargs):
        return cls(ir.nodes, ir.boundaryNodes or None, **kwargs)

    @property
    def boundaryNames(self) -> list[str]:
        return list(self.boundaryNodes.keys())

    def boundingBox(self) -> tuple[np.ndarray, np.ndarray]:
        return self.nodes.min(axis=0).copy(), self.nodes.max(axis=0).copy()

    def _getHull(self) -> ConvexHull:
        if self._hull is None:
            if self.nodes.shape[0] < self.dim + 1:
                raise ValueError('need at least dim+1 nodes for convexHull')
            self._hull = ConvexHull(self.nodes)
        return self._hull

    def convexHullMeasure(self) -> float:
        return float(self._getHull().volume)

    def isInside(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if self.insideMode == 'all':
            return np.ones(x.shape[0], dtype=bool)
        eq = self._getHull().equations
        margins = x @ eq[:, :-1].T + eq[:, -1]
        return np.all(margins <= self.hullTol, axis=1)

    def _mergedAllBoundaryNodes(self) -> np.ndarray | None:
        if not self.boundaryNodes:
            return None
        mergedGroups = [self._mergeBoundaryNodes(group) for group in self.boundaryNodes.values()]
        return np.concatenate(mergedGroups, axis=0)

    def onBoundary(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        b = self._mergedAllBoundaryNodes()
        if b is None or b.size == 0:
            return np.zeros(x.shape[0], dtype=bool)
        tree = cKDTree(b)
        d, _ = tree.query(x, k=1)
        return d <= self.boundaryTol

    def randomPoints(self, n: int) -> np.ndarray:
        m = self.nodes.shape[0]
        replace = n > m
        indices = np.random.choice(m, n, replace=replace)
        return self.nodes[indices]

    def _mergeBoundaryNodes(self, target: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]) -> np.ndarray:
        if isinstance(target, (list, tuple)):
            return np.concatenate(target, axis=0)
        return target

    def randomBoundaryPoints(self, n: int, boundaryName: str | None = None) -> np.ndarray:
        if not self.boundaryNodes:
            raise ValueError('boundaryNodes is empty')
        if boundaryName is not None:
            if boundaryName not in self.boundaryNodes:
                raise KeyError(boundaryName)
            targetNodes = self._mergeBoundaryNodes(self.boundaryNodes[boundaryName])
        else:
            mergedGroups = [self._mergeBoundaryNodes(group) for group in self.boundaryNodes.values()]
            targetNodes = np.concatenate(mergedGroups, axis=0)
        replace = targetNodes.shape[0] < n
        indices = np.random.choice(targetNodes.shape[0], n, replace=replace)
        return targetNodes[indices]

    def fitUnitCube(self) -> None:
        mn, mx = self.boundingBox()
        span = mx - mn
        span = np.where(span < 1e-15, 1.0, span)
        self._normShift = mn.astype(float)
        self._normScale = span.astype(float)

    def refactorMesh(self, mesh: np.ndarray) -> np.ndarray:
        mesh = np.asarray(mesh, dtype=float)
        if self._normShift is None:
            self.fitUnitCube()
        shift = self._normShift
        scale = self._normScale
        return (mesh - shift) / scale

    def refactorBoundary(self, boundary: np.ndarray) -> np.ndarray:
        return self.refactorMesh(boundary)
