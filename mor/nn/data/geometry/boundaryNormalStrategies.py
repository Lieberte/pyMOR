import numpy as np
from abc import ABC, abstractmethod

class baseBoundaryNormalStrategy(ABC):
    @abstractmethod
    def boundaryNormal(self, domain, x: np.ndarray, boundaryName: str | None = None) -> np.ndarray:
        pass

class localPcaBoundaryNormalStrategy(baseBoundaryNormalStrategy):
    def __init__(self, kNeighbors: int = 8):
        self.kNeighbors = int(kNeighbors)

    def boundaryNormal(self, domain, x: np.ndarray, boundaryName: str | None = None) -> np.ndarray:
        if x.shape[0] == 0:
            return np.empty((0, domain.dim), dtype=float)
        if boundaryName is not None:
            return self._boundaryNormalForName(domain, x, boundaryName)
        located = domain.locateBoundary(x)
        if np.any(located == None):
            raise ValueError('boundaryName is required for points outside boundary tolerance')
        normals = np.empty_like(x, dtype=float)
        for name in np.unique(located):
            mask = located == name
            normals[mask] = self._boundaryNormalForName(domain, x[mask], boundaryName=name)
        return normals

    def _boundaryNormalForName(self, domain, x: np.ndarray, boundaryName: str) -> np.ndarray:
        boundaryPoints = domain.boundaryPoints(boundaryName=boundaryName)
        tree = domain._getBoundaryTree(boundaryName)
        kNeighbors = min(max(self.kNeighbors, domain.dim), boundaryPoints.shape[0])
        _, idx = tree.query(x, k=kNeighbors)
        idx = np.asarray(idx, dtype=int)
        if idx.ndim == 1:
            idx = idx[:, None]
        domainCenter = domain.nodes.mean(axis=0)
        normals = np.empty_like(x, dtype=float)
        for i in range(x.shape[0]):
            neighborhood = boundaryPoints[idx[i]]
            centered = neighborhood - neighborhood.mean(axis=0)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normal = vh[-1]
            norm = np.linalg.norm(normal)
            if norm == 0:
                normal = x[i] - domainCenter
                norm = np.linalg.norm(normal)
            if norm == 0:
                raise ValueError('cannot estimate boundary normal')
            normal = normal / norm
            if np.dot(normal, x[i] - domainCenter) < 0:
                normal = -normal
            normals[i] = normal
        return normals
