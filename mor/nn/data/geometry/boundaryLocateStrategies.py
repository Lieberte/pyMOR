import numpy as np
from abc import ABC, abstractmethod

class baseBoundaryLocateStrategy(ABC):
    @abstractmethod
    def locateBoundary(self, domain, x: np.ndarray) -> np.ndarray:
        pass

class nearestBoundaryLocateStrategy(baseBoundaryLocateStrategy):
    def locateBoundary(self, domain, x: np.ndarray) -> np.ndarray:
        located = np.empty(x.shape[0], dtype=object)
        located.fill(None)
        distances = np.full(x.shape[0], np.inf, dtype=float)
        for name in domain.boundaryNames:
            nodes = domain.boundaryNodes[name]
            if nodes.shape[0] == 0:
                continue
            tree = domain._getBoundaryTree(name)
            d, _ = tree.query(x, k=1)
            mask = d <= domain.boundaryTol
            better = mask & (d < distances)
            located[better] = name
            distances[better] = d[better]
        return located
