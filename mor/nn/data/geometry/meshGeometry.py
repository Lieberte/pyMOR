import numpy as np
from .baseGeometry import baseGeometry

class meshGeometry(baseGeometry):
    def __init__(self, nodes: np.ndarray, boundaryNodes: dict[str, np.ndarray] = None):
        super().__init__(dim=nodes.shape[1])
        self.nodes = nodes
        self.boundaryNodes = boundaryNodes or {}

    def isInside(self, x: np.ndarray) -> np.ndarray:
        return np.ones(x.shape[0], dtype=bool)

    def onBoundary(self, x: np.ndarray) -> np.ndarray:
        # TODO: Implement expensive boundary check or use pre-indexed nodes
        pass

    def randomPoints(self, n: int) -> np.ndarray:
        indices = np.random.choice(self.nodes.shape[0], n, replace=False)
        return self.nodes[indices]

    def randomBoundaryPoints(self, n: int, boundaryName: str = None) -> np.ndarray:
        if boundaryName and boundaryName in self.boundaryNodes:
            targetNodes = self.boundaryNodes[boundaryName]
        else:
            targetNodes = np.concatenate(list(self.boundaryNodes.values()), axis=0)
            
        indices = np.random.choice(targetNodes.shape[0], n, replace=False)
        return targetNodes[indices]
