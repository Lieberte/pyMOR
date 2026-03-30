import numpy as np
from .baseGeometry import baseGeometry

class meshGeometry(baseGeometry):
    def __init__(self, nodes: np.ndarray, boundaryNodes: dict[str, np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]] | None = None):
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

    def _mergeBoundaryNodes(self, target: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]) -> np.ndarray:
        if isinstance(target, (list, tuple)):
            return np.concatenate(target, axis=0)
        return target

    def randomBoundaryPoints(self, n: int, boundaryName: str = None) -> np.ndarray:
        if not self.boundaryNodes:
            raise ValueError('boundaryNodes is empty')
        if boundaryName and boundaryName in self.boundaryNodes:
            targetNodes = self._mergeBoundaryNodes(self.boundaryNodes[boundaryName])
        else:
            mergedGroups = [self._mergeBoundaryNodes(group) for group in self.boundaryNodes.values()]
            targetNodes = np.concatenate(mergedGroups, axis=0)
        replace = targetNodes.shape[0] < n
        indices = np.random.choice(targetNodes.shape[0], n, replace=replace)
        return targetNodes[indices]
