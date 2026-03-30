import numpy as np
from abc import ABC, abstractmethod

class baseGeometry(ABC):
    def __init__(self, dim: int):
        self.dim = dim

    @abstractmethod
    def isInside(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def onBoundary(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def randomPoints(self, n: int) -> np.ndarray:
        pass

    @abstractmethod
    def randomBoundaryPoints(self, n: int, boundaryName: str | None = None) -> np.ndarray:
        pass
