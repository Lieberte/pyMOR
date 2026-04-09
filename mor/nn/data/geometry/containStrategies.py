import numpy as np
from abc import ABC, abstractmethod

class baseContainStrategy(ABC):
    @abstractmethod
    def contains(self, domain, x: np.ndarray) -> np.ndarray:
        pass

class allContainStrategy(baseContainStrategy):
    def contains(self, domain, x: np.ndarray) -> np.ndarray:
        return np.ones(x.shape[0], dtype=bool)

class convexHullContainStrategy(baseContainStrategy):
    def contains(self, domain, x: np.ndarray) -> np.ndarray:
        eq = domain._getHull().equations
        margins = x @ eq[:, :-1].T + eq[:, -1]
        return np.all(margins <= domain.hullTol, axis=1)
