import numpy as np
from abc import ABC, abstractmethod
from .sampleBatch import sampleBatch

class baseDomain(ABC):
    def __init__(self, dim: int):
        self.dim = dim

    @property
    def boundaryNames(self) -> list[str]:
        return []

    @abstractmethod
    def contains(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def locateBoundary(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def boundaryNormal(self, x: np.ndarray, boundaryName: str | None = None) -> np.ndarray:
        pass

    @abstractmethod
    def sampleInterior(self, n: int, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def sampleBoundary(self, n: int, boundaryName: str | None = None, **kwargs) -> np.ndarray:
        pass

    def boundaryMask(self, x: np.ndarray, boundaryName: str | None = None) -> np.ndarray:
        located = self.locateBoundary(x)
        if boundaryName is None:
            return np.fromiter((item is not None for item in located), dtype=bool, count=located.shape[0])
        return located == boundaryName

    def sampleInteriorBatch(self, n: int, regionName: str = 'interior', **kwargs) -> sampleBatch:
        x = self.sampleInterior(n, **kwargs)
        regionNames = np.full(x.shape[0], regionName, dtype=object)
        return sampleBatch(x=x, regionNames=regionNames)

    def sampleRegion(self, n: int, regionName: str, **kwargs) -> np.ndarray:
        if regionName == 'interior':
            return self.sampleInterior(n, **kwargs)
        return self.sampleBoundary(n, boundaryName=regionName, **kwargs)

    def sampleRegionBatch(self, n: int, regionName: str, **kwargs) -> sampleBatch:
        if regionName == 'interior':
            return self.sampleInteriorBatch(n, regionName=regionName, **kwargs)
        return self.sampleBoundaryBatch(n, boundaryName=regionName, **kwargs)

    def sampleBoundaryBatch(self, n: int, boundaryName: str | None = None, **kwargs) -> sampleBatch:
        x = self.sampleBoundary(n, boundaryName=boundaryName, **kwargs)
        if boundaryName is None:
            regionNames = self.locateBoundary(x)
        else:
            regionNames = np.full(x.shape[0], boundaryName, dtype=object)
        normals = self.boundaryNormal(x, boundaryName=boundaryName)
        return sampleBatch(x=x, regionNames=regionNames, normals=normals)
