import numpy as np
from abc import ABC, abstractmethod
from functools import partial
from .sampleBatch import sampleBatch

class baseDomain(ABC):
    interiorRegionName = 'interior'

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

    def sampleInteriorBatch(self, n: int, regionName: str = interiorRegionName, **kwargs) -> sampleBatch:
        x = self.sampleInterior(n, **kwargs)
        regionNames = np.full(x.shape[0], regionName, dtype=object)
        return sampleBatch(x=x, regionNames=regionNames)

    def _samplerForRegion(self, regionName: str, *, batch: bool):
        interior = regionName == self.interiorRegionName
        if batch:
            return (
                partial(self.sampleInteriorBatch, regionName=regionName)
                if interior
                else partial(self.sampleBoundaryBatch, boundaryName=regionName)
            )
        return self.sampleInterior if interior else partial(self.sampleBoundary, boundaryName=regionName)

    def sampleRegion(self, n: int, regionName: str, **kwargs) -> np.ndarray:
        return self._samplerForRegion(regionName, batch=False)(n, **kwargs)

    def sampleRegionBatch(self, n: int, regionName: str, **kwargs) -> sampleBatch:
        return self._samplerForRegion(regionName, batch=True)(n, **kwargs)

    def sampleMixedBatch(self, sampleCountByRegion: dict[str, int], **kwargs) -> sampleBatch:
        batches: list[sampleBatch] = []
        for regionName, n in sampleCountByRegion.items():
            if int(n) <= 0:
                continue
            batches.append(self.sampleRegionBatch(int(n), regionName=regionName, **kwargs))
        if not batches:
            raise ValueError('sampleCountByRegion must contain a positive sample count')
        return sampleBatch.concat(batches)

    def sampleBoundaryBatch(self, n: int, boundaryName: str | None = None, **kwargs) -> sampleBatch:
        x = self.sampleBoundary(n, boundaryName=boundaryName, **kwargs)
        if boundaryName is None:
            regionNames = self.locateBoundary(x)
        else:
            regionNames = np.full(x.shape[0], boundaryName, dtype=object)
        normals = self.boundaryNormal(x, boundaryName=boundaryName)
        return sampleBatch(x=x, regionNames=regionNames, normals=normals)
