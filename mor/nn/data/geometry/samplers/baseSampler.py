import numpy as np
from abc import ABC, abstractmethod
from ..sampleBatch import sampleBatch

class baseSampler(ABC):
    @abstractmethod
    def sampleInterior(self, domain, n: int) -> np.ndarray:
        pass

    @abstractmethod
    def sampleBoundary(self, domain, n: int, boundaryName: str | None = None) -> np.ndarray:
        pass

    def sampleRegion(self, domain, n: int, regionName: str) -> np.ndarray:
        if regionName == 'interior':
            return self.sampleInterior(domain, n)
        return self.sampleBoundary(domain, n, boundaryName=regionName)

    def sampleInteriorBatch(self, domain, n: int, regionName: str = 'interior') -> sampleBatch:
        x = self.sampleInterior(domain, n)
        regionNames = np.full(x.shape[0], regionName, dtype=object)
        return sampleBatch(x=x, regionNames=regionNames)

    def sampleRegionBatch(self, domain, n: int, regionName: str) -> sampleBatch:
        if regionName == 'interior':
            return self.sampleInteriorBatch(domain, n, regionName=regionName)
        return self.sampleBoundaryBatch(domain, n, boundaryName=regionName)

    def sampleBoundaryBatch(self, domain, n: int, boundaryName: str | None = None) -> sampleBatch:
        x = self.sampleBoundary(domain, n, boundaryName=boundaryName)
        if boundaryName is None:
            regionNames = domain.locateBoundary(x)
        else:
            regionNames = np.full(x.shape[0], boundaryName, dtype=object)
        normals = domain.boundaryNormal(x, boundaryName=boundaryName)
        return sampleBatch(x=x, regionNames=regionNames, normals=normals)

    def sampleMixedBatch(self, domain, sampleCountByRegion: dict[str, int]) -> sampleBatch:
        batches: list[sampleBatch] = []
        for regionName, n in sampleCountByRegion.items():
            if int(n) <= 0:
                continue
            batches.append(self.sampleRegionBatch(domain, int(n), regionName=regionName))
        if not batches:
            raise ValueError('sampleCountByRegion must contain a positive sample count')
        return sampleBatch.concat(batches)
