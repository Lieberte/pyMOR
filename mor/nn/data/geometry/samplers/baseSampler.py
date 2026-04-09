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

    def sampleInteriorBatch(self, domain, n: int, regionName: str = 'interior') -> sampleBatch:
        x = self.sampleInterior(domain, n)
        regionNames = np.full(x.shape[0], regionName, dtype=object)
        return sampleBatch(x=x, regionNames=regionNames)

    def sampleBoundaryBatch(self, domain, n: int, boundaryName: str | None = None) -> sampleBatch:
        x = self.sampleBoundary(domain, n, boundaryName=boundaryName)
        regionName = boundaryName if boundaryName is not None else 'boundary'
        regionNames = np.full(x.shape[0], regionName, dtype=object)
        return sampleBatch(x=x, regionNames=regionNames)
