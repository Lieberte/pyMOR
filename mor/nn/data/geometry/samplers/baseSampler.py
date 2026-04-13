import numpy as np
from abc import ABC, abstractmethod
from functools import partial
from ..baseDomain import baseDomain
from ..sampleBatch import sampleBatch
from ..utils import expandSampleWeights
from ..utils import normalizeSampleCountMap
from ..utils import splitSampleWeightMap

class baseSampler(ABC):
    interiorRegionName = baseDomain.interiorRegionName

    @abstractmethod
    def sampleInterior(self, domain, n: int) -> np.ndarray:
        pass

    @abstractmethod
    def sampleBoundary(self, domain, n: int, boundaryName: str | None = None) -> np.ndarray:
        pass

    def _samplerForRegion(self, regionName: str, *, batch: bool):
        interior = regionName == self.interiorRegionName
        if batch:
            return (
                partial(self.sampleInteriorBatch, regionName=regionName)
                if interior
                else partial(self.sampleBoundaryBatch, boundaryName=regionName)
            )
        return self.sampleInterior if interior else partial(self.sampleBoundary, boundaryName=regionName)

    def sampleRegion(self, domain, n: int, regionName: str) -> np.ndarray:
        return self._samplerForRegion(regionName, batch=False)(domain, n)

    def sampleInteriorBatch(self, domain, n: int, regionName: str = interiorRegionName) -> sampleBatch:
        x = self.sampleInterior(domain, n)
        regionNames = np.full(x.shape[0], regionName, dtype=object)
        return sampleBatch(x=x, regionNames=regionNames)

    def sampleRegionBatch(self, domain, n: int, regionName: str) -> sampleBatch:
        return self._samplerForRegion(regionName, batch=True)(domain, n)

    def sampleBoundaryBatch(self, domain, n: int, boundaryName: str | None = None) -> sampleBatch:
        x = self.sampleBoundary(domain, n, boundaryName=boundaryName)
        if boundaryName is None:
            regionNames = domain.locateBoundary(x)
        else:
            regionNames = np.full(x.shape[0], boundaryName, dtype=object)
        normals = domain.boundaryNormal(x, boundaryName=boundaryName)
        return sampleBatch(x=x, regionNames=regionNames, normals=normals)

    def sampleMixedBatch(self, domain, sampleCountByRegion: dict[str, int]) -> sampleBatch:
        sampleCountByRegion = normalizeSampleCountMap(
            sampleCountByRegion,
            emptyError='sampleCountByRegion must contain a positive sample count',
        )
        batches = [self.sampleRegionBatch(domain, n, regionName=regionName) for regionName, n in sampleCountByRegion.items()]
        return sampleBatch.concat(batches)

    def sampleBoundaryMixedBatch(self, domain, sampleCountByBoundary: dict[str, int]) -> sampleBatch:
        sampleCountByBoundary = normalizeSampleCountMap(
            sampleCountByBoundary,
            emptyError='sampleCountByBoundary must contain a positive sample count',
        )
        batches = [self.sampleBoundaryBatch(domain, n, boundaryName=boundaryName) for boundaryName, n in sampleCountByBoundary.items()]
        return sampleBatch.concat(batches)

    def sampleWeightedBatch(self, domain, n: int, regionWeights: dict[str, float] | None = None) -> sampleBatch:
        activeRegionWeights = dict(regionWeights) if regionWeights is not None else {}
        if not activeRegionWeights:
            activeRegionWeights = {self.interiorRegionName: 1.0, **{name: 1.0 for name in domain.boundaryNames}}
        regionNames, weights, sampleCountByRegion = splitSampleWeightMap(n, activeRegionWeights)
        batch = self.sampleMixedBatch(domain, sampleCountByRegion)
        return batch.withWeights(expandSampleWeights(regionNames, sampleCountByRegion, weights))

    def sampleBoundaryWeightedBatch(self, domain, n: int, boundaryWeights: dict[str, float] | None = None) -> sampleBatch:
        activeBoundaryWeights = dict(boundaryWeights) if boundaryWeights is not None else {name: 1.0 for name in domain.boundaryNames}
        if not activeBoundaryWeights:
            raise ValueError('boundaryWeights must not be empty')
        boundaryNames, weights, sampleCountByBoundary = splitSampleWeightMap(n, activeBoundaryWeights)
        batch = self.sampleBoundaryMixedBatch(domain, sampleCountByBoundary)
        return batch.withWeights(expandSampleWeights(boundaryNames, sampleCountByBoundary, weights))
