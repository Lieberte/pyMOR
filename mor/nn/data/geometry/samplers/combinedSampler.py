from .baseSampler import baseSampler
from ..sampleBatch import sampleBatch
from ..utils import normalizeSampleCountMap

class combinedSampler(baseSampler):
    def __init__(self, interiorSampler, boundarySampler):
        self.interiorSampler = interiorSampler
        self.boundarySampler = boundarySampler

    def sampleInterior(self, domain, n: int):
        return self.interiorSampler.sampleInterior(domain, n)

    def sampleBoundary(self, domain, n: int, boundaryName: str | None = None):
        return self.boundarySampler.sampleBoundary(domain, n, boundaryName=boundaryName)

    def sampleRegion(self, domain, n: int, regionName: str):
        if regionName == 'interior':
            return self.sampleInterior(domain, n)
        return self.sampleBoundary(domain, n, boundaryName=regionName)

    def sampleInteriorBatch(self, domain, n: int, regionName: str = 'interior'):
        return self.interiorSampler.sampleInteriorBatch(domain, n, regionName=regionName)

    def sampleBoundaryBatch(self, domain, n: int, boundaryName: str | None = None):
        return self.boundarySampler.sampleBoundaryBatch(domain, n, boundaryName=boundaryName)

    def sampleRegionBatch(self, domain, n: int, regionName: str):
        if regionName == 'interior':
            return self.sampleInteriorBatch(domain, n, regionName=regionName)
        return self.sampleBoundaryBatch(domain, n, boundaryName=regionName)

    def sampleMixedBatch(self, domain, sampleCountByRegion: dict[str, int]):
        sampleCountByRegion = normalizeSampleCountMap(
            sampleCountByRegion,
            emptyError='sampleCountByRegion must contain a positive sample count',
        )
        batches = [self.sampleRegionBatch(domain, n, regionName=regionName) for regionName, n in sampleCountByRegion.items()]
        return sampleBatch.concat(batches)

    def sampleBoundaryMixedBatch(self, domain, sampleCountByBoundary: dict[str, int]):
        return self.boundarySampler.sampleBoundaryMixedBatch(domain, sampleCountByBoundary)

    def sampleBoundaryWeightedBatch(self, domain, n: int, boundaryWeights: dict[str, float] | None = None):
        return self.boundarySampler.sampleBoundaryWeightedBatch(domain, n, boundaryWeights=boundaryWeights)
