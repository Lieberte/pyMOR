from .baseSampler import baseSampler
from ..sampleBatch import sampleBatch

class combinedSampler(baseSampler):
    def __init__(self, interiorSampler, boundarySampler):
        self.interiorSampler = interiorSampler
        self.boundarySampler = boundarySampler

    def sampleInterior(self, domain, n: int):
        return self.interiorSampler.sampleInterior(domain, n)

    def sampleBoundary(self, domain, n: int, boundaryName: str | None = None):
        return self.boundarySampler.sampleBoundary(domain, n, boundaryName=boundaryName)

    def sampleBoundaryMixedBatch(self, domain, sampleCountByBoundary: dict[str, int]) -> sampleBatch:
        return self.boundarySampler.sampleBoundaryMixedBatch(domain, sampleCountByBoundary)

    def sampleBoundaryWeightedBatch(self, domain, n: int, boundaryWeights: dict[str, float] | None = None) -> sampleBatch:
        return self.boundarySampler.sampleBoundaryWeightedBatch(domain, n, boundaryWeights=boundaryWeights)
