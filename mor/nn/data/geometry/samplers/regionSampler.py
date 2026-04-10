import numpy as np
from .baseSampler import baseSampler
from ..utils import sampleRows

class regionSampler(baseSampler):
    def __init__(self, regionWeights: dict[str, float] | None = None):
        self.regionWeights = dict(regionWeights) if regionWeights is not None else {}

    def sampleInterior(self, domain, n: int) -> np.ndarray:
        return sampleRows(domain.interiorPoints(), n)

    def sampleBoundary(self, domain, n: int, boundaryName: str | None = None) -> np.ndarray:
        return sampleRows(domain.boundaryPoints(boundaryName=boundaryName), n)

    def sampleWeightedBatch(self, domain, n: int, regionWeights: dict[str, float] | None = None):
        activeRegionWeights = dict(self.regionWeights)
        if regionWeights is not None:
            activeRegionWeights.update(regionWeights)
        return super().sampleWeightedBatch(domain, n, regionWeights=activeRegionWeights)
