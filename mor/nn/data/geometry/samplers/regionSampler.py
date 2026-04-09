import numpy as np
from .baseSampler import baseSampler
from ..utils import normalizeSampleWeights
from ..utils import sampleRows
from ..utils import splitSampleCounts

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
        if not activeRegionWeights:
            activeRegionWeights = {'interior': 1.0, **{name: 1.0 for name in domain.boundaryNames}}
        regionNames = list(activeRegionWeights.keys())
        weights = normalizeSampleWeights(np.asarray([activeRegionWeights[name] for name in regionNames], dtype=float))
        counts = splitSampleCounts(n, weights)
        sampleCountByRegion = {name: int(count) for name, count in zip(regionNames, counts)}
        batch = self.sampleMixedBatch(domain, sampleCountByRegion)
        sampleWeights = np.concatenate(
            [np.full(sampleCountByRegion[name], weights[i], dtype=float) for i, name in enumerate(regionNames) if sampleCountByRegion[name] > 0],
            axis=0,
        )
        batch.weights = sampleWeights
        return batch
