import numpy as np
from .baseSampler import baseSampler
from ..utils import sampleRows

class weightedBoundarySampler(baseSampler):
    def __init__(self, boundaryWeights: dict[str, float] | None = None):
        self.boundaryWeights = dict(boundaryWeights) if boundaryWeights is not None else {}

    def sampleInterior(self, domain, n: int) -> np.ndarray:
        return sampleRows(domain.interiorPoints(), n)

    def sampleBoundary(self, domain, n: int, boundaryName: str | None = None) -> np.ndarray:
        return sampleRows(domain.boundaryPoints(boundaryName=boundaryName), n)

    def sampleBoundaryWeightedBatch(self, domain, n: int, boundaryWeights: dict[str, float] | None = None):
        activeBoundaryWeights = dict(self.boundaryWeights)
        if boundaryWeights is not None:
            activeBoundaryWeights.update(boundaryWeights)
        return super().sampleBoundaryWeightedBatch(domain, n, boundaryWeights=activeBoundaryWeights)
