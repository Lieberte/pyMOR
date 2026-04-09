import numpy as np
from .baseSampler import baseSampler
from ..utils import sampleRows

class randomSampler(baseSampler):
    def sampleInterior(self, domain, n: int) -> np.ndarray:
        return sampleRows(domain.interiorPoints(), n)

    def sampleBoundary(self, domain, n: int, boundaryName: str | None = None) -> np.ndarray:
        return sampleRows(domain.boundaryPoints(boundaryName=boundaryName), n)
