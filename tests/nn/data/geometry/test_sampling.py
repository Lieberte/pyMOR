import numpy as np
import pytest
from mor.nn.data.geometry.samplers import baseSampler
from mor.nn.data.geometry.samplers import combinedSampler
from mor.nn.data.geometry.samplers import regionSampler
from mor.nn.data.geometry.utils import expandSampleWeights
from mor.nn.data.geometry.utils import normalizeSampleCountMap
from mor.nn.data.geometry.utils import splitSampleWeightMap

class deterministicSampler(baseSampler):
    def sampleInterior(self, domain, n: int) -> np.ndarray:
        return domain.interiorPoints()[:int(n)]

    def sampleBoundary(self, domain, n: int, boundaryName: str | None = None) -> np.ndarray:
        return domain.boundaryPoints(boundaryName=boundaryName)[:int(n)]

class dummyDomain:
    def __init__(self):
        self.boundaryNames = ['left', 'right']
        self._interior = np.array([[0.2, 0.2], [0.4, 0.4], [0.6, 0.6]], dtype=float)
        self._boundary = {
            'left': np.array([[0.0, 0.1], [0.0, 0.9], [0.0, 0.5]], dtype=float),
            'right': np.array([[1.0, 0.2], [1.0, 0.8], [1.0, 0.5]], dtype=float),
        }

    def interiorPoints(self) -> np.ndarray:
        return self._interior

    def boundaryPoints(self, boundaryName: str | None = None) -> np.ndarray:
        if boundaryName is None:
            return np.concatenate([self._boundary[name] for name in self.boundaryNames], axis=0)
        return self._boundary[boundaryName]

    def locateBoundary(self, x: np.ndarray) -> np.ndarray:
        return np.where(x[:, 0] < 0.5, 'left', 'right').astype(object)

    def boundaryNormal(self, x: np.ndarray, boundaryName: str | None = None) -> np.ndarray:
        if boundaryName is not None:
            if boundaryName == 'left':
                return np.tile(np.array([[-1.0, 0.0]]), (x.shape[0], 1))
            return np.tile(np.array([[1.0, 0.0]]), (x.shape[0], 1))
        names = self.locateBoundary(x)
        normals = np.zeros_like(x, dtype=float)
        normals[names == 'left'] = np.array([-1.0, 0.0])
        normals[names == 'right'] = np.array([1.0, 0.0])
        return normals

def testNormalizeSampleCountMap():
    sampleCountByName = normalizeSampleCountMap({'interior': 2, 'left': 0, 'right': -1}, emptyError='empty')
    assert sampleCountByName == {'interior': 2}

def testSplitSampleWeightMap():
    names, weights, sampleCountByName = splitSampleWeightMap(5, {'interior': 2.0, 'left': 1.0})
    assert names == ['interior', 'left']
    assert np.allclose(weights, np.array([2.0 / 3.0, 1.0 / 3.0]))
    assert sum(sampleCountByName.values()) == 5
    assert np.allclose(
        expandSampleWeights(names, sampleCountByName, weights),
        np.array([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
    )

def testBaseSamplerWeightedBatch():
    sampler = deterministicSampler()
    domain = dummyDomain()
    batch = sampler.sampleWeightedBatch(domain, 5, regionWeights={'interior': 2.0, 'left': 1.0})
    assert batch.size == 5
    assert list(batch.regionNames) == ['interior', 'interior', 'interior', 'left', 'left']
    assert np.allclose(batch.weights, np.array([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))

def testRegionAndCombinedSampler():
    domain = dummyDomain()
    batch = regionSampler(regionWeights={'interior': 1.0, 'right': 2.0}).sampleWeightedBatch(domain, 3)
    assert batch.size == 3
    assert list(batch.regionNames) == ['interior', 'right', 'right']
    mixedBatch = combinedSampler(deterministicSampler(), deterministicSampler()).sampleMixedBatch(
        domain,
        {'interior': 1, 'left': 2},
    )
    assert list(mixedBatch.regionNames) == ['interior', 'left', 'left']


@pytest.mark.xfail(
    reason='known bug: sampleBatch.concat drops ALL normals when any constituent batch has '
           'normals=None (e.g. interior points), so mixing interior + boundary loses boundary '
           'normals. Tracked for the geometry-fix branch. See mor/nn/data/geometry/sampleBatch.py:39-41.',
    strict=True,
)
def testMixedBatchBoundaryNormalsPreserved():
    domain = dummyDomain()
    mixedBatch = combinedSampler(deterministicSampler(), deterministicSampler()).sampleMixedBatch(
        domain,
        {'interior': 1, 'left': 2},
    )
    assert np.allclose(mixedBatch.normals[1:], np.array([[-1.0, 0.0], [-1.0, 0.0]]))
