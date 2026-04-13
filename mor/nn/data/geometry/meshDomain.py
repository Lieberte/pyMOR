import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from .baseDomain import baseDomain
from .boundaryLocateStrategies import nearestBoundaryLocateStrategy
from .boundaryNormalStrategies import localPcaBoundaryNormalStrategy
from .containStrategies import convexHullContainStrategy
from .geometryRegion import geometryRegion
from .meshIr import meshIr
from .samplers import randomSampler
from .utils import as2dFloatArray
from .utils import computeBoundingBox
from .utils import computeCharacteristicLength
from .utils import fitUnitCubeTransform
from .utils import mergeBoundaryMap
from .utils import normalizeBoundaryMap
from .utils import validateNodes

class meshDomain(baseDomain):
    def __init__(
        self,
        nodes: np.ndarray,
        boundaryNodes: dict[str, np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]] | None = None,
        *,
        containStrategy=convexHullContainStrategy(),
        boundaryLocateStrategy=nearestBoundaryLocateStrategy(),
        boundaryNormalStrategy=localPcaBoundaryNormalStrategy(),
        boundaryTol: float | None = None,
        hullTol: float | None = None,
        sampler=randomSampler(),
    ):
        normalizedNodes = validateNodes(nodes)
        super().__init__(dim=normalizedNodes.shape[1])
        self.nodes = normalizedNodes
        self.boundaryNodes = normalizeBoundaryMap(boundaryNodes, dim=self.dim)
        self.containStrategy = containStrategy
        self.boundaryLocateStrategy = boundaryLocateStrategy
        self.boundaryNormalStrategy = boundaryNormalStrategy
        char = computeCharacteristicLength(self.nodes)
        self.boundaryTol = float(boundaryTol) if boundaryTol is not None else max(char * 1e-6, 1e-12)
        self.hullTol = float(hullTol) if hullTol is not None else max(char * 1e-9, 1e-12)
        self.sampler = sampler
        self._hull: ConvexHull | None = None
        self._allBoundaryNodes = mergeBoundaryMap(self.boundaryNodes, dim=self.dim)
        self._boundaryTrees: dict[str, cKDTree] = {}
        self._normShift: np.ndarray | None = None
        self._normScale: np.ndarray | None = None

    @classmethod
    def fromMeshIr(cls, ir: meshIr, **kwargs):
        return cls(ir.nodes, ir.boundaryNodes, **kwargs)

    @property
    def boundaryNames(self) -> list[str]:
        return list(self.boundaryNodes.keys())

    @property
    def regions(self) -> list[geometryRegion]:
        regions = [geometryRegion(name=self.interiorRegionName, kind='interior', dim=self.dim)]
        regions.extend(geometryRegion(name=name, kind='boundary', dim=max(self.dim - 1, 0)) for name in self.boundaryNames)
        return regions

    def interiorPoints(self) -> np.ndarray:
        return self.nodes

    def boundaryPoints(self, boundaryName: str | None = None) -> np.ndarray:
        if boundaryName is not None:
            if boundaryName not in self.boundaryNodes:
                raise KeyError(boundaryName)
            return self.boundaryNodes[boundaryName]
        return self._mergedAllBoundaryNodes()

    def boundingBox(self) -> tuple[np.ndarray, np.ndarray]:
        return computeBoundingBox(self.nodes)

    def _getHull(self) -> ConvexHull:
        if self._hull is None:
            if self.nodes.shape[0] < self.dim + 1:
                raise ValueError('need at least dim+1 nodes for convexHull')
            self._hull = ConvexHull(self.nodes)
        return self._hull

    def convexHullMeasure(self) -> float:
        return float(self._getHull().volume)

    def contains(self, x: np.ndarray) -> np.ndarray:
        x = as2dFloatArray(x, dim=self.dim)
        return self.containStrategy.contains(self, x)

    def _mergedAllBoundaryNodes(self) -> np.ndarray:
        return self._allBoundaryNodes

    def _getBoundaryTree(self, boundaryName: str) -> cKDTree:
        if boundaryName not in self.boundaryNodes:
            raise KeyError(boundaryName)
        if boundaryName not in self._boundaryTrees:
            nodes = self.boundaryNodes[boundaryName]
            if nodes.shape[0] == 0:
                raise ValueError(f'boundary {boundaryName} is empty')
            self._boundaryTrees[boundaryName] = cKDTree(nodes)
        return self._boundaryTrees[boundaryName]

    def locateBoundary(self, x: np.ndarray) -> np.ndarray:
        x = as2dFloatArray(x, dim=self.dim)
        return self.boundaryLocateStrategy.locateBoundary(self, x)

    def boundaryNormal(self, x: np.ndarray, boundaryName: str | None = None) -> np.ndarray:
        x = as2dFloatArray(x, dim=self.dim)
        return self.boundaryNormalStrategy.boundaryNormal(self, x, boundaryName=boundaryName)

    def sampleInterior(self, n: int, sampler=None) -> np.ndarray:
        if sampler is None:
            sampler = self.sampler
        return sampler.sampleInterior(self, n)

    def sampleBoundary(self, n: int, boundaryName: str | None = None, sampler=None) -> np.ndarray:
        if sampler is None:
            sampler = self.sampler
        return sampler.sampleBoundary(self, n, boundaryName=boundaryName)

    def sampleMixedBatch(self, sampleCountByRegion: dict[str, int], sampler=None):
        if sampler is None:
            sampler = self.sampler
        return sampler.sampleMixedBatch(self, sampleCountByRegion)

    def sampleBoundaryMixedBatch(self, sampleCountByBoundary: dict[str, int], sampler=None):
        if sampler is None:
            sampler = self.sampler
        return sampler.sampleBoundaryMixedBatch(self, sampleCountByBoundary)

    def sampleWeightedBatch(self, n: int, regionWeights: dict[str, float] | None = None, sampler=None):
        if sampler is None:
            sampler = self.sampler
        if not hasattr(sampler, 'sampleWeightedBatch'):
            raise AttributeError('sampler does not support sampleWeightedBatch')
        return sampler.sampleWeightedBatch(self, n, regionWeights=regionWeights)

    def sampleBoundaryWeightedBatch(self, n: int, boundaryWeights: dict[str, float] | None = None, sampler=None):
        if sampler is None:
            sampler = self.sampler
        if not hasattr(sampler, 'sampleBoundaryWeightedBatch'):
            raise AttributeError('sampler does not support sampleBoundaryWeightedBatch')
        return sampler.sampleBoundaryWeightedBatch(self, n, boundaryWeights=boundaryWeights)

    def fitUnitCube(self) -> None:
        self._normShift, self._normScale = fitUnitCubeTransform(self.nodes)

    def refactorMesh(self, mesh: np.ndarray) -> np.ndarray:
        mesh = as2dFloatArray(mesh, dim=self.dim)
        if self._normShift is None:
            self.fitUnitCube()
        shift = self._normShift
        scale = self._normScale
        return (mesh - shift) / scale

    def refactorBoundary(self, boundary: np.ndarray) -> np.ndarray:
        return self.refactorMesh(boundary)
