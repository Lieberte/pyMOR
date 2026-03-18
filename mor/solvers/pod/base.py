from typing import Any
from mor.algorithm.registry import algorithmRegistry
from mor.backends import backendRegistry
from mor.operators.operatorsBase import operatorBase

class podSolverBase:
    def __init__(self, backendName: str | None = None, **kwargs):
        self.backendName = backendName
        self.options = kwargs
        self.localBackend = backendRegistry.get(backendName)
        self._lastAlgorithmName = 'unknown'

    def preprocessSnapshots(self, snapshots: operatorBase) -> operatorBase:
        return snapshots

    def postprocessDecomposition(self, U: Any, S: Any, Vt: Any) -> tuple[Any, Any, Any]:
        return U, S, Vt

    def selectSvdVariant(self, method: str) -> str:
        if method != 'auto':
            return method
        return self.options.get('svdVariant', self.options.get('variant', 'auto'))

    def solve(self, snapshots: operatorBase, rank: int | None = None, tol: float | None = None, method: str = 'auto') -> tuple[Any, Any, Any]:
        snapshotsPrepared = self.preprocessSnapshots(snapshots)
        backendName = snapshotsPrepared.backendName or self.localBackend.name
        svdAlgorithm = algorithmRegistry.get(category='svd', variant=self.selectSvdVariant(method), forceOptions=self.options, backendName=backendName, xOperator=snapshotsPrepared)
        U, S, Vt = svdAlgorithm.decompose(snapshotsPrepared, rank=rank, tol=tol)
        self._lastAlgorithmName = svdAlgorithm.__class__.__name__
        return self.postprocessDecomposition(U, S, Vt)
