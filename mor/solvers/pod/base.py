from typing import Any
from mor.algorithm.registry import algorithmRegistry
from mor.backends import backendRegistry
from mor.operators.operatorsBase import operatorBase
from mor.operators.matrix import matrixOperator

class podSolverBase:
    def __init__(self, backendName: str | None = None, **kwargs):
        self.backendName = backendName
        self.options = kwargs
        self.localBackend = backendRegistry.get(backendName)
        self._lastAlgorithmName = 'unknown'
        self.snapshotMean = None
        self.product = None

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
        options = dict(self.options)
        self.snapshotMean = None
        if options.pop('centerSnapshots', False):
            data = snapshotsPrepared.toBackendData()
            self.snapshotMean = self.localBackend.array.sum(data, axis=1) / data.shape[1]
            centered = data - self.localBackend.array.reshape(self.snapshotMean, (-1, 1))
            snapshotsPrepared = matrixOperator(centered, backendName=snapshotsPrepared.backendName)
        backendName = snapshotsPrepared.backendName or self.localBackend.name
        product = options.get('product')
        svdAlgorithm = algorithmRegistry.get(
            category='svd',
            variant=self.selectSvdVariant(method),
            forceOptions=options,
            backendName=backendName,
            xOperator=snapshotsPrepared,
            rank=rank,
            product=product,
        )
        if product is not None:
            from mor.algorithm.decompose.svd.dualSVD import dualSVD
            if not isinstance(svdAlgorithm, dualSVD):
                raise ValueError("Weighted POD requires the dual SVD variant.")
            U, S, Vt = svdAlgorithm.decompose(snapshotsPrepared, rank=rank, tol=tol, product=product)
        else:
            U, S, Vt = svdAlgorithm.decompose(snapshotsPrepared, rank=rank, tol=tol)
        self._lastAlgorithmName = svdAlgorithm.__class__.__name__
        self.product = product
        return self.postprocessDecomposition(U, S, Vt)
