from typing import Any

from .svd import svd
from mor.algorithm.registry import registerAlgorithm
from ..truncation.truncation import truncate
from mor.operators.operatorsBase import operatorBase

@registerAlgorithm('svd', 'static')
class staticSVD(svd):
    def decompose(self, xOperator: operatorBase, rank: int | None = None, tol: float | None = None, fullMatrices: bool = False) -> tuple[Any, Any, Any]:
        backend = self.localBackend
        data = xOperator.toBackendData()
        self.snapshotMean = None
        if self.options.get('centerSnapshots', False):
            nSnapshots = data.shape[1]
            self.snapshotMean = backend.array.sum(data, axis=1) / nSnapshots
            data = data - self.snapshotMean.reshape((-1, 1))
        U, S, Vt = backend.decomposition.svdDense(data, fullMatrices=fullMatrices)
        r = truncate(S, rank=rank, tol=tol, backend=backend)
        return U[:, :r], S[:r], Vt[:r, :]
