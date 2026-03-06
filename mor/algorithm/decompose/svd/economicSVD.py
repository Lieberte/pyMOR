from typing import Any

from .svd import svd
from mor.algorithm.registry import registerAlgorithm
from ..truncation.truncation import truncate

@registerAlgorithm('svd', 'economic')
class economicSVD(svd):
    def decompose(self, xOperator: Any, rank: int | None = None, tol: float | None = None) -> tuple[Any, Any, Any]:
        backend = self.localBackend
        U, S, Vt = backend.decomposition.svdDense(xOperator.data, fullMatrices=False)
        r = truncate(S, rank=rank, tol=tol, backend=backend)
        return U[:, :r], S[:r], Vt[:r, :]
