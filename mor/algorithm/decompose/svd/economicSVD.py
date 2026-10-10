from typing import Any

from .svd import svd
from mor.algorithm.registry import registerAlgorithm
from ..truncation.truncation import truncate

@registerAlgorithm('svd', 'economic')
class economicSVD(svd):
    def decompose(self, xOperator: Any, rank: int | None = None, tol: float | None = None) -> tuple[Any, Any, Any]:
        backend = self.localBackend
        data = xOperator.toBackendData()
        truncate(backend.array.zeros((0,)), rank=rank, tol=tol, backend=backend)
        if rank == 0:
            return (backend.array.zeros((data.shape[0], 0), dtype=data.dtype),
                    backend.array.zeros((0,)),
                    backend.array.zeros((0, data.shape[1]), dtype=data.dtype))
        if backend.array.isSparse(data) and rank is not None and rank < min(data.shape):
            U, S, Vt = backend.decomposition.svdSparse(data, k=int(rank))
        else:
            U, S, Vt = backend.decomposition.svdDense(data, fullMatrices=False)
        r = truncate(S, rank=rank, tol=tol, backend=backend)
        return U[:, :r], S[:r], Vt[:r, :]
