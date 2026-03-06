from typing import Any
from .svd import svd
from mor.algorithm.registry import registerAlgorithm
from ..truncation.truncation import truncate

@registerAlgorithm('svd', 'qrSVD')
class qrSVD(svd):
    def decompose(self, xOperator: Any, rank: int | None = None, tol: float | None = None, fullMatrices: bool = False) -> tuple[Any, Any, Any]:
        backend, S = self.localBackend, xOperator.data
        Q, R = backend.linalg.qr(S, mode='reduced')
        uHat, s, Vh = backend.decomposition.svdDense(R, fullMatrices=fullMatrices)
        U = backend.linalg.dot(Q, uHat)
        r = truncate(s, rank=rank, tol=tol, backend=backend)
        return U[:, :r], s[:r], Vh[:r, :]
