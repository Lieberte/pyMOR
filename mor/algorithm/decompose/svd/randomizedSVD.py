from typing import Any
from .svd import svd
from mor.algorithm.registry import registerAlgorithm
from ..truncation.truncation import truncate

@registerAlgorithm('svd', 'randomized')
class randomizedSVD(svd):
    def decompose(self, xOperator: Any, rank: int, tol: float | None = None, oversampling: int = 10, iterations: int = 2) -> tuple[Any, Any, Any]:
        backend, S = self.localBackend, xOperator.data
        n, k = S.shape
        l = rank + oversampling
        Omega = backend.array.randn((k, l), dtype=S.dtype)
        Y = backend.linalg.dot(S, Omega)
        for _ in range(iterations):
            Q, _ = backend.linalg.qr(Y, mode='reduced')
            Y = backend.linalg.dot(S, backend.linalg.dot(backend.linalg.transpose(backend.linalg.conj(S)), Q))
        Q, _ = backend.linalg.qr(Y, mode='reduced')
        B = backend.linalg.dot(backend.linalg.transpose(backend.linalg.conj(Q)), S)
        uHat, s, Vh = backend.decomposition.svdDense(B, fullMatrices=False)
        U = backend.linalg.dot(Q, uHat)
        r = truncate(s, rank=rank, tol=tol, backend=backend)
        return U[:, :r], s[:r], Vh[:r, :]
