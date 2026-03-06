from typing import Any, Tuple
from .svd import svd
from mor.algorithm.registry import registerAlgorithm
from ..truncation.truncation import truncate

@registerAlgorithm('svd', 'incrementalBrand')
class incrementalBrandSVD(svd):
    def __init__(self, backendName: str | None = None, **kwargs):
        super().__init__(backendName, **kwargs)
        self.U = None
        self.S = None
        self.Vh = None

    def decompose(self, xOperator: Any, rank: int | None = None, tol: float | None = None) -> Tuple[Any, Any, Any]:
        backend, newData = self.localBackend, xOperator.data
        if self.U is None:
            self.U, self.S, self.Vh = backend.decomposition.svdDense(newData, fullMatrices=False)
        else:
            Ut = backend.linalg.transpose(backend.linalg.conj(self.U))
            m = backend.linalg.dot(Ut, newData)
            p = newData - backend.linalg.dot(self.U, m)
            p, ra = backend.linalg.qr(p, mode='reduced')
            nS = backend.array.size(self.S)
            kTop = backend.array.hstack([backend.array.diag(self.S), m])
            kBottom = backend.array.hstack([backend.array.zeros((backend.array.shape(ra)[0], nS), dtype=self.S.dtype), ra])
            K = backend.array.vstack([kTop, kBottom])
            uHat, sHat, vhHat = backend.decomposition.svdDense(K, fullMatrices=False)
            self.U = backend.linalg.dot(backend.array.hstack([self.U, p]), uHat)
            self.S = sHat
            self.Vh = vhHat
        r = truncate(self.S, rank=rank, tol=tol, backend=backend)
        self.U, self.S, self.Vh = self.U[:, :r], self.S[:r], self.Vh[:r, :]
        return self.U, self.S, self.Vh
