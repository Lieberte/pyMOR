from typing import Any, Tuple
from .svd import svd
from mor.algorithm.registry import registerAlgorithm
from ..truncation.truncation import truncate

@registerAlgorithm('svd', 'incrementalFast')
class incrementalFastSVD(svd):
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
            combinedS = backend.array.hstack([backend.array.diag(self.S), m])
            _, sHat, vhHat = backend.decomposition.svdDense(combinedS, fullMatrices=False)
            self.S = sHat
            self.Vh = vhHat
        r = truncate(self.S, rank=rank, tol=tol, backend=backend)
        self.S, self.Vh = self.S[:r], self.Vh[:r, :]
        return self.U, self.S, self.Vh
