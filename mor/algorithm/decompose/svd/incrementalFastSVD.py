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
        backend, newData = self.localBackend, xOperator.toBackendData()
        if self.U is None:
            self.U, self.S, self.Vh = backend.decomposition.svdDense(newData, fullMatrices=False)
        else:
            currentData = backend.linalg.dot(self.U, backend.linalg.dot(backend.array.diag(self.S), self.Vh))
            combined = backend.array.hstack([currentData, newData])
            self.U, self.S, self.Vh = backend.decomposition.svdDense(combined, fullMatrices=False)
        r = truncate(self.S, rank=rank, tol=tol, backend=backend)
        self.U, self.S, self.Vh = self.U[:, :r], self.S[:r], self.Vh[:r, :]
        return self.U, self.S, self.Vh
