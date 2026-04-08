from typing import Any
from .svd import svd
from mor.algorithm.registry import registerAlgorithm
from ..truncation.truncation import truncate

from mor.operators.operatorsBase import operatorBase

@registerAlgorithm('svd', 'randomized')
class randomizedSVD(svd):
    def decompose(self, xOperator: operatorBase, rank: int, tol: float | None = None, oversampling: int = 10, iterations: int = 2) -> tuple[Any, Any, Any]:
        backend, S = self.localBackend, xOperator.toBackendData()
        l = rank + oversampling
        U, s, Vh = backend.decomposition.randomizedSvd(S, nComponents=l, iterations=iterations)
        r = truncate(s, rank=rank, tol=tol, backend=backend)
        return U[:, :r], s[:r], Vh[:r, :]
