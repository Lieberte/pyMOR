from typing import Any
from .svd import svd
from mor.algorithm.registry import registerAlgorithm
from ..truncation.truncation import truncate

@registerAlgorithm('svd', 'dual')
class dualSVD(svd):
    def decompose(self, xOperator: Any, rank: int | None = None, tol: float | None = None, product: Any | None = None) -> tuple[Any, Any, Any]:
        backend, S = self.localBackend, xOperator.data
        if product is not None:
            K = backend.linalg.dot(backend.linalg.transpose(backend.linalg.conj(S)), product.apply(S))
        else:
            K = backend.linalg.dot(backend.linalg.transpose(backend.linalg.conj(S)), S)
        evals, V = backend.eigen.eigh(K)
        idx = backend.array.argsort(evals)[::-1]
        evals, V = evals[idx], V[:, idx]
        mask = evals > 0
        evals, V = evals[mask], V[:, mask]
        s = backend.array.sqrt(evals)
        r = truncate(s, rank=rank, tol=tol, backend=backend)
        U = backend.linalg.dot(S, V[:, :r]) / s[:r]
        return U, s[:r], backend.linalg.transpose(backend.linalg.conj(V[:, :r]))
