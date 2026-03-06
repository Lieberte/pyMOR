from typing import Any
from .svd import svd
from mor.algorithm.registry import registerAlgorithm

@registerAlgorithm('svd', 'static')
class staticSVD(svd):
    def decompose(self, xOperator: Any, rank: int | None = None, fullMatrices: bool = False) -> tuple[Any, Any, Any]:
        backend = self.localBackend
        U, S, Vt = backend.decomposition.svdDense(xOperator.data, fullMatrices=fullMatrices)
        if rank is not None:
            U = U[:, :rank]
            S = S[:rank]
            Vt = Vt[:rank, :]
        return U, S, Vt
