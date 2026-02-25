from typing import Any, Optional
from .linear import linearAlgorithm
from mor.algorithm.registry import registerAlgorithm
from mor.operators import matrixOperator

@registerAlgorithm('linear', 'shifted')
class shiftedLinearAlgorithm(linearAlgorithm):
    def solve(self, A: matrixOperator, rhs: Any, E: Optional[matrixOperator] = None, shift: complex = 0.0, trans: bool = False, **kwargs) -> Any:
        backend = self.localBackend
        n = A.shape[0]
        eData = E.data if E is not None else backend.array.eye(n, dtype=A.dtype)
        m = A.data + shift * eData
        if trans:
            m = backend.linalg.transpose(m)
        return backend.linalg.solve(m, rhs)
