from typing import Any, Optional
from .linear import linear
from mor.algorithm.registry import registerAlgorithm
from mor.operators import matrixOperator

@registerAlgorithm('linear', 'shifted')
class shiftedLinear(linear):
    def solve(self, A: matrixOperator, rhs: Any, E: Optional[matrixOperator] = None, shift: complex = 0.0, trans: bool = False, **kwargs) -> Any:
        backend, n = self.localBackend, A.shape[0]
        isSparse = backend.array.isSparse(A.data)
        if isSparse: EData = E.data if E is not None else backend.array.eyeSparse(n, dtype=A.dtype)
        else: EData = E.data if E is not None else backend.array.eye(n, dtype=A.dtype)
        M = A.data + shift * EData
        if trans: M = backend.linalg.transpose(M)
        return backend.linalg.solve(M, rhs)
