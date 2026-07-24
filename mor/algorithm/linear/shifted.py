from typing import Any, Optional
from .linear import linear
from mor.algorithm.registry import registerAlgorithm
from mor.operators import matrixOperator

@registerAlgorithm('linear', 'shifted')
class shiftedLinear(linear):
    def solve(self, A: matrixOperator, rhs: Any, E: Optional[matrixOperator] = None, shift: complex = 0.0, trans: bool = False, **kwargs) -> Any:
        backend, n = self.localBackend, A.shape[0]
        aData = A.toBackendData()
        isSparse = backend.array.isSparse(aData)
        if isSparse:
            EData = E.toBackendData() if E is not None else backend.array.eyeSparse(n, dtype=A.dtype)
        else:
            EData = E.toBackendData() if E is not None else backend.array.eye(n, dtype=A.dtype)
        M = aData + shift * EData
        if trans: M = backend.linalg.transpose(M)
        result = backend.linalg.solve(M, rhs)
        if backend.array.ndim(rhs) == 2 and backend.array.ndim(result) == 1:
            result = backend.array.reshape(result, (-1, 1))
        return result
