from typing import Any, Optional
from .linear import linear
from mor.algorithm.registry import registerAlgorithm
from mor.operators import matrixOperator

@registerAlgorithm('linear', 'shifted')
class shiftedLinear(linear):
    def factorize(self, A, E=None, shift=0.0, trans=False):
        backend = self.localBackend
        aData = A.toBackendData()
        eData = backend.array.eyeLike(aData) if E is None else E.toBackendData()
        data = aData + shift * eData
        if trans:
            data = backend.linalg.transpose(data)
        return backend.linalg.factorize(data)

    def solve(self, A: matrixOperator, rhs: Any, E: Optional[matrixOperator] = None, shift: complex = 0.0, trans: bool = False, **kwargs) -> Any:
        backend = self.localBackend
        aData = A.toBackendData()
        EData = E.toBackendData() if E is not None else backend.array.eyeLike(aData)
        M = aData + shift * EData
        if trans: M = backend.linalg.transpose(M)
        result = backend.linalg.solve(M, rhs)
        if backend.array.ndim(rhs) == 2 and backend.array.ndim(result) == 1:
            result = backend.array.reshape(result, (-1, 1))
        return result
