from typing import Tuple, Any, Optional
from .operatorsBase import operatorBase

class matrixOperator(operatorBase):
    def __init__(self, data: Any, backendName: str | None = None):
        super().__init__(backendName)
        self.data = data
        self._isSparse = self._detectSparsity()

    def _detectSparsity(self) -> bool:
        # TODO: implement more robust sparsity detection across backends
        import scipy.sparse as sp
        return sp.issparse(self.data)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> Any:
        return self.data.dtype

    @property
    def isSparse(self) -> bool:
        return self._isSparse

    @property
    def T(self) -> 'matrixOperator':
        backend = self.localBackend
        return matrixOperator(backend.linalg.transpose(self.data), backendName=self.backendName)

    def apply(self, x: Any, trans: bool = False) -> Any:
        backend = self.localBackend
        data = self.data.T if trans else self.data
        return backend.linalg.dot(data, x)

    def toNumpy(self) -> Any:
        backend = self.localBackend
        return backend.array.toNumpy(self.data)

    def __repr__(self) -> str:
        return f"matrixOperator(shape={self.shape}, isSparse={self.isSparse}, backend={self.backendName})"
