from typing import Tuple, Any, Optional
from .operatorsBase import operatorBase

class matrixOperator(operatorBase):
    def __init__(self, data: Any, backendName: str | None = None):
        super().__init__(backendName)
        # TODO: check backend fmt first
        self.data = self.localBackend.array.toArray(data)
        self._isSparse = self._detectSparsity()

    def _detectSparsity(self) -> bool:
        return self.localBackend.array.isSparse(self.data)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> Any:
        return getattr(self.data, 'dtype', None)

    @property
    def isSparse(self) -> bool:
        return self._isSparse

    @property
    def T(self) -> 'matrixOperator':
        backend = self.localBackend
        return matrixOperator(backend.linalg.transpose(self.data), backendName=self.backendName)

    def apply(self, x: Any, trans: bool = False) -> Any:
        backend = self.localBackend
        xData = x.data if isinstance(x, matrixOperator) else backend.array.toArray(x)
        data = backend.linalg.transpose(self.data) if trans else self.data
        return backend.linalg.dot(data, xData)

    def __mul__(self, scalar: float) -> 'matrixOperator':
        return matrixOperator(self.data * scalar, backendName=self.backendName)

    def toNumpy(self) -> Any:
        backend = self.localBackend
        return backend.array.toNumpy(self.data)

    def __repr__(self) -> str:
        return f"matrixOperator(shape={self.shape}, isSparse={self.isSparse}, backend={self.backendName})"
