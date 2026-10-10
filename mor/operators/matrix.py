from typing import Tuple, Any, Optional
from numbers import Number
from .operatorsBase import operatorBase

class matrixOperator(operatorBase):
    def __init__(self, data: Any, backendName: str | None = None):
        super().__init__(backendName)
        self.data = self.localBackend.array.toArray(data)
        self._isSparse = self._detectSparsity()
        self._isDiagonal = None

    @property
    def isDiagonal(self) -> bool:
        if self._isDiagonal is None:
            self._isDiagonal = self._detectDiagonal()
        return self._isDiagonal

    def _detectSparsity(self) -> bool:
        return self.localBackend.array.isSparse(self.data)

    def _detectDiagonal(self) -> bool:
        return self.localBackend.array.isDiagonal(self.data)

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
    def structureTags(self) -> frozenset[str]:
        tags = {'sparse'} if self.isSparse else {'dense'}
        tags.add('matrix')
        if self.isDiagonal:
            tags.add('diagonal')
        return frozenset(tags)

    @property
    def T(self) -> 'matrixOperator':
        backend = self.localBackend
        return matrixOperator(backend.linalg.transpose(self.data), backendName=self.backendName)

    def apply(self, x: Any, trans: bool = False) -> Any:
        backend = self.localBackend
        xData = self._inputData(x, like=self.data)
        data = backend.linalg.transpose(self.data) if trans else self.data
        return backend.linalg.dot(data, xData)

    def toBackendData(self) -> Any:
        return self.data

    def diagonal(self) -> Any:
        if not self.isDiagonal:
            raise ValueError("Operator is not diagonal")
        return self.localBackend.array.diagonal(self.data)

    def __mul__(self, scalar: float) -> 'matrixOperator':
        if not isinstance(scalar, Number):
            return NotImplemented
        return matrixOperator(self.data * scalar, backendName=self.backendName)

    def toNumpy(self) -> Any:
        backend = self.localBackend
        return backend.array.toNumpy(self.data)

    def __repr__(self) -> str:
        return f"matrixOperator(shape={self.shape}, isSparse={self.isSparse}, backend={self.backendName})"
