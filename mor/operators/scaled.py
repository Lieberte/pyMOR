from typing import Tuple, Any
from .operatorsBase import operatorBase


class scaledOperator(operatorBase):
    def __init__(self, operator: operatorBase, scalar: float, backendName: str | None = None):
        super().__init__(backendName or operator.backendName)
        self.operator = operator
        self.scalar = scalar

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.operator.shape

    @property
    def dtype(self) -> Any:
        return self.operator.dtype

    @property
    def isSparse(self) -> bool:
        return self.operator.isSparse

    @property
    def T(self) -> 'scaledOperator':
        return scaledOperator(self.operator.T, self.scalar, backendName=self.backendName)

    def apply(self, x: Any, trans: bool = False) -> Any:
        result = self.operator.apply(x, trans=trans)
        return result * self.scalar

    def toBackendData(self) -> Any:
        backend = self.localBackend
        return backend.array.toArray(self.operator.toBackendData()) * self.scalar

    def __mul__(self, scalar: float) -> 'scaledOperator':
        return scaledOperator(self.operator, self.scalar * scalar, backendName=self.backendName)

    def __repr__(self) -> str:
        return f"scaledOperator(scalar={self.scalar}, shape={self.shape}, backend={self.backendName})"
