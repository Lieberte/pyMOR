from typing import Tuple, Any, List
from .operatorsBase import operatorBase

class sumOperator(operatorBase):
    def __init__(self, operators: List[operatorBase], backendName: str | None = None):
        if not operators: raise ValueError("Operators list cannot be empty")
        super().__init__(backendName or operators[0].backendName)
        self.operators = operators
        self._validate()

    def _validate(self):
        shape = self.operators[0].shape
        for op in self.operators[1:]:
            if op.shape != shape:
                raise ValueError(f"Operator shape {op.shape} does not match {shape}")

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.operators[0].shape

    @property
    def dtype(self) -> Any:
        return self.operators[0].dtype

    @property
    def isSparse(self) -> bool:
        return all(op.isSparse for op in self.operators)

    @property
    def T(self) -> 'sumOperator':
        return sumOperator([op.T for op in self.operators], backendName=self.backendName)

    def apply(self, x: Any, trans: bool = False) -> Any:
        res = self.operators[0].apply(x, trans=trans)
        for op in self.operators[1:]:
            res = res + op.apply(x, trans=trans)
        return res

    def toNumpy(self) -> Any:
        res = self.operators[0].toNumpy()
        for op in self.operators[1:]:
            res = res + op.toNumpy()
        return res

    def __repr__(self) -> str:
        return f"sumOperator(count={len(self.operators)}, shape={self.shape}, backend={self.backendName})"
