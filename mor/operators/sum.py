from typing import Tuple, Any, List
from .operatorsBase import operatorBase

class sumOperator(operatorBase):
    def __init__(self, operators: List[operatorBase], backendName: str | None = None):
        if not operators: raise ValueError("Operators list cannot be empty")
        if any(not isinstance(op, operatorBase) for op in operators):
            raise TypeError("All entries must be operatorBase instances")
        super().__init__(backendName or operators[0].backendName)
        self.operators = operators
        self._validate()

    def _validate(self):
        shape = self.operators[0].shape
        backendName = self.operators[0].backendName
        for op in self.operators[1:]:
            if op.shape != shape:
                raise ValueError(f"Operator shape {op.shape} does not match {shape}")
            if op.backendName != backendName:
                raise ValueError(f"Operator backend {op.backendName} does not match {backendName}")

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
    def structureTags(self) -> frozenset[str]:
        tags = {'composed'}
        tags.add('sparse' if self.isSparse else 'dense')
        if all(op.isDiagonal for op in self.operators):
            tags.add('diagonal')
        if any(op.isLowRank for op in self.operators):
            tags.add('contains_low_rank')
        return frozenset(tags)

    @property
    def materializationCost(self) -> int | None:
        costs = [op.materializationCost for op in self.operators]
        if any(cost is None for cost in costs):
            return None
        return int(sum(costs))

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

    def toBackendData(self) -> Any:
        backend = self.localBackend
        data = backend.array.toArray(self.operators[0].toBackendData())
        for op in self.operators[1:]:
            data = data + backend.array.toArray(op.toBackendData())
        return data

    def __mul__(self, scalar: float) -> 'sumOperator':
        return sumOperator([op * scalar for op in self.operators], backendName=self.backendName)

    def __repr__(self) -> str:
        return f"sumOperator(count={len(self.operators)}, shape={self.shape}, backend={self.backendName})"
