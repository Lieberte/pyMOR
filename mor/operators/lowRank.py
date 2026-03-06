from typing import Tuple, Any
from .operatorsBase import operatorBase

class lowRankOperator(operatorBase):
    def __init__(self, left: Any, right: Any | None = None, backendName: str | None = None):
        super().__init__(backendName)
        self.left = self.localBackend.array.toArray(left)
        self.right = self.localBackend.array.toArray(right) if right is not None else None
        self._validate()

    def _validate(self):
        if self.right is not None:
            if self.left.shape[0] != self.right.shape[0]:
                raise ValueError("Left and right factors must have same number of rows")
        
    @property
    def shape(self) -> Tuple[int, ...]:
        n = self.left.shape[0]
        return (n, n)

    @property
    def dtype(self) -> Any:
        return getattr(self.left, 'dtype', None)

    @property
    def isSparse(self) -> bool:
        return False

    @property
    def T(self) -> 'lowRankOperator':
        if self.right is None: return self
        return lowRankOperator(self.right, self.left, backendName=self.backendName)

    def apply(self, x: Any, trans: bool = False) -> Any:
        backend = self.localBackend
        xData = x.data if isinstance(x, matrixOperator) else backend.array.toArray(x)
        if self.right is None:
            tmp = backend.linalg.dot(backend.linalg.transpose(self.left), xData)
            return backend.linalg.dot(self.left, tmp)
        else:
            l, r = (self.right, self.left) if trans else (self.left, self.right)
            tmp = backend.linalg.dot(backend.linalg.transpose(r), xData)
            return backend.linalg.dot(l, tmp)

    def __mul__(self, scalar: float) -> 'lowRankOperator':
        import numpy as np
        return lowRankOperator(self.left * np.sqrt(scalar), self.right * np.sqrt(scalar) if self.right is not None else None, backendName=self.backendName)

    def toNumpy(self) -> Any:
        backend = self.localBackend
        l_np = backend.array.toNumpy(self.left)
        if self.right is None:
            import numpy as np
            return l_np @ l_np.T
        r_np = backend.array.toNumpy(self.right)
        return l_np @ r_np.T

    def __repr__(self) -> str:
        kind = "Symmetric" if self.right is None else "General"
        return f"lowRankOperator({kind}, shape={self.shape}, rank={self.left.shape[1]}, backend={self.backendName})"
