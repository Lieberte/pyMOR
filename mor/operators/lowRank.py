from typing import Tuple, Any
from numbers import Number, Real
from .operatorsBase import operatorBase

class lowRankOperator(operatorBase):
    def __init__(self, left: Any, right: Any | None = None, backendName: str | None = None):
        super().__init__(backendName)
        self.left = self.localBackend.array.toArray(left)
        self.right = self.localBackend.array.toArray(right) if right is not None else None
        self._validate()
        if self.right is not None:
            dtype = self._resultDtype(self.left.dtype, self.right.dtype)
            self.left = self.localBackend.array.astype(self.left, dtype)
            self.right = self.localBackend.array.astype(self.right, dtype)

    def _validate(self):
        if len(self.left.shape) != 2:
            raise ValueError("Left factor must be a 2D array")
        if self.right is not None:
            if len(self.right.shape) != 2:
                raise ValueError("Right factor must be a 2D array")
            if self.left.shape[1] != self.right.shape[1]:
                raise ValueError("Left and right factors must have same number of columns")
        
    @property
    def shape(self) -> Tuple[int, ...]:
        if self.right is None:
            n = self.left.shape[0]
            return (n, n)
        return (self.left.shape[0], self.right.shape[0])

    @property
    def dtype(self) -> Any:
        leftDtype = getattr(self.left, 'dtype', None)
        if self.right is None:
            return leftDtype
        return self._resultDtype(leftDtype, getattr(self.right, 'dtype', None))

    @property
    def isSparse(self) -> bool:
        return False

    @property
    def structureTags(self) -> frozenset[str]:
        return frozenset({'dense', 'low_rank', 'symmetric_low_rank'} if self.right is None else {'dense', 'low_rank'})

    @property
    def materializationCost(self) -> int:
        if self.right is None:
            return int(self.left.shape[0] * self.left.shape[1])
        return int((self.left.shape[0] + self.right.shape[0]) * self.left.shape[1])

    @property
    def T(self) -> 'lowRankOperator':
        backend = self.localBackend
        if self.right is None:
            return lowRankOperator(backend.linalg.conj(self.left), backendName=self.backendName)
        return lowRankOperator(
            backend.linalg.conj(self.right),
            backend.linalg.conj(self.left),
            backendName=self.backendName,
        )

    def apply(self, x: Any, trans: bool = False) -> Any:
        backend = self.localBackend
        xData = self._inputData(x, like=self.left)
        right = self.left if self.right is None else self.right
        if trans:
            left, right = backend.linalg.conj(right), backend.linalg.conj(self.left)
        else:
            left = self.left
        rightAdjoint = backend.linalg.transpose(backend.linalg.conj(right))
        return backend.linalg.dot(left, backend.linalg.dot(rightAdjoint, xData))

    def __mul__(self, scalar: float) -> 'lowRankOperator':
        if not isinstance(scalar, Number):
            return NotImplemented
        if self.right is not None:
            return lowRankOperator(self.left * scalar, self.right, backendName=self.backendName)
        if isinstance(scalar, Real) and scalar >= 0:
            factor = self.localBackend.array.sqrt(scalar)
            return lowRankOperator(self.left * factor, backendName=self.backendName)
        return lowRankOperator(self.left * scalar, self.left, backendName=self.backendName)

    def toNumpy(self) -> Any:
        backend = self.localBackend
        l_np = backend.array.toNumpy(self.left)
        if self.right is None:
            return l_np @ l_np.conj().T
        r_np = backend.array.toNumpy(self.right)
        return l_np @ r_np.conj().T

    def toBackendData(self) -> Any:
        backend = self.localBackend
        leftAdjoint = backend.linalg.transpose(backend.linalg.conj(self.left))
        if self.right is None:
            return backend.linalg.dot(self.left, leftAdjoint)
        rightAdjoint = backend.linalg.transpose(backend.linalg.conj(self.right))
        return backend.linalg.dot(self.left, rightAdjoint)

    def __repr__(self) -> str:
        kind = "Symmetric" if self.right is None else "General"
        return f"lowRankOperator({kind}, shape={self.shape}, rank={self.left.shape[1]}, backend={self.backendName})"
