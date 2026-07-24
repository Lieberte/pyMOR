from typing import Tuple, Any, Optional
from .operatorsBase import operatorBase

class matrixOperator(operatorBase):
    def __init__(self, data: Any, backendName: str | None = None):
        super().__init__(backendName)
        # TODO: check backend fmt first
        self.data = self.localBackend.array.toArray(data)
        self._isSparse = self._detectSparsity()
        self._isDiagonal = self._detectDiagonal()

    def _detectSparsity(self) -> bool:
        return self.localBackend.array.isSparse(self.data)

    def _detectDiagonal(self) -> bool:
        shape = getattr(self.data, 'shape', ())
        if len(shape) != 2 or shape[0] != shape[1]:
            return False
        if type(self.data).__module__.startswith('torch'):
            return False
        try:
            from scipy.sparse import issparse
            if issparse(self.data):
                off_diagonal = self.data.copy()
                off_diagonal.setdiag(0)
                off_diagonal.eliminate_zeros()
                return off_diagonal.nnz == 0
        except Exception:
            pass
        try:
            import numpy as np
            array = self.localBackend.array.toNumpy(self.data)
            return bool(np.allclose(array, np.diag(np.diag(array))))
        except Exception:
            return False

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
        if self._isDiagonal:
            tags.add('diagonal')
        return frozenset(tags)

    @property
    def T(self) -> 'matrixOperator':
        backend = self.localBackend
        return matrixOperator(backend.linalg.transpose(self.data), backendName=self.backendName)

    def apply(self, x: Any, trans: bool = False) -> Any:
        backend = self.localBackend
        xData = x.toBackendData() if isinstance(x, operatorBase) else backend.array.toArray(x)
        data = backend.linalg.transpose(self.data) if trans else self.data
        return backend.linalg.dot(data, xData)

    def toBackendData(self) -> Any:
        return self.data

    def __mul__(self, scalar: float) -> 'matrixOperator':
        return matrixOperator(self.data * scalar, backendName=self.backendName)

    def toNumpy(self) -> Any:
        backend = self.localBackend
        return backend.array.toNumpy(self.data)

    def __repr__(self) -> str:
        return f"matrixOperator(shape={self.shape}, isSparse={self.isSparse}, backend={self.backendName})"
