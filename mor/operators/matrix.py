import numpy as np
from typing import Tuple, Any, Optional
from .operatorsBase import operatorBase

class matrixOperator(operatorBase):

    def __init__(self, data: Any, backendName: str = 'numpy'):
        super().__init__(backendName)
        self.data = data
        self._format = self._detectFormat()

    def _detectFormat(self) -> str:
        import scipy.sparse as sp
        if sp.issparse(self.data):
            return 'sparse'
        return 'dense'

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def nRows(self) -> int:
        return self.shape[0]

    @property
    def nCols(self) -> int:
        return self.shape[1]

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def format(self) -> str:
        return self._format

    @property
    def isSparse(self) -> bool:
        return self._format == 'sparse'

    @property
    def isDense(self) -> bool:
        return self._format == 'dense'

    @property
    def T(self) -> 'matrixOperator':
        return matrixOperator(self.data.T, backendName=self.backendName)

    def transpose(self) -> 'matrixOperator':
        return self.T

    def __matmul__(self, other: Any) -> 'matrixOperator':
        if isinstance(other, matrixOperator):
            result = self.data @ other.data
        elif isinstance(other, np.ndarray):
            result = self.data @ other
        else:
            raise TypeError(f"Unsupported operand type for @: {type(other)}")
        return matrixOperator(result, backendName=self.backendName)

    def apply(self, x: Any) -> Any:
        if self.isSparse:
            return self.data @ x
        else:
            return self.backend.linalg.dot(self.data, x)

    def toDense(self) -> 'matrixOperator':
        if self.isDense:
            return self
        return matrixOperator(self.data.toarray(), backendName=self.backendName)

    def toSparse(self, format: str = 'csr') -> 'matrixOperator':
        import scipy.sparse as sp
        if self.isSparse:
            return self
        sparseData = sp.csr_matrix(self.data) if format == 'csr' else sp.csc_matrix(self.data)
        return matrixOperator(sparseData, backendName=self.backendName)

    def solveShifted(self, E: Optional['matrixOperator'], shift: complex, rhs: Any, trans: bool = False,) -> Any:
        n = self.shape[0]
        a = self.toNumpy()
        e = E.toNumpy() if E is not None else np.eye(n, dtype=a.dtype)
        m = a + shift * e
        if trans:
            return self.backend.linalg.solve(m.T, rhs)
        return self.backend.linalg.solve(m, rhs)

    def svd(self, fullMatrices: bool = False, rank: Optional[int] = None) -> Tuple[Any, Any, Any]:
        if self.isSparse:
            if rank is None:
                rank = min(self.shape) - 1
            return self.backend.decomposition.svdSparse(self.data, k=rank)
        else:
            U, S, Vt = self.backend.decomposition.svdDense(self.data, fullMatrices=fullMatrices)
            if rank is not None:
                U = U[:, :rank]
                S = S[:rank]
                Vt = Vt[:rank, :]
            return U, S, Vt

    def toNumpy(self) -> np.ndarray:
        if self.isSparse:
            return self.data.toarray()
        return self.backend.array.toNumpy(self.data)

    def norm(self, ord=None) -> float:
        if self.isSparse:
            import scipy.sparse.linalg as spla
            return spla.norm(self.data, ord=ord)
        else:
            return self.backend.linalg.norm(self.data, ord=ord)

    def __repr__(self) -> str:
        return f"matrixOperator(shape={self.shape}, format={self.format}, backend={self.backendName})"