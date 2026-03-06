from abc import ABC, abstractmethod
from typing import Tuple, Any
import numpy as np

class backendBase(ABC):
    class linalg:
        @staticmethod
        @abstractmethod
        def solve(A: Any, b: Any) -> Any:
            pass

        @staticmethod
        @abstractmethod
        def solveTriangular(A: Any, b: Any, lower: bool = False) -> Any:
            pass

        @staticmethod
        @abstractmethod
        def qr(A: Any, mode: str = 'reduced') -> Tuple[Any, Any]:
            pass

        @staticmethod
        @abstractmethod
        def schur(A: Any, output: str = 'real') -> Tuple[Any, Any]:
            pass

        @staticmethod
        @abstractmethod
        def norm(x: Any, ord: Any = None) -> float:
            pass
        
        @staticmethod
        @abstractmethod
        def dot(a: Any, b: Any) -> Any:
            pass

        @staticmethod
        @abstractmethod
        def det(a: Any) -> float:
            pass

        @staticmethod
        @abstractmethod
        def slogdet(a: Any) -> Tuple[float, float]:
            pass

        @staticmethod
        @abstractmethod
        def solveContinuousLyapunov(A: Any, Q: Any) -> Any:
            pass

        @staticmethod
        @abstractmethod
        def solveGeneralizedContinuousLyapunov(A: Any, E: Any, Q: Any) -> Any:
            pass

        @staticmethod
        def transpose(a: Any) -> Any:
            return getattr(a, 'T', a)

        @staticmethod
        def conj(a: Any) -> Any:
            return getattr(a, 'conj', lambda: a)()

        @staticmethod
        @abstractmethod
        def balance(A: Any) -> Any:
            pass

    class decomposition:
        @staticmethod
        @abstractmethod
        def svdDense(A: Any, fullMatrices: bool = False) -> Tuple[Any, Any, Any]:
            pass

        @staticmethod
        @abstractmethod
        def svdSparse(A: Any, k: int, which: str = 'LM') -> Tuple[Any, Any, Any]:
            pass

        @staticmethod
        def qrOrthogonalize(B: Any, backend: 'backendBase') -> Any:
            Q, _ = backend.linalg.qr(B, mode='reduced')
            return Q

    class eigen:
        @staticmethod
        @abstractmethod
        def eigvalsGeneralized(A: Any, B: Any) -> Any:
            pass

        @staticmethod
        @abstractmethod
        def eigvals(A: Any) -> Any:
            pass

        @staticmethod
        @abstractmethod
        def eigh(A: Any) -> Tuple[Any, Any]:
            pass

    class array:
        @staticmethod
        @abstractmethod
        def zeros(shape: Tuple[int, ...], dtype: Any = None) -> Any:
            pass

        @staticmethod
        @abstractmethod
        def eye(n: int, dtype: Any = None) -> Any:
            pass

        @staticmethod
        @abstractmethod
        def eyeSparse(n: int, dtype: Any = None) -> Any:
            pass

        @staticmethod
        def eyeLike(a: Any) -> Any:
            pass

        @staticmethod
        def hstack(arrays: list[Any]) -> Any:
            pass

        @staticmethod
        @abstractmethod
        def vstack(arrays: list[Any]) -> Any:
            pass

        @staticmethod
        @abstractmethod
        def toNumpy(data: Any) -> np.ndarray:
            pass

        @staticmethod
        @abstractmethod
        def toArray(data: Any) -> Any:
            pass

        @staticmethod
        def diag(v: Any, k: int = 0) -> Any:
            pass

        @staticmethod
        def trace(a: Any) -> Any:
            pass

        @staticmethod
        def ndim(data: Any) -> int:
            return getattr(data, 'ndim', 0)

        @staticmethod
        def shape(data: Any) -> Tuple[int, ...]:
            return getattr(data, 'shape', ())

        @staticmethod
        def size(data: Any) -> int:
            return getattr(data, 'size', 0)

        @staticmethod
        def reshape(data: Any, shape: Tuple[int, ...]) -> Any:
            return getattr(data, 'reshape')(shape)

        @staticmethod
        def real(data: Any) -> Any:
            return getattr(data, 'real', data)

        @staticmethod
        def imag(data: Any) -> Any:
            return getattr(data, 'imag', 0.0)

        @staticmethod
        def abs(data: Any) -> Any:
            return np.abs(data) 

        @staticmethod
        def sqrt(data: Any) -> Any:
            return np.sqrt(data)

        @staticmethod
        def exp(data: Any) -> Any:
            return np.exp(data)

        @staticmethod
        def sum(data: Any, axis: int | None = None) -> Any:
            return getattr(data, 'sum')(axis=axis)

        @staticmethod
        def cumsum(data: Any, axis: int | None = None) -> Any:
            return getattr(data, 'cumsum')(axis=axis)

        @staticmethod
        def all(data: Any) -> bool:
            return bool(getattr(data, 'all')())

        @staticmethod
        def isfinite(data: Any) -> Any:
            return np.isfinite(data) 

        @staticmethod
        def where(condition: Any) -> Tuple[Any, ...]:
            return np.where(condition)

        @staticmethod
        def iscomplexobj(data: Any) -> bool:
            return np.iscomplexobj(data)

        @staticmethod
        def argsort(data: Any) -> Any:
            return np.argsort(data)

        @staticmethod
        def array(data: Any, dtype: Any = None) -> Any:
            return np.array(data, dtype=dtype)

        @staticmethod
        def randn(shape: Tuple[int, ...], dtype: Any = None) -> Any:
            return np.random.randn(*shape).astype(dtype) if dtype else np.random.randn(*shape)

        @staticmethod
        def copy(data: Any) -> Any:
            return getattr(data, 'copy')()

        @staticmethod
        def any(data: Any) -> bool:
            return np.any(data)

        @staticmethod
        def size(data: Any) -> int:
            return np.size(data)

        @staticmethod
        @abstractmethod
        def isSparse(data: Any) -> bool:
            pass

    class specialized:
        @staticmethod
        def gramMatrixNorm(w: Any, backend: 'backendBase') -> float:
            return backend.linalg.norm(backend.linalg.dot(w.T, w), ord=2)

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def arrayType(self) -> type:
        pass

    @classmethod
    def isAvailable(cls) -> bool:
        return True
