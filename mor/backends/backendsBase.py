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
        def qr(A: Any, mode: str = 'reduced') -> Tuple[Any, Any]:
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
        def transpose(a: Any) -> Any:
            return getattr(a, 'T', a)

        @staticmethod
        def conj(a: Any) -> Any:
            return getattr(a, 'conj', lambda: a)()

        @staticmethod
        def robustSqrtFactor(A: Any, tol: float | None = None, name: str = "Matrix") -> Any:
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
        def eyeLike(a: Any) -> Any:
            pass

        @staticmethod
        @abstractmethod
        def hstack(arrays: list[Any]) -> Any:
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
        def sum(data: Any, axis: int | None = None) -> Any:
            return getattr(data, 'sum')(axis=axis)

        @staticmethod
        def all(data: Any) -> bool:
            return bool(getattr(data, 'all')())

        @staticmethod
        def isfinite(data: Any) -> Any:
            return np.isfinite(data) 

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
        def copy(data: Any) -> Any:
            return getattr(data, 'copy')()

        @staticmethod
        @abstractmethod
        def isSparse(data: Any) -> bool:
            pass

    class specialized:
        @staticmethod
        def gramMatrixNorm(w: Any, backend: 'backendBase') -> float:
            return backend.linalg.norm(backend.linalg.dot(w.T, w), ord=2)

    class lyapunov:
        @staticmethod
        def solveContinuous(a: Any, q: Any) -> Any:
            pass

        @staticmethod
        def solveDiscrete(a: Any, q: Any) -> Any:
            pass

        @staticmethod
        def solveContinuousGeneralized(a: Any, e: Any, q: Any) -> Any:
            pass

        @staticmethod
        def solveDiscreteGeneralized(a: Any, e: Any, q: Any) -> Any:
            pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def arrayType(self) -> type:
        pass

    @property
    def supportsLyapunov(self) -> bool:
        return False

    @classmethod
    def isAvailable(cls) -> bool:
        return True
