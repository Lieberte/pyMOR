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
        def eigvalsGeneralized(A: Any, B: Any) -> np.ndarray:
            raise NotImplementedError("Backend does not support generalized eigenvalues")

        @staticmethod
        def eigvals(A: Any) -> np.ndarray:
            raise NotImplementedError("Backend does not support eigenvalues")

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
