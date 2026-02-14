from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional
from mor.backends import backendRegistry
from mor.operators import matrixOperator

def _requireLyapunovSupport(backend) -> None:
    if not getattr(backend, 'supportsLyapunov', False):
        raise ValueError(
            f"Backend '{backend.name}' does not support Lyapunov. Use backend 'scipy'."
        )

class lyapunovSolverBase(ABC):
    def __init__(self, backendName: str = 'numpy', **kwargs):
        self.setBackend(backendName)
        self.options = kwargs

    @abstractmethod
    def solve(self, a: matrixOperator, b: matrixOperator) -> matrixOperator:
        pass

    def setBackend(self, backendName: str):
        self.backend = backendRegistry.get(backendName)
        self._backendName = backendName

    @property
    def backendName(self) -> str:
        return self._backendName

    def _validateInputs(self, a: matrixOperator, b: matrixOperator) -> None:
        if a.shape[0] != a.shape[1]:
            raise ValueError(f"Matrix A must be square, got shape {a.shape}")

        if b.shape[0] != a.shape[0]:
            raise ValueError(f"Matrix B rows {b.shape[0]} must match A size {a.shape[0]}")

    def solveControllabilityAndObservability(self, A: matrixOperator, E: Optional[matrixOperator], B: matrixOperator, C: matrixOperator) -> Tuple[Any, Any, matrixOperator]:
        n = A.shape[0]
        backendName = self.backendName
        Eeff = E if E is not None else matrixOperator(self.backend.array.eye(n, dtype=A.dtype), backendName=backendName)
        At = matrixOperator(A.toNumpy().T, backendName=backendName)
        Ct = matrixOperator(C.toNumpy().T, backendName=backendName)
        Et = matrixOperator(Eeff.toNumpy().T, backendName=backendName)
        Zc = self.solve(A, B).toNumpy()
        Zo = self.solve(At, Ct).toNumpy()
        return Zc, Zo, Eeff

class baseGeneralizedLyapunovSolver(ABC):
    def __init__(self, backendName: str = 'numpy', **kwargs):
        self.setBackend(backendName)
        self.options = kwargs

    @abstractmethod
    def solve(self,a: matrixOperator,e: matrixOperator,b: matrixOperator) -> matrixOperator:
        pass

    def setBackend(self, backendName: str):
        self.backend = backendRegistry.get(backendName)
        self._backendName = backendName

    @property
    def backendName(self) -> str:
        return self._backendName

    def _validateInputs(self,a: matrixOperator,e: matrixOperator,b: matrixOperator) -> None:
        if a.shape[0] != a.shape[1]:
            raise ValueError(f"Matrix A must be square, got shape {a.shape}")

        if e.shape != a.shape:
            raise ValueError(f"Matrix E shape {e.shape} must match A shape {a.shape}")

        if b.shape[0] != a.shape[0]:
            raise ValueError(f"Matrix B rows {b.shape[0]} must match A size {a.shape[0]}")

    def solveControllabilityAndObservability(self, A: matrixOperator, E: Optional[matrixOperator], B: matrixOperator, C: matrixOperator) -> Tuple[Any, Any, matrixOperator]:
        n = A.shape[0]
        backendName = self.backendName
        Eeff = E if E is not None else matrixOperator(self.backend.array.eye(n, dtype=A.dtype), backendName=backendName)
        At = matrixOperator(A.toNumpy().T, backendName=backendName)
        Ct = matrixOperator(C.toNumpy().T, backendName=backendName)
        Et = matrixOperator(Eeff.toNumpy().T, backendName=backendName)
        Zc = self.solve(A, Eeff, B).toNumpy()
        Zo = self.solve(At, Et, Ct).toNumpy()
        return Zc, Zo, Eeff
