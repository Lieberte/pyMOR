from abc import ABC, abstractmethod
from mor.backends import backendRegistry
from mor.operators import matrixOperator

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
