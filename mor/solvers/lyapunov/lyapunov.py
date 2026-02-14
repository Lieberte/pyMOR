from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional
from mor.backends import backendRegistry
from mor.operators import matrixOperator

def _requireLyapunovSupport(backend) -> None:
    if not getattr(backend, 'supportsLyapunov', False):
        raise ValueError(f"Backend does not support Lyapunov.")

class lyapunovSolverBase(ABC):
    def __init__(self, globalBackendName: str, **kwargs):
        self.localBackend = backendRegistry.get(globalBackendName)
        self.options = kwargs

    @abstractmethod
    def solve(self, A: matrixOperator, E: Optional[matrixOperator], B: matrixOperator, C: matrixOperator) -> matrixOperator:
        pass

    @property
    def backendName(self) -> str:
        return self._backendName

    def _validateInputs(self, A: matrixOperator, E: Optional[matrixOperator], B: matrixOperator, C: matrixOperator) -> None:
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Matrix A must be square, got shape {A.shape}")
        if B.shape[0] != A.shape[0]:
            raise ValueError(f"Matrix B rows {B.shape[0]} must match A size {A.shape[0]}")
        if C.shape[1] != A.shape[0]:
            raise ValueError(f"Matrix C columns {C.shape[1]} must match A size {A.shape[0]}")
        if E is not None and E.shape != A.shape:
            raise ValueError(f"Matrix E shape {E.shape} must match A shape {A.shape}")

    def solveControllabilityAndObservability(self, A: matrixOperator, E: Optional[matrixOperator], B: matrixOperator, C: matrixOperator) -> Tuple[Any, Any, matrixOperator]:
        n = A.shape[0]
        backendName = self.localBackend.name
        Eeff = E if E is not None else matrixOperator(self.localBackend.array.eye(n, dtype=A.dtype), backendName=backendName)
        At = matrixOperator(A.toNumpy().T, backendName=backendName)
        Ct = matrixOperator(C.toNumpy().T, backendName=backendName)
        Et = matrixOperator(Eeff.toNumpy().T, backendName=backendName)
        Zc = self.solve(A, Eeff, B).toNumpy()
        Zo = self.solve(At, Et, Ct).toNumpy()
        return Zc, Zo, Eeff


