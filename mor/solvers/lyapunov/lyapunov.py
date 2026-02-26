from typing import Tuple, Any

from mor.backends import backendRegistry
from mor.operators import matrixOperator
from mor.solvers.registry import registerLyapunovSolver
from mor.algorithm.registry import algorithmRegistry

@registerLyapunovSolver('unified')
class lyapunovSolver:
    def __init__(self, backendName: str | None = None, **kwargs):
        self.localBackend, self.options = backendRegistry.get(backendName), kwargs

    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator) -> matrixOperator:
        backend = self.localBackend
        algorithm = algorithmRegistry.get(category='lyapunov', variant=self.options.get('variant', 'auto'), forceOptions=self.options, backendName=backend.name, A=A, E=E, B=B)
        return matrixOperator(algorithm.solve(A, E, B), backendName=backend.name)

    def validateInputs(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator, C: matrixOperator) -> None:
        if A.shape[0] != A.shape[1]: raise ValueError(f"Matrix A must be square, got shape {A.shape}")
        if E is not None and E.shape != A.shape: raise ValueError(f"Matrix E shape {E.shape} must match A shape {A.shape}")
        if B.shape[0] != A.shape[0]: raise ValueError(f"Matrix B rows {B.shape[0]} must match A size {A.shape[0]}")
        if C.shape[1] != A.shape[0]: raise ValueError(f"Matrix C columns {C.shape[1]} must match A size {A.shape[0]}")

    def solveControllabilityAndObservability(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator, C: matrixOperator, isContinuous: bool = True) -> Tuple[matrixOperator, matrixOperator, matrixOperator]:
        self.validateInputs(A, E, B, C)
        backend = self.localBackend
        At, Ct = matrixOperator(backend.linalg.transpose(A.data), backendName=backend.name), matrixOperator(backend.linalg.transpose(C.data), backendName=backend.name)
        if E is not None:
            Et = matrixOperator(backend.linalg.transpose(E.data), backendName=backend.name)
            return self.solve(A, E, B), self.solve(At, Et, Ct), E
        n = A.shape[0]
        eEff = matrixOperator(backend.array.eye(n, dtype=A.dtype), backendName=backend.name)
        return self.solve(A, None, B), self.solve(At, None, Ct), eEff
