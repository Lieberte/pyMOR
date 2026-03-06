from typing import Tuple, Any

from mor.backends import backendRegistry
from mor.operators import matrixOperator
from mor.solvers.registry import registerLyapunovSolver
from mor.algorithm.registry import algorithmRegistry

@registerLyapunovSolver('unified')
class lyapunovSolver:
    def __init__(self, backendName: str | None = None, **kwargs):
        self.localBackend, self.options = backendRegistry.get(backendName), kwargs

    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator, trans: bool = False) -> Any:
        backend = self.localBackend
        backendName = A.backendName or backend.name
        variant = self.options.get('lyapunovVariant', self.options.get('variant', 'auto'))
        algorithm = algorithmRegistry.get(category='lyapunov', variant=variant, forceOptions=self.options, backendName=backendName, A=A, E=E, B=B)
        self._lastAlgorithmName = algorithm.__class__.__name__
        return algorithm.solve(A, E, B, trans=trans)

    def validateInputs(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator, C: matrixOperator) -> None:
        if A.shape[0] != A.shape[1]: raise ValueError(f"Matrix A must be square, got shape {A.shape}")
        if E is not None and E.shape != A.shape: raise ValueError(f"Matrix E shape {E.shape} must match A shape {A.shape}")
        if B.shape[0] != A.shape[0]: raise ValueError(f"Matrix B rows {B.shape[0]} must match A size {A.shape[0]}")
        if C.shape[1] != A.shape[0]: raise ValueError(f"Matrix C columns {C.shape[1]} must match A size {A.shape[0]}")

    def solveControllabilityAndObservability(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator, C: matrixOperator, isContinuous: bool = True) -> Tuple[Any, Any, matrixOperator]:
        self.validateInputs(A, E, B, C)
        backend = self.localBackend
        if E is not None: return self.solve(A, E, B, trans=False), self.solve(A, E, C, trans=True), E
        n = A.shape[0]
        eEff = matrixOperator(backend.array.eye(n), backendName=backend.name)
        return self.solve(A, None, B, trans=False), self.solve(A, None, C, trans=True), eEff
