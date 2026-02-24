from mor.operators import matrixOperator
from mor.solvers.registry import registerLyapunovSolver
from .lyapunov import lyapunovSolver

@registerLyapunovSolver('continuousHrGeneralized')
class continuousHrGeneralizedSolver(lyapunovSolver):
    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator) -> matrixOperator:
        backend = self.localBackend
        Q = backend.linalg.dot(B.data, B.data.T)
        X_data = backend.lyapunov.solveContinuousGeneralized(A.data, E.data, Q)
        return matrixOperator(X_data, backendName=backend.name)
