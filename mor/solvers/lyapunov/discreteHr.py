from mor.operators import matrixOperator
from mor.solvers.registry import registerLyapunovSolver
from .lyapunov import lyapunovSolver

@registerLyapunovSolver('discreteHr')
class discreteHrSolver(lyapunovSolver):
    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator) -> matrixOperator:
        backend = self.localBackend
        Q = backend.linalg.dot(B.data, B.data.T)
        X_data = backend.lyapunov.solveDiscrete(A.data, Q)
        return matrixOperator(X_data, backendName=backend.name)
