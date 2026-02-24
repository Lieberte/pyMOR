from mor.operators import matrixOperator
from mor.solvers.registry import registerLyapunovSolver
from .lyapunov import lyapunovSolver
from mor.algorithm.lyapunov import solveLyapunovLrDiscreteGeneralized

@registerLyapunovSolver('discreteLrGeneralized')
class discreteLrGeneralizedSolver(lyapunovSolver):
    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator) -> matrixOperator:
        backend = self.localBackend
        Z_data = solveLyapunovLrDiscreteGeneralized(
            A, E, B, 
            backendName=backend.name, 
            maxIter=self.options.get('maxIter', 200), 
            tol=self.options.get('tol', 1e-10), 
            maxRank=self.options.get('maxRank', None)
        )
        return matrixOperator(Z_data, backendName=backend.name)
