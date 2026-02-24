from mor.operators import matrixOperator
from mor.solvers.registry import registerLyapunovSolver
from .lyapunov import lyapunovSolver
from mor.algorithm.lyapunov import solveLyapunovLrGeneralized, shiftComputationOptions

@registerLyapunovSolver('continuousLrGeneralized')
class continuousLrGeneralizedSolver(lyapunovSolver):
    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator) -> matrixOperator:
        backend = self.localBackend
        tol = self.options.get('tol', 1e-10)
        maxIter = self.options.get('maxIter', 500)
        trans = self.options.get('trans', False)
        shiftOpts = shiftComputationOptions(
            initMaxiter=self.options.get('initMaxiter', 20),
            subspaceColumns=self.options.get('subspaceColumns', 6)
        )
        Z_data = solveLyapunovLrGeneralized(A, E, B, trans=trans, backendName=backend.name, tol=tol, maxIter=maxIter, shiftOptions=shiftOpts)
        return matrixOperator(Z_data, backendName=backend.name)
