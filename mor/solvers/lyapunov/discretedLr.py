from .lyapunov import lyapunovSolverBase
from mor.operators import matrixOperator
from mor.algorithm.lyapunov import solveLyapunovLrDiscrete


class discreteLrLyapunovSolver(lyapunovSolverBase):
    def solve(self, a: matrixOperator, b: matrixOperator) -> matrixOperator:
        self._validateInputs(a, b)
        maxIter = self.options.get('maxIter', 200)
        tol = self.options.get('tol', 1e-10)
        maxRank = self.options.get('maxRank', None)
        zData = solveLyapunovLrDiscrete(
            a, b,
            backendName=self.backendName,
            maxIter=maxIter,
            tol=tol,
            maxRank=maxRank
        )
        return matrixOperator(zData, backendName=self.backendName)
