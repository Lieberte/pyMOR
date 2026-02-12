from .lyapunov import lyapunovSolverBase
from mor.operators import matrixOperator


class continuousHrLyapunovSolver(lyapunovSolverBase):

    def solve(self, a: matrixOperator, b: matrixOperator) -> matrixOperator:
        self._validateInputs(a, b)

        if a.isSparse or b.isSparse:
            return self._solveSparse(a, b)
        else:
            return self._solveDense(a, b)

    def _solveDense(self, a: matrixOperator, b: matrixOperator) -> matrixOperator:
        raise NotImplementedError


    def _solveSparse(self, a: matrixOperator, b: matrixOperator) -> matrixOperator:
        raise NotImplementedError