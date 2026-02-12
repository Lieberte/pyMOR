from .lyapunov import lyapunovSolverBase
from mor.operators import matrixOperator


class discreteLrLyapunovSolver(lyapunovSolverBase):

    def solve(self, a: matrixOperator, b: matrixOperator) -> matrixOperator:
        self._validateInputs(a, b)
        raise NotImplementedError(
            "Discrete Lyapunov low-rank solver not yet implemented. "
            "Requires LR-Smith or LR-Stein iteration algorithm.")
