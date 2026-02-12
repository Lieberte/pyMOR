from .lyapunov import baseGeneralizedLyapunovSolver
from mor.operators import matrixOperator


class discreteLrGeneralizedLyapunovSolver(baseGeneralizedLyapunovSolver):

    def solve(
        self,
        a: matrixOperator,
        e: matrixOperator,
        b: matrixOperator
    ) -> matrixOperator:
        self._validateInputs(a, e, b)
        raise NotImplementedError(
            "Discrete generalized Lyapunov low-rank solver not yet implemented. "
            "Requires generalized LR-Smith or LR-Stein iteration algorithm."
        )
