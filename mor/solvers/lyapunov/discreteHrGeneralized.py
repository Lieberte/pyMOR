from .lyapunov import baseGeneralizedLyapunovSolver
from mor.operators import matrixOperator

class discreteHrGeneralizedLyapunovSolver(baseGeneralizedLyapunovSolver):
    def solve(
        self,
        a: matrixOperator,
        e: matrixOperator,
        b: matrixOperator
    ) -> matrixOperator:
        self._validateInputs(a, e, b)
        raise NotImplementedError(
            "Generalized discrete Lyapunov full (high-rank) solver not in SciPy. "
            "Use discreteLrGeneralizedLyapunovSolver for low-rank when implemented."
        )
