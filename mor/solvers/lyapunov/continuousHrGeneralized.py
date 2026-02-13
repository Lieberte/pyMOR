from .lyapunov import baseGeneralizedLyapunovSolver
from mor.operators import matrixOperator

class continuousHrGeneralizedLyapunovSolver(baseGeneralizedLyapunovSolver):

    def solve(
        self,
        a: matrixOperator,
        e: matrixOperator,
        b: matrixOperator
    ) -> matrixOperator:
        self._validateInputs(a, e, b)
        raise NotImplementedError(
            "Generalized continuous Lyapunov full (high-rank) solver not in SciPy. "
            "Use continuousLrGeneralizedLyapunovSolver for low-rank."
        )
