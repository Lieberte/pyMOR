import numpy as np

from .lyapunov import lyapunovSolverBase, _requireLyapunovSupport
from mor.operators import matrixOperator

class discreteHrLyapunovSolver(lyapunovSolverBase):
    def solve(self, a: matrixOperator, b: matrixOperator) -> matrixOperator:
        _requireLyapunovSupport(self.backend)
        self._validateInputs(a, b)
        if a.isSparse or b.isSparse:
            return self._solveSparse(a, b)
        return self._solveDense(a, b)

    def _solveDense(self, a: matrixOperator, b: matrixOperator) -> matrixOperator:
        aData = a.toNumpy()
        bData = b.toNumpy()
        if bData.ndim == 1:
            bData = bData[:, np.newaxis]
        q = bData @ bData.T
        xData = self.backend.lyapunov.solveDiscrete(aData, q)
        return matrixOperator(xData, backendName=self.backendName)

    def _solveSparse(self, a: matrixOperator, b: matrixOperator) -> matrixOperator:
        aDense = a.toDense()
        bDense = b.toDense()
        return self._solveDense(aDense, bDense)
