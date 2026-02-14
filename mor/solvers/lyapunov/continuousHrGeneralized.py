import numpy as np

from .lyapunov import baseGeneralizedLyapunovSolver, _requireLyapunovSupport
from mor.operators import matrixOperator

class continuousHrGeneralizedLyapunovSolver(baseGeneralizedLyapunovSolver):

    def solve(
        self,
        a: matrixOperator,
        e: matrixOperator,
        b: matrixOperator
    ) -> matrixOperator:
        _requireLyapunovSupport(self.backend)
        self._validateInputs(a, e, b)
        if a.isSparse or e.isSparse or b.isSparse:
            return self._solveDense(a.toDense(), e.toDense(), b.toDense())
        return self._solveDense(a, e, b)

    def _solveDense(self, a: matrixOperator, e: matrixOperator, b: matrixOperator) -> matrixOperator:
        aData = a.toNumpy()
        eData = e.toNumpy()
        bData = b.toNumpy()
        if bData.ndim == 1:
            bData = bData[:, np.newaxis]
        q = bData @ bData.T
        xData = self.backend.lyapunov.solveContinuousGeneralized(aData, eData, q)
        return matrixOperator(xData, backendName=self.backendName)
