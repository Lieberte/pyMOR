from typing import Any
from mor.operators import matrixOperator
from mor.algorithm.registry import registerAlgorithm
from .base import backendLyapunov

@registerAlgorithm('lyapunov', 'bartelsStewart')
class bartelsStewart(backendLyapunov):
    # TODO: Add support for generalized Schur decomposition in torchBackend
    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator, trans: bool = False) -> Any:
        backend = self.localBackend
        Q = self.prepareQ(B, trans=trans)
        aData = A.toBackendData()
        if backend.name == 'scipy':
            aData = backend.array.toNumpy(aData)
        if trans:
            aData = backend.linalg.transpose(aData)
        if E is not None:
            eData = E.toBackendData()
            if backend.name == 'scipy':
                eData = backend.array.toNumpy(eData)
            if trans:
                eData = backend.linalg.transpose(eData)
            X = backend.linalg.solveGeneralizedContinuousLyapunov(aData, eData, Q)
        else:
            X = backend.linalg.solveContinuousLyapunov(aData, Q)
        return matrixOperator(X, backendName=backend.name)
