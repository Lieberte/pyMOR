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
        aData = backend.linalg.transpose(A.data) if trans else A.data
        if E is not None:
            eData = backend.linalg.transpose(E.data) if trans else E.data
            X = backend.linalg.solveGeneralizedContinuousLyapunov(aData, eData, Q)
        else:
            X = backend.linalg.solveContinuousLyapunov(aData, Q)
        return matrixOperator(X, backendName=backend.name)
