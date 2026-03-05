from typing import Any
import torch
from mor.operators import matrixOperator
from mor.algorithm.registry import registerAlgorithm
from .base import backendLyapunovAlgorithm

@registerAlgorithm('lyapunov', 'bartelsStewart')
class bartelsStewartAlgorithm(backendLyapunovAlgorithm):
    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator) -> Any:
        backend = self.localBackend
        q = self.prepareQ(B)
        if E is not None:
            return backend.linalg.solveGeneralizedContinuousLyapunov(A.data, E.data, q)
        return backend.linalg.solveContinuousLyapunov(A.data, q)
