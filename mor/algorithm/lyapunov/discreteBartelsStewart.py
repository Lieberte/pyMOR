import numpy as np
from scipy import linalg

from mor.algorithm.registry import registerAlgorithm
from mor.backends.scipyBackend import _denseDecompositionData
from mor.operators import matrixOperator
from .base import backendLyapunov


@registerAlgorithm('lyapunov', 'discreteBartelsStewart')
class discreteBartelsStewart(backendLyapunov):
    def solve(self, A, E, B, trans=False):
        if self.localBackend.name != 'scipy':
            raise NotImplementedError('discreteBartelsStewart requires the scipy backend; use lrsmith for Torch')
        self.lastConverged = False
        a, b = _denseDecompositionData(A.toBackendData()), _denseDecompositionData(B.toBackendData())
        e = _denseDecompositionData(E.toBackendData()) if E is not None else np.eye(a.shape[0])
        if any(np.iscomplexobj(value) for value in (a, b, e)):
            raise ValueError('discreteBartelsStewart currently supports real operators')
        if a.ndim != 2 or a.shape[0] != a.shape[1] or e.shape != a.shape:
            raise ValueError('A must be square and E must match A')
        if trans:
            a, e, b = a.T, e.T, b.T
        if b.ndim == 1: b = b[:, None]
        if b.ndim != 2 or b.shape[0] != a.shape[0]:
            raise ValueError('B must match the state dimension')
        if not all(np.all(np.isfinite(value)) for value in (a, b, e)):
            raise ValueError('discrete Lyapunov inputs must be finite')
        descriptorSolve = self.localBackend.linalg.factorize(e)
        effectiveA, effectiveB = descriptorSolve(a), descriptorSolve(b)
        radius = float(np.max(np.abs(linalg.eigvals(effectiveA))))
        self.lastSpectralRadiusEstimate = radius
        if radius >= 1.:
            raise ValueError('Discrete Lyapunov requires a stable operator with spectral radius below one')
        q = b @ b.T
        x = linalg.solve_discrete_lyapunov(effectiveA, effectiveB @ effectiveB.T)
        self.lastTolerance = float(self.options.get('acceptanceResidual', 1e-8))
        if not np.isfinite(self.lastTolerance) or self.lastTolerance < 0:
            raise ValueError('acceptanceResidual must be finite and non-negative')
        for iteration in range(3):
            residual = a @ x @ a.T - e @ x @ e.T + q
            self.lastResidual = float(linalg.norm(residual) / max(linalg.norm(q), np.finfo(float).tiny))
            if self.lastResidual <= self.lastTolerance or iteration == 2: break
            transformed = descriptorSolve(descriptorSolve(residual).T).T
            x += linalg.solve_discrete_lyapunov(effectiveA, transformed)
            x = (x + x.T) / 2
        self.lastIterations = iteration + 1
        self.lastConverged = bool(np.isfinite(self.lastResidual) and self.lastResidual <= self.lastTolerance)
        if not self.lastConverged:
            raise ValueError(f'Discrete Lyapunov residual {self.lastResidual:.3g} exceeds acceptance tolerance')
        return matrixOperator(x, backendName='scipy')
