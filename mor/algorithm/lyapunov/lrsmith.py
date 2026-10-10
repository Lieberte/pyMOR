from dataclasses import dataclass
from typing import Any

from mor.backends import backendRegistry
from mor.operators import lowRankOperator, matrixOperator
from mor.algorithm.registry import registerAlgorithm
from mor.algorithm.lyapunov_discrete import estimateDiscreteRadius
from .base import backendLyapunov

@dataclass(frozen=True, slots=True)
class smithOptions:
    maxIter: int = 200
    tol: float = 1e-10
    maxRank: int | None = None

@registerAlgorithm('lyapunov', 'lrsmith')
class lrsmith(backendLyapunov):
    def __init__(self, backendName: str | None = None, **kwargs):
        self.localBackend, self.options = backendRegistry.get(backendName), kwargs
        self.lastResidualHistory: list[float] = []
        self.lastIterations = 0
        self.lastConverged = False
        self.lastResidual: float | None = None
        self.lastTolerance: float | None = None
        self.lastSpectralRadiusEstimate: float | None = None

    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator, trans: bool = False) -> Any:
        backend, bData = self.localBackend, B.toBackendData()
        self._resetStats()
        self.validateIterationOptions()
        maxRank = self.options.get('maxRank')
        if maxRank is not None:
            self.validateColumnOption('maxRank', maxRank)
        if self.options.get('checkStability', True):
            radius = estimateDiscreteRadius(A, E)
            self.lastSpectralRadiusEstimate = radius
            if radius is not None and radius >= 1.0:
                raise ValueError(f"LRSmith requires a stable discrete operator, got spectral radius {radius:.6g}.")
        if trans:
            bData = backend.linalg.transpose(bData)
        if backend.array.ndim(bData) == 1: bData = backend.array.reshape(bData, (-1, 1))
        descriptorSolve = self._descriptorSolver(E, trans=trans)
        effectiveRhs = descriptorSolve(bData)
        maxIter, tol, maxRank = self.options.get('maxIter', 200), self.options.get('tol', 1e-10), self.options.get('maxRank', None)
        z = backend.array.copy(effectiveRhs)
        bNorm = backend.linalg.norm(backend.linalg.dot(effectiveRhs.T, effectiveRhs), ord=2)
        bTol = backend.array.max(backend.array.array([bNorm * tol, 1e-20]))
        w = None
        self.lastTolerance = float(bTol)
        for iteration in range(1, maxIter + 1):
            transformedZ = descriptorSolve(A.apply(z, trans=trans))
            res = self.discreteLyapunovResidualNorm(transformedZ, z, effectiveRhs)
            self.lastResidualHistory.append(float(res))
            self.lastIterations = iteration
            if res <= bTol or iteration == maxIter:
                break
            if w is None:
                w = descriptorSolve(A.apply(effectiveRhs, trans=trans))
            z = backend.array.hstack([z, w])
            if maxRank is not None and backend.array.shape(z)[1] > maxRank:
                U, S, _ = backend.decomposition.svdDense(z, fullMatrices=False)
                z = backend.linalg.dot(U[:, :maxRank], backend.array.diag(S[:maxRank]))
            w = descriptorSolve(A.apply(w, trans=trans))
        self.lastResidual = float(res)
        self.lastConverged = bool(res <= bTol)
        return lowRankOperator(z, backendName=backend.name)

    def _resetStats(self) -> None:
        self.lastResidualHistory = []
        self.lastIterations = 0
        self.lastConverged = False
        self.lastResidual = None
        self.lastTolerance = None
        self.lastSpectralRadiusEstimate = None

    def _advance(self, A: matrixOperator, E: matrixOperator | None, z: Any, trans: bool = False) -> Any:
        applied = A.apply(z, trans=trans)
        return self._solveDescriptor(E, applied, trans=trans)

    def _solveDescriptor(self, E: matrixOperator | None, rhs: Any, trans: bool = False) -> Any:
        if E is None:
            return rhs
        eData = E.toBackendData()
        if trans:
            eData = self.localBackend.linalg.transpose(eData)
        return self.localBackend.linalg.solve(eData, rhs)

    def _descriptorSolver(self, E, trans=False):
        if E is None:
            return lambda rhs: rhs
        eData = E.toBackendData()
        if trans:
            eData = self.localBackend.linalg.transpose(eData)
        return self.localBackend.linalg.factorize(eData)

    def discreteLyapunovResidualNorm(self, u: Any, v: Any, w: Any) -> float:
        backend = self.localBackend
        factors = backend.array.hstack([u, v, w])
        _, triangular = backend.linalg.qr(factors, mode='reduced')
        uColumns, vColumns = u.shape[1], v.shape[1]
        ru = triangular[:, :uColumns]
        rv = triangular[:, uColumns:uColumns + vColumns]
        rw = triangular[:, uColumns + vColumns:]
        core = (
            backend.linalg.dot(ru, backend.linalg.transpose(ru))
            - backend.linalg.dot(rv, backend.linalg.transpose(rv))
            + backend.linalg.dot(rw, backend.linalg.transpose(rw))
        )
        return float(backend.linalg.norm(core))
