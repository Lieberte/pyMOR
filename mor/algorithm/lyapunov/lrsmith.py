from dataclasses import dataclass
from typing import Any

from mor.backends import backendRegistry
from mor.operators import lowRankOperator, matrixOperator
from mor.algorithm.registry import registerAlgorithm
from .base import backendLyapunov

@dataclass(frozen=True, slots=True)
class smithOptions:
    maxIter, tol, maxRank = 200, 1e-10, None

@registerAlgorithm('lyapunov', 'lrsmith')
class lrsmith(backendLyapunov):
    def __init__(self, backendName: str | None = None, **kwargs):
        self.localBackend, self.options = backendRegistry.get(backendName), kwargs

    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator, trans: bool = False) -> Any:
        backend, bData = self.localBackend, B.toBackendData()
        if trans:
            bData = backend.linalg.transpose(bData)
        if backend.array.ndim(bData) == 1: bData = backend.array.reshape(bData, (-1, 1))
        maxIter, tol, maxRank = self.options.get('maxIter', 200), self.options.get('tol', 1e-10), self.options.get('maxRank', None)
        z = backend.decomposition.qrOrthogonalize(bData, backend)
        bNorm = backend.linalg.norm(backend.linalg.dot(bData.T, bData), ord=2)
        bTol = backend.array.max(backend.array.array([bNorm * tol, 1e-20]))
        w = self._advance(A, E, z, trans=trans)
        for _ in range(maxIter - 1):
            if self.discreteLyapunovResidualNorm(w, z, bData) <= bTol: break
            z = backend.decomposition.qrOrthogonalize(backend.array.hstack([w, bData]), backend)
            w = self._advance(A, E, z, trans=trans)
            if maxRank is not None and backend.array.shape(z)[1] > maxRank:
                U, S, Vt = backend.decomposition.svdDense(z, fullMatrices=False)
                z = backend.linalg.dot(U[:, :maxRank], backend.array.diag(S[:maxRank]))
                w = self._advance(A, E, z, trans=trans)
        return lowRankOperator(z, backendName=backend.name)

    def _advance(self, A: matrixOperator, E: matrixOperator | None, z: Any, trans: bool = False) -> Any:
        applied = A.apply(z, trans=trans)
        if E is None:
            return applied
        eData = E.toBackendData()
        if trans:
            eData = self.localBackend.linalg.transpose(eData)
        return self.localBackend.linalg.solve(eData, applied)

    def discreteLyapunovResidualNorm(self, u: Any, v: Any, w: Any) -> float:
        backend = self.localBackend
        utu, vtv, wtw, utv, utw, vtw = backend.linalg.dot(u.T, u), backend.linalg.dot(v.T, v), backend.linalg.dot(w.T, w), backend.linalg.dot(u.T, v), backend.linalg.dot(u.T, w), backend.linalg.dot(v.T, w)
        tr2 = backend.array.sum(utu * utu) + backend.array.sum(vtv * vtv) + backend.array.sum(wtw * wtw) - 2 * backend.array.sum(utv ** 2) + 2 * backend.array.sum(utw ** 2) - 2 * backend.array.sum(vtw ** 2)        
        return float(backend.array.sqrt(backend.array.max(backend.array.array([0.0, tr2]))))
