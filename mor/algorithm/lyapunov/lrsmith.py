from dataclasses import dataclass
from typing import Any, Optional
from mor.backends import backendRegistry
from mor.operators import matrixOperator
from mor.algorithm.registry import registerAlgorithm

@dataclass(frozen=True, slots=True)
class smithOptions:
    maxIter: int = 200
    tol: float = 1e-10
    maxRank: int | None = None

@registerAlgorithm('lyapunov', 'lrsmith')
class lrsmithAlgorithm:
    def __init__(self, backendName: str | None = None, **kwargs):
        self.localBackend = backendRegistry.get(backendName)
        self.options = kwargs

    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator) -> Any:
        backend = self.localBackend
        n = A.shape[0]
        bData = B.data
        if backend.array.ndim(bData) == 1:
            bData = backend.array.reshape(bData, (-1, 1))
        maxIter = self.options.get('maxIter', 200)
        tol = self.options.get('tol', 1e-10)
        maxRank = self.options.get('maxRank', None)
        z = backend.decomposition.qrOrthogonalize(bData, backend)
        bNorm = backend.linalg.norm(backend.linalg.dot(bData.T, bData), ord=2)
        bTol = backend.array.max(backend.array.array([bNorm * tol, 1e-20]))
        if E is not None:
            az = A.apply(z)
            w = backend.linalg.solve(E.data, az)
        else:
            w = A.apply(z)
        for _ in range(maxIter - 1):
            res = self.discreteLyapunovResidualNorm(w, z, bData)
            if res <= bTol:
                break
            zNew = backend.array.hstack([w, bData])
            z = backend.decomposition.qrOrthogonalize(zNew, backend)
            if E is not None:
                az = A.apply(z)
                w = backend.linalg.solve(E.data, az)
            else:
                w = A.apply(z)
            if maxRank is not None and backend.array.shape(z)[1] > maxRank:
                U, S, Vt = backend.decomposition.svdDense(z, fullMatrices=False)
                z = backend.linalg.dot(U[:, :maxRank], backend.array.diag(S[:maxRank]))
                if E is not None:
                    az = A.apply(z)
                    w = backend.linalg.solve(E.data, az)
                else:
                    w = A.apply(z)
        return z

    def discreteLyapunovResidualNorm(self, u: Any, v: Any, w: Any) -> float:
        backend = self.localBackend
        utu = backend.linalg.dot(u.T, u)
        vtv = backend.linalg.dot(v.T, v)
        wtw = backend.linalg.dot(w.T, w)
        utv = backend.linalg.dot(u.T, v)
        utw = backend.linalg.dot(u.T, w)
        vtw = backend.linalg.dot(v.T, w)
        tr2 = backend.array.sum(utu * utu) + backend.array.sum(vtv * vtv) + backend.array.sum(wtw * wtw)
        tr2 += -2 * backend.array.sum(utv ** 2) + 2 * backend.array.sum(utw ** 2) - 2 * backend.array.sum(vtw ** 2)        
        return float(backend.array.sqrt(backend.array.max(backend.array.array([0.0, tr2]))))
