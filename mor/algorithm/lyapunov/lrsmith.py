import numpy as np
from dataclasses import dataclass
from typing import Any

from mor.backends import backendRegistry
from mor.operators import matrixOperator

@dataclass(frozen=True, slots=True)
class smithOptions:
    maxIter: int = 200
    tol: float = 1e-10

def _discreteLyapunovResidualNorm(u: np.ndarray, v: np.ndarray, w: np.ndarray, backend: Any) -> float:
    utu = backend.linalg.dot(u.T, u)
    vtv = backend.linalg.dot(v.T, v)
    wtw = backend.linalg.dot(w.T, w)
    utv = backend.linalg.dot(u.T, v)
    utw = backend.linalg.dot(u.T, w)
    vtw = backend.linalg.dot(v.T, w)
    tr2 = np.trace(utu @ utu) + np.trace(vtv @ vtv) + np.trace(wtw @ wtw)
    tr2 += -2 * np.sum(utv ** 2) + 2 * np.sum(utw ** 2) - 2 * np.sum(vtw ** 2)
    return float(np.sqrt(max(0.0, tr2)))

def solveLyapunovLrDiscrete(A: matrixOperator,B: matrixOperator,*,backendName: str = 'numpy',maxIter: int = 200,tol: float = 1e-10,maxRank: int | None = None,) -> np.ndarray:
    backend = backendRegistry.get(backendName)
    bData = B.toNumpy()
    if bData.ndim == 1:
        bData = bData[:, np.newaxis]
    z = backend.decomposition.qrOrthogonalize(bData, backend)
    bNorm = backend.linalg.norm(backend.linalg.dot(bData.T, bData), ord=2)
    bTol = max(bNorm * tol, 1e-20)
    w = A.apply(z)
    for _ in range(maxIter - 1):
        res = _discreteLyapunovResidualNorm(w, z, bData, backend)
        if res <= bTol:
            break
        zNew = backend.array.hstack([w, bData])
        z = backend.decomposition.qrOrthogonalize(zNew, backend)
        w = A.apply(z)
        if maxRank is not None and z.shape[1] > maxRank:
            U, S, Vt = backend.decomposition.svdDense(z, fullMatrices=False)
            z = backend.linalg.dot(U[:, :maxRank], np.diag(S[:maxRank]))
            w = A.apply(z)
    return z

def solveLyapunovLrDiscreteGeneralized(A: matrixOperator, E: matrixOperator,B: matrixOperator,*,backendName: str = 'numpy',maxIter: int = 200,tol: float = 1e-10,maxRank: int | None = None,) -> np.ndarray:
    backend = backendRegistry.get(backendName)
    aData = A.toNumpy()
    eData = E.toNumpy()
    bData = B.toNumpy()
    if bData.ndim == 1:
        bData = bData[:, np.newaxis]
    aTilde = backend.linalg.solve(eData, aData)
    bTilde = backend.linalg.solve(eData, bData)
    aTildeOp = matrixOperator(aTilde, backendName=backendName)
    bTildeOp = matrixOperator(bTilde, backendName=backendName)
    z = solveLyapunovLrDiscrete(aTildeOp, bTildeOp,backendName=backendName,maxIter=maxIter,tol=tol,maxRank=maxRank,)
    return backend.linalg.solve(eData, z)
