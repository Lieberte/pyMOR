import numpy as np
from dataclasses import dataclass
from typing import Any

from mor.backends import backendRegistry
from mor.operators import matrixOperator

@dataclass(frozen=True, slots=True)
class smithOptions:
    maxIter: int = 200
    tol: float = 1e-10

def solveLyapunovLrDiscrete(
    A: matrixOperator,
    B: matrixOperator,
    *,
    backendName: str = 'numpy',
    maxIter: int = 200,
    tol: float = 1e-10,
    maxRank: int | None = None,
) -> np.ndarray:
    backend = backendRegistry.get(backendName)
    bData = B.toNumpy()
    if bData.ndim == 1:
        bData = bData[:, np.newaxis]
    z = backend.decomposition.qrOrthogonalize(bData, backend)
    for _ in range(maxIter - 1):
        w = A.apply(z)
        zNew = backend.array.hstack([w, bData])
        z = backend.decomposition.qrOrthogonalize(zNew, backend)
        if maxRank is not None and z.shape[1] > maxRank:
            U, S, Vt = backend.decomposition.svdDense(z, fullMatrices=False)
            z = backend.linalg.dot(U[:, :maxRank], np.diag(S[:maxRank]))
    return z

def solveLyapunovLrDiscreteGeneralized(
    A: matrixOperator,
    E: matrixOperator,
    B: matrixOperator,
    *,
    backendName: str = 'numpy',
    maxIter: int = 200,
    tol: float = 1e-10,
) -> np.ndarray:
    raise NotImplementedError(
        "Generalized discrete Lyapunov LR-Smith not yet implemented. "
        "Requires solving E Y E' = A Z Z' A' + B B' each step."
    )
