from typing import Any
from numbers import Integral, Real

def truncate(S: Any, rank: int | None = None, tol: float | None = None, backend: Any | None = None) -> int:
    if rank is not None:
        if not isinstance(rank, Integral) or isinstance(rank, bool) or rank < 0:
            raise ValueError("rank must be a non-negative integer.")
        return min(int(rank), backend.array.size(S))
    if tol is not None:
        if not isinstance(tol, Real) or isinstance(tol, bool) or not 0 <= tol <= 1:
            raise ValueError("tol must be a real number between 0 and 1.")
        s2 = S**2
        totalEnergy = backend.array.sum(s2)
        if float(totalEnergy) == 0:
            return 0
        energy = backend.array.cumsum(s2) / totalEnergy
        # Find first index where energy >= (1 - tol)
        # Note: energy is 1-indexed for rank
        mask = energy >= (1.0 - tol)
        indices = backend.array.where(mask)[0]
        if backend.array.size(indices) > 0:
            return int(indices[0]) + 1
    return backend.array.size(S)
