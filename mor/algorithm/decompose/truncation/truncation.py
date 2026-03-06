from typing import Any

def truncate(S: Any, rank: int | None = None, tol: float | None = None, backend: Any | None = None) -> int:
    if rank is not None:
        return min(int(rank), backend.array.size(S))
    if tol is not None:
        s2 = S**2
        energy = backend.array.cumsum(s2) / backend.array.sum(s2)
        # Find first index where energy >= (1 - tol)
        # Note: energy is 1-indexed for rank
        mask = energy >= (1.0 - tol)
        indices = backend.array.where(mask)[0]
        if backend.array.size(indices) > 0:
            return int(indices[0]) + 1
    return backend.array.size(S)
