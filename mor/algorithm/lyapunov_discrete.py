from typing import Any

import numpy as np


def estimateDiscreteRadius(A, E=None) -> float | None:
    if A is None:
        return None
    if not getattr(A, 'isDiagonal', False) and getattr(A, 'shape', (0, 0))[0] > 300:
        return None
    try:
        aData = A.toBackendData()
        if E is not None:
            matrix = np.linalg.solve(toDense(E.toBackendData()), toDense(aData))
        elif getattr(A, 'isDiagonal', False):
            return float(np.max(np.abs(diagonalValues(aData))))
        else:
            matrix = toDense(aData)
        return float(np.max(np.abs(np.linalg.eigvals(matrix))))
    except Exception:
        return None


def toDense(data: Any):
    return data.toarray() if hasattr(data, 'toarray') else np.asarray(data)


def diagonalValues(data: Any):
    if hasattr(data, 'diagonal'):
        return np.asarray(data.diagonal())
    return np.diag(np.asarray(data))
