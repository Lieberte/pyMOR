from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import csc_matrix, eye, issparse
from scipy.sparse.linalg import splu


def validateFrequencies(frequencies: Any, *, allowEmpty: bool = True) -> np.ndarray:
    values = np.asarray(frequencies, dtype=float).reshape(-1)
    if not allowEmpty and values.size == 0:
        raise ValueError("frequencies must be non-empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("frequencies must be finite")
    if np.any(values < 0):
        raise ValueError("frequencies must be non-negative")
    return values


def frequencyPoints(frequencies: np.ndarray, *, isContinuous: bool) -> np.ndarray:
    return 1j * frequencies if isContinuous else np.exp(1j * frequencies)


def prepareTransferData(
    A: Any,
    B: Any,
    C: Any,
    E: Any | None = None,
    D: Any | None = None,
    *,
    forceComplex: bool = True,
):
    def dense(value: Any) -> np.ndarray:
        return value.toarray() if issparse(value) else np.asarray(value)

    aDtype = A.dtype if hasattr(A, 'dtype') else np.asarray(A).dtype
    bData = dense(B)
    cData = dense(C)
    eDtype = aDtype if E is None else (E.dtype if hasattr(E, 'dtype') else np.asarray(E).dtype)
    dDtype = aDtype if D is None else dense(D).dtype
    requiredDtype = np.complex64 if forceComplex else np.float64
    dtype = np.result_type(aDtype, bData.dtype, cData.dtype, eDtype, dDtype, requiredDtype)

    useSparse = issparse(A) or (E is not None and issparse(E))
    aData = csc_matrix(A, dtype=dtype) if useSparse else np.asarray(A, dtype=dtype)
    if E is None:
        eData = eye(aData.shape[0], dtype=dtype, format='csc') if useSparse else np.eye(aData.shape[0], dtype=dtype)
    else:
        eData = csc_matrix(E, dtype=dtype) if useSparse else np.asarray(E, dtype=dtype)
    bData = np.asarray(bData, dtype=dtype)
    cData = np.asarray(cData, dtype=dtype)
    if bData.ndim == 1:
        bData = bData.reshape(-1, 1)
    dData = (
        np.zeros((cData.shape[0], bData.shape[1]), dtype=dtype)
        if D is None else np.asarray(dense(D), dtype=dtype)
    )
    return aData, bData, cData, eData, dData


def solveShiftedSystem(A: Any, E: Any, B: np.ndarray, point: complex) -> np.ndarray:
    shifted = point * E - A
    if issparse(shifted):
        return splu(csc_matrix(shifted)).solve(B)
    return np.linalg.solve(shifted, B)


def transferResponse(
    A: Any,
    B: Any,
    C: Any,
    frequencies: Any,
    E: Any | None = None,
    D: Any | None = None,
    *,
    isContinuous: bool = True,
) -> np.ndarray:
    """Evaluate transfer values at rad/s or rad/sample frequencies."""
    values = validateFrequencies(frequencies)
    aData, bData, cData, eData, dData = prepareTransferData(A, B, C, E, D)
    if values.size == 0:
        return np.empty((0, cData.shape[0], bData.shape[1]), dtype=aData.dtype)
    return np.asarray([
        cData @ solveShiftedSystem(aData, eData, bData, point) + dData
        for point in frequencyPoints(values, isContinuous=isContinuous)
    ])
