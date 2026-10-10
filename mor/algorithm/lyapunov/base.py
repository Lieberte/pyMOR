from typing import Any
from math import isfinite
from numbers import Integral, Real
from mor.operators import matrixOperator
from mor.backends import backendRegistry

class backendLyapunov:
    def __init__(self, backendName: str | None = None, **kwargs):
        self.localBackend, self.options = backendRegistry.get(backendName), kwargs

    def validateIterationOptions(self):
        maxIter = self.options.get('maxIter', 500)
        tol = self.options.get('tol', 1e-10)
        if not isinstance(maxIter, Integral) or isinstance(maxIter, bool) or maxIter <= 0:
            raise ValueError('maxIter must be a positive integer')
        if not isinstance(tol, Real) or isinstance(tol, bool) or not isfinite(tol) or tol < 0:
            raise ValueError('tol must be a finite non-negative real number')

    def validateColumnOption(self, name, value):
        if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')

    def prepareQ(self, B: matrixOperator, trans: bool = False):
        backend, bData = self.localBackend, B.toBackendData()
        dtype = getattr(bData, 'dtype', None)
        isFloating = getattr(dtype, 'is_floating_point', False)
        isComplex = getattr(dtype, 'is_complex', False)
        if dtype is not None and not isFloating and not isComplex:
            try:
                import numpy as np
                if not np.issubdtype(dtype, np.inexact):
                    bData = bData * 1.0
            except TypeError:
                bData = bData * 1.0
        if backend.array.ndim(bData) == 1: bData = backend.array.reshape(bData, (-1, 1))
        if trans: return backend.linalg.dot(backend.linalg.transpose(bData), bData)
        return backend.linalg.dot(bData, backend.linalg.transpose(bData))

    def filterStableShifts(self, shifts: Any, tol: float = 1e-14, selectPositiveImag: bool = False) -> Any:
        backend = self.localBackend
        realParts, absShifts = backend.array.real(shifts), backend.array.abs(shifts)
        mask = (realParts < 0) & (absShifts > tol) & (backend.array.abs(realParts) > tol)
        if selectPositiveImag: mask &= (backend.array.imag(shifts) >= 0)
        return shifts[mask]

    def compressFactor(self, Z: Any, n: int) -> Any:
        backend = self.localBackend
        if backend.array.size(Z) == 0:
            return backend.array.zeros((n, 1), dtype=getattr(Z, 'dtype', None))
        Q, R = backend.linalg.qr(Z, mode='reduced')
        U, S, _ = backend.decomposition.svdDense(R, fullMatrices=False)
        if backend.array.size(S) == 0:
            return backend.array.zeros((n, 1), dtype=getattr(Z, 'dtype', None))
        tol = S[0] * n * 2.22e-16
        mask = S > tol
        if not backend.array.any(mask):
            return backend.array.zeros((n, 1), dtype=getattr(Z, 'dtype', None))
        U_keep = U[:, mask]
        S_keep = S[mask]
        return backend.linalg.dot(Q, U_keep * S_keep)
