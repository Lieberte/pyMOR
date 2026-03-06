from typing import Any
from mor.operators import matrixOperator
from mor.backends import backendRegistry

class backendLyapunov:
    def __init__(self, backendName: str | None = None, **kwargs):
        self.localBackend, self.options = backendRegistry.get(backendName), kwargs

    def prepareQ(self, B: matrixOperator, trans: bool = False):
        backend, bData = self.localBackend, B.data
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
