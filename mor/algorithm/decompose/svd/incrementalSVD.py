from typing import Any

from .svd import svd
from ..truncation.truncation import truncate


class incrementalSVD(svd):
    def __init__(self, backendName: str | None = None, **kwargs):
        super().__init__(backendName, **kwargs)
        self.reset()

    def reset(self):
        self.U = self.S = self.Vh = None

    def decompose(self, xOperator: Any, rank: int | None = None, tol: float | None = None):
        backend, data = self.localBackend, xOperator.toBackendData()
        if self.U is not None and data.shape[0] != self.U.shape[0]:
            raise ValueError('incremental SVD batches must have the same row count; call reset for a new stream')
        if backend.name == 'scipy':
            from mor.backends.scipyBackend import _denseDecompositionData
            data = _denseDecompositionData(data)
        if self.U is None:
            U, S, Vh = backend.decomposition.svdDense(data, fullMatrices=False)
        else:
            adjoint = backend.linalg.transpose(backend.linalg.conj(self.U))
            projection = backend.linalg.dot(adjoint, data)
            residual = data - backend.linalg.dot(self.U, projection)
            correction = backend.linalg.dot(adjoint, residual)
            projection = projection + correction
            residual = residual - backend.linalg.dot(self.U, correction)
            Q, residualValues, _ = backend.decomposition.svdDense(residual, fullMatrices=False)
            import numpy as np
            epsilon = np.finfo(np.float32 if str(residualValues.dtype) in ('float32', 'torch.float32') else np.float64).eps
            scale = max(float(backend.array.max(backend.array.abs(projection))) if backend.array.size(projection) else 0.,
                        float(residualValues[0]) if len(residualValues) else 0.,
                        float(self.S[0]) if len(self.S) else 0.)
            mask = residualValues > epsilon * max(data.shape) * scale
            Q = Q[:, mask]
            top = backend.array.hstack([backend.array.diag(self.S), projection])
            bottom = backend.array.hstack([
                self._zeros(data, (Q.shape[1], len(self.S))),
                backend.linalg.dot(backend.linalg.transpose(backend.linalg.conj(Q)), residual),
            ])
            core = backend.array.vstack([top, bottom])
            coreU, S, coreVh = backend.decomposition.svdDense(core, fullMatrices=False)
            r = truncate(S, rank=rank, tol=tol, backend=backend)
            U = backend.linalg.dot(backend.array.hstack([self.U, Q]), coreU[:, :r])
            Vh = backend.array.hstack([
                backend.linalg.dot(coreVh[:r, :len(self.S)], self.Vh),
                coreVh[:r, len(self.S):],
            ])
        r = truncate(S, rank=rank, tol=tol, backend=backend)
        self.U, self.S, self.Vh = U[:, :r], S[:r], Vh[:r, :]
        return self.U, self.S, self.Vh

    def _zeros(self, data, shape):
        return data.new_zeros(shape) if hasattr(data, 'new_zeros') else self.localBackend.array.zeros(shape, dtype=data.dtype)
