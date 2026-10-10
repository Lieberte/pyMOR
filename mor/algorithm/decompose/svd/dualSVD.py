from typing import Any
from .svd import svd
from mor.algorithm.registry import registerAlgorithm
from ..truncation.truncation import truncate

from mor.operators.operatorsBase import operatorBase

@registerAlgorithm('svd', 'dual')
class dualSVD(svd):
    def _unweighted(self, data, rank, tol):
        backend = self.localBackend
        if min(data.shape) == 0:
            return self._direct(data, rank, tol)
        scale = backend.array.max(backend.array.abs(data))
        if float(scale) == 0:
            return self._direct(data, rank, tol)
        scaled = data / scale
        wide = data.shape[0] < data.shape[1]
        adjoint = backend.linalg.transpose(backend.linalg.conj(scaled))
        gram = backend.linalg.dot(scaled, adjoint) if wide else backend.linalg.dot(adjoint, scaled)
        evals, vectors = backend.eigen.eigh(gram)
        indices = backend.array.argsort(-evals)
        evals, vectors = evals[indices], vectors[:, indices]
        positive = evals > 0
        values, vectors = evals[positive], vectors[:, positive]
        singular = backend.array.sqrt(values)
        retained = truncate(singular, rank=rank, tol=tol, backend=backend)
        if backend.name == 'torch':
            import torch
            eps = torch.finfo(evals.dtype).eps
        else:
            import numpy as np
            eps = np.finfo(evals.dtype).eps
        cutoff = float(evals[0]) * eps * gram.shape[0] * 10
        missing = ((rank is None and tol is None and len(values) < min(data.shape))
                   or (rank is not None and rank > len(values)))
        if missing or (retained and float(values[retained - 1]) <= cutoff):
            return self._direct(data, rank, tol)
        vectors, singular = vectors[:, :retained], singular[:retained]
        if wide:
            right = backend.linalg.dot(backend.linalg.transpose(backend.linalg.conj(vectors)), scaled) / singular[:, None]
            return vectors, singular * scale, right
        left = backend.linalg.dot(scaled, vectors) / singular
        return left, singular * scale, backend.linalg.transpose(backend.linalg.conj(vectors))

    def _direct(self, data, rank, tol):
        u, s, vh = self.localBackend.decomposition.svdDense(data, fullMatrices=False)
        retained = truncate(s, rank=rank, tol=tol, backend=self.localBackend)
        return u[:, :retained], s[:retained], vh[:retained]

    def _weighted(self, data, product, rank, tol):
        backend = self.localBackend
        if product.backendName != backend.name or product.shape != (data.shape[0], data.shape[0]):
            raise ValueError('product must match the snapshot rows and backend')
        if backend.name == 'scipy':
            import numpy as np
            from scipy import linalg
            from mor.backends.scipyBackend import _denseDecompositionData
            if product.isDiagonal:
                diagonal = np.asarray(product.diagonal())
                if not np.all(np.isfinite(diagonal)) or np.any(diagonal.real <= 0) or np.any(diagonal.imag != 0):
                    raise ValueError('product must be Hermitian positive definite')
                factor = np.sqrt(diagonal.real)
                data = _denseDecompositionData(data)
                u, s, vh = self._unweighted(factor[:, None] * data, rank, tol)
                return u / factor[:, None], s, vh
            weight = _denseDecompositionData(product.toBackendData())
            if not np.all(np.isfinite(weight)) or not np.allclose(weight, weight.conj().T):
                raise ValueError('product must be Hermitian positive definite')
            try:
                factor = linalg.cholesky(weight, lower=False)
            except linalg.LinAlgError as error:
                raise ValueError('product must be Hermitian positive definite') from error
            u, s, vh = self._unweighted(factor @ _denseDecompositionData(data), rank, tol)
            return linalg.solve_triangular(factor, u), s, vh
        if backend.name == 'torch':
            import torch
            weight = product.toBackendData()
            if product.isDiagonal:
                diagonal = product.diagonal()
                real = diagonal.real
                if not torch.isfinite(diagonal).all() or torch.any(real <= 0) or (diagonal.is_complex() and torch.any(diagonal.imag != 0)):
                    raise ValueError('product must be Hermitian positive definite')
                factor = real.sqrt()
                u, s, vh = self._unweighted(factor[:, None] * data, rank, tol)
                return u / factor[:, None], s, vh
            if backend.array.isSparse(weight):
                raise NotImplementedError('General sparse weighted Torch SVD requires an explicit dense product')
            if not torch.isfinite(weight).all() or not torch.allclose(weight, weight.mH):
                raise ValueError('product must be Hermitian positive definite')
            try:
                factor = torch.linalg.cholesky(weight).mH
            except RuntimeError as error:
                raise ValueError('product must be Hermitian positive definite') from error
            u, s, vh = self._unweighted(factor @ data, rank, tol)
            return torch.linalg.solve_triangular(factor, u, upper=True), s, vh
        raise NotImplementedError('Weighted SVD requires scipy or torch')

    def decompose(self, xOperator: operatorBase, rank: int | None = None, tol: float | None = None, product: Any | None = None) -> tuple[Any, Any, Any]:
        backend, S = self.localBackend, xOperator.toBackendData()
        if product is None:
            return self._unweighted(S, rank, tol)
        return self._weighted(S, product, rank, tol)
