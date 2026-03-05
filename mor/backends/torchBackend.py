import torch
import numpy as np
from typing import Tuple, Any
from .backendsBase import backendBase
from .registry import registerBackend

@registerBackend('torch', priority=70)
class torchBackend(backendBase):
    def __init__(self, device: str = 'cuda'):
        self._device = torch.device(device if torch.cuda.is_available() else 'cpu')

    class linalg(backendBase.linalg):
        @staticmethod
        def solve(A, b):
            return torch.linalg.solve(A, b)

        @staticmethod
        def solveTriangular(A, b, lower=False):
            return torch.linalg.solve_triangular(A, b, upper=not lower)

        @staticmethod
        def qr(A, mode='reduced'):
            return torch.linalg.qr(A, mode=mode)

        @staticmethod
        def schur(A, output='real'):
            return torch.linalg.schur(A)

        @staticmethod
        def norm(x, ord=None):
            return torch.linalg.norm(x, ord=ord)

        @staticmethod
        def dot(a, b):
            return torch.matmul(a, b)

        @staticmethod
        def det(a):
            return torch.linalg.det(a)

        @staticmethod
        def slogdet(a):
            res = torch.linalg.slogdet(a)
            return res.sign, res.logabsdet

        @staticmethod
        def solveContinuousLyapunov(A, Q):
            L, V = torch.linalg.eig(A)
            vInv = torch.linalg.inv(V)
            qTilde = vInv @ Q.to(torch.complex128) @ vInv.mT
            denom = L.unsqueeze(1) + L.unsqueeze(0)
            Y = qTilde / denom
            return (V @ Y @ V.mT).real.to(A.dtype)

        @staticmethod
        def solveGeneralizedContinuousLyapunov(A, E, Q):
            # TODO: Implement blocked generalized Lyapunov solver for Torch
            raise NotImplementedError("Generalized Lyapunov solver for Torch not yet implemented.")

        @staticmethod
        def robustSqrtFactor(A, tol=None, name="Matrix"):
            if A.shape[0] != A.shape[1]: return A
            A = (A + A.mT) / 2
            eigvals, eigvecs = torch.linalg.eigh(A)
            maxEig = torch.max(eigvals)
            if tol is None: tol = torch.maximum(maxEig, torch.tensor(1.0, device=A.device)) * 1e-12
            mask = eigvals > tol
            return eigvecs[:, mask] * torch.sqrt(eigvals[mask]).unsqueeze(0)

        @staticmethod
        def transpose(a):
            return a.mT

    class decomposition(backendBase.decomposition):
        @staticmethod
        def svdDense(A, fullMatrices=False):
            return torch.linalg.svd(A, full_matrices=fullMatrices)

        @staticmethod
        def svdSparse(A, k, which='LM'):
            return torch.linalg.svd(A, full_matrices=False)

        @staticmethod
        def qrOrthogonalize(B, backend):
            Q, _ = torch.linalg.qr(B, mode='reduced')
            return Q

    class eigen(backendBase.eigen):
        @staticmethod
        def eigvalsGeneralized(A, B):
            return torch.linalg.eigvals(torch.linalg.solve(B, A))

        @staticmethod
        def eigvals(A):
            return torch.linalg.eigvals(A)

    class array(backendBase.array):
        @staticmethod
        def zeros(shape, dtype=None):
            dev = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
            return torch.zeros(shape, dtype=dtype, device=dev)

        @staticmethod
        def eye(n, dtype=None):
            dev = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
            return torch.eye(n, dtype=dtype, device=dev)

        @staticmethod
        def eyeLike(a):
            return torch.eye(a.shape[0], dtype=a.dtype, device=a.device)

        @staticmethod
        def hstack(arrays):
            return torch.hstack(arrays)

        @staticmethod
        def toNumpy(data):
            return data.detach().cpu().numpy()

        @staticmethod
        def toArray(data):
            if isinstance(data, torch.Tensor): return data
            dev = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
            return torch.as_tensor(data, device=dev)

        @staticmethod
        def diag(v, k=0):
            return torch.diag(v, diagonal=k)

        @staticmethod
        def trace(a):
            return torch.trace(a)

        @staticmethod
        def abs(data):
            if not isinstance(data, torch.Tensor): return abs(data)
            return torch.abs(data)

        @staticmethod
        def sqrt(data):
            if not isinstance(data, torch.Tensor): return np.sqrt(data)
            return torch.sqrt(data)

        @staticmethod
        def isfinite(data):
            if not isinstance(data, torch.Tensor): return np.isfinite(data)
            return torch.isfinite(data)

        @staticmethod
        def array(data, dtype=None):
            dev = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
            return torch.tensor(data, dtype=dtype, device=dev)

        @staticmethod
        def isSparse(data):
            return data.is_sparse

        @staticmethod
        def copy(data):
            return data.clone()

        @staticmethod
        def ndim(data):
            return data.ndim

        @staticmethod
        def reshape(data, shape):
            return data.reshape(shape)

    class specialized(backendBase.specialized):
        @staticmethod
        def gramMatrixNorm(w, backend):
            return backend.linalg.norm(backend.linalg.dot(w.T, w), ord=2)

    @property
    def name(self):
        return 'torch'

    @property
    def arrayType(self):
        return torch.Tensor

# TODO: Optimize numerical stability and precision for stiff systems in Torch backend
