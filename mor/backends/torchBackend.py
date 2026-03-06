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
            origDtype = A.dtype
            A_c, Q_c = A.to(torch.complex128), Q.to(torch.complex128)
            L, V = torch.linalg.eig(A_c)
            vInv = torch.linalg.inv(V)
            qTilde = vInv @ Q_c @ vInv.mH
            denom = L.unsqueeze(1) + L.conj().unsqueeze(0)
            eps = torch.finfo(torch.complex128).eps * 100
            denom = torch.where(torch.abs(denom) < eps, torch.tensor(eps, dtype=torch.complex128, device=A.device), denom)
            Y = -qTilde / denom
            X = V @ Y @ V.mH
            X = (X + X.mH) / 2
            return X.real.to(origDtype)

        @staticmethod
        def solveGeneralizedContinuousLyapunov(A, E, Q):
            origDtype = A.dtype
            A_c, E_c, Q_c = A.to(torch.complex128), E.to(torch.complex128), Q.to(torch.complex128)
            AE_inv = torch.linalg.solve(E_c, A_c)
            L, V = torch.linalg.eig(AE_inv)
            vInv = torch.linalg.inv(V)
            temp = torch.linalg.solve(E_c, Q_c)
            temp = torch.linalg.solve(E_c.mH, temp.mH).mH
            qTilde = vInv @ temp @ vInv.mH
            denom = L.unsqueeze(1) + L.conj().unsqueeze(0)
            eps = torch.finfo(torch.complex128).eps * 10
            denom = torch.where(torch.abs(denom) < eps, torch.tensor(eps, dtype=torch.complex128, device=A.device), denom)
            Y = -qTilde / denom
            X = V @ Y @ V.mH
            X = (X + X.mH) / 2
            return X.real.to(origDtype)

        @staticmethod
        def robustSqrtFactor(A, tol=None, name="Matrix"):
            if A.shape[0] != A.shape[1]: return A
            origDtype = A.dtype
            A_work = A.to(torch.float64)
            A_work = (A_work + A_work.mT) / 2
            eigvals, eigvecs = torch.linalg.eigh(A_work)
            if tol is None:
                maxEig = torch.max(torch.abs(eigvals))
                tol = maxEig * eigvals.numel() * torch.finfo(torch.float64).eps
            eigvals = torch.clamp(eigvals, min=0)
            mask = eigvals > tol
            if not torch.any(mask):
                return torch.zeros((A.shape[0], 1), dtype=origDtype, device=A.device)
            res = eigvecs[:, mask] * torch.sqrt(eigvals[mask]).unsqueeze(0)
            return res.to(origDtype)

        @staticmethod
        def balance(A):
            origDtype = A.dtype
            work = A.detach().abs().to(torch.float64)
            n = work.shape[0]
            d = torch.ones(n, device=A.device, dtype=torch.float64)
            converged = False
            for _ in range(100):
                last_d = d.clone()
                for i in range(n):
                    r = torch.sum(work[i, :]) - work[i, i]
                    c = torch.sum(work[:, i]) - work[i, i]
                    if r == 0 or c == 0: continue
                    g = r / 2.0
                    f = 1.0
                    s = r + c
                    while c < g:
                        f *= 2.0
                        c *= 4.0
                        s = r + c
                    g = r * 2.0
                    while c > g:
                        f /= 2.0
                        c /= 4.0
                        s = r + c
                    if (c + r) / f < 0.95 * s:
                        d[i] *= f
                        work[i, :] *= f
                        work[:, i] /= f
                if torch.allclose(d, last_d, rtol=1e-3):
                    converged = True
                    break
            return d.to(origDtype)

        @staticmethod
        def transpose(a):
            return a.mT

    class decomposition(backendBase.decomposition):
        @staticmethod
        def svdDense(A, fullMatrices=False):
            origDtype = A.dtype
            U, S, Vh = torch.linalg.svd(A.to(torch.float64), full_matrices=fullMatrices)
            return U.to(origDtype), S.to(origDtype), Vh.to(origDtype)

        @staticmethod
        def svdSparse(A, k, which='LM'):
            return torch.linalg.svd(A, full_matrices=False)

        @staticmethod
        def qrOrthogonalize(B, backend):
            Q, _ = torch.linalg.qr(B, mode='reduced')
            return Q

        @staticmethod
        def conj(a):
            return a.conj()

    class eigen(backendBase.eigen):
        @staticmethod
        def eigvalsGeneralized(A, B):
            return torch.linalg.eigvals(torch.linalg.solve(B, A))

        @staticmethod
        def eigvals(A):
            return torch.linalg.eigvals(A)

        @staticmethod
        def eigh(A):
            return torch.linalg.eigh(A)

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
        def exp(data):
            if not isinstance(data, torch.Tensor): return np.exp(data)
            return torch.exp(data)

        @staticmethod
        def isfinite(data):
            if not isinstance(data, torch.Tensor): return np.isfinite(data)
            return torch.isfinite(data)

        @staticmethod
        def array(data, dtype=None):
            dev = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
            return torch.tensor(data, dtype=dtype, device=dev)

        @staticmethod
        def randn(shape, dtype=None):
            dev = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
            return torch.randn(shape, dtype=dtype, device=dev)

        @staticmethod
        def any(data):
            if not isinstance(data, torch.Tensor): return any(data)
            return torch.any(data)

        @staticmethod
        def size(data):
            if not isinstance(data, torch.Tensor): return np.size(data)
            return data.numel()

        @staticmethod
        def isSparse(data):
            if not isinstance(data, torch.Tensor): return False
            return data.is_sparse

        @staticmethod
        def eyeSparse(n, dtype=None):
            # TODO: Implement native sparse eye for torch
            dev = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
            return torch.eye(n, dtype=dtype, device=dev).to_sparse()

        @staticmethod
        def copy(data):
            return data.clone() if isinstance(data, torch.Tensor) else data.copy()

        @staticmethod
        def ndim(data):
            return data.ndim

        @staticmethod
        def reshape(data, shape):
            return data.reshape(shape)

    @property
    def name(self):
        return 'torch'

    @property
    def arrayType(self):
        return torch.Tensor
