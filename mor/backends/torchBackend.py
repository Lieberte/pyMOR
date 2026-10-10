import torch
import numpy as np
from typing import Tuple, Any
from .backendsBase import backendBase
from .registry import registerBackend
from mor.runtime import resolveDeviceName

@registerBackend('torch', priority=70)
class torchBackend(backendBase):
    def __init__(self, device: str = 'auto'):
        self._device = torch.device(resolveDeviceName(device))
        self.array = torchArrayForDevice(self._device)

    class linalg(backendBase.linalg):
        LinAlgError = torch.linalg.LinAlgError

        @staticmethod
        def applyDirichlet(A, rhs, indices, values):
            index = torch.as_tensor(indices, dtype=torch.int64, device=A.device)
            prescribed = torch.as_tensor(values, device=A.device)
            dtype = torch.promote_types(torch.promote_types(A.dtype, rhs.dtype), prescribed.dtype)
            if not (dtype.is_floating_point or dtype.is_complex):
                dtype = torch.float64
            matrix = A.to(dtype=dtype)
            result = rhs.to(device=A.device, dtype=dtype).reshape(-1).clone()
            prescribed = prescribed.to(dtype=dtype)
            fixed = torch.zeros(A.shape[0], dtype=dtype, device=A.device)
            fixed[index] = prescribed
            result -= torchBackend.linalg.dot(matrix, fixed)
            mask = torch.zeros(A.shape[0], dtype=torch.bool, device=A.device)
            mask[index] = True
            if matrix.layout != torch.strided:
                coo = matrix.to_sparse_coo().coalesce()
                coordinates = coo.indices()
                keep = ~mask[coordinates[0]] & ~mask[coordinates[1]]
                rows = torch.cat((coordinates[0, keep], index))
                columns = torch.cat((coordinates[1, keep], index))
                entries = torch.cat((coo.values()[keep], torch.ones(len(index), dtype=dtype, device=A.device)))
                constrained = torchBackend.array.sparseFromTriplets(entries, rows, columns, A.shape, like=A)
            else:
                constrained = matrix.clone()
                constrained[:, index] = 0
                constrained[index, :] = 0
                constrained[index, index] = 1
            result[index] = prescribed
            return constrained, result.reshape(-1, 1)

        @staticmethod
        def factorize(data):
            if data.layout != torch.strided:
                raise NotImplementedError('Torch reusable LU requires a dense tensor')
            if not (data.is_floating_point() or data.is_complex()):
                data = data.to(torch.float64)
            factor, pivots = torch.linalg.lu_factor(data)
            def solve(rhs):
                vector = rhs.ndim == 1
                values = rhs.unsqueeze(-1) if vector else rhs
                if values.is_complex() and not factor.is_complex():
                    result = (torch.linalg.lu_solve(factor, pivots, values.real.to(factor.dtype))
                              + 1j * torch.linalg.lu_solve(factor, pivots, values.imag.to(factor.dtype)))
                else:
                    result = torch.linalg.lu_solve(factor, pivots, values.to(factor.dtype))
                return result.squeeze(-1) if vector else result
            solve.factorBytes = factor.numel() * factor.element_size() + pivots.numel() * pivots.element_size()
            return solve

        @staticmethod
        def solve(A, b):
            if A.layout != torch.strided:
                raise NotImplementedError('Torch direct solve requires a dense tensor; use scipy for sparse direct solves')
            return torch.linalg.solve(A, b)

        @staticmethod
        def lstsq(A, b):
            return torch.linalg.lstsq(A, b).solution

        @staticmethod
        def solveTriangular(A, b, lower=False):
            return torch.linalg.solve_triangular(A, b, upper=not lower)

        @staticmethod
        def qr(A, mode='reduced'):
            if A.layout != torch.strided:
                raise NotImplementedError('Torch QR requires a dense tensor; use scipy for bounded sparse QR conversion')
            return torch.linalg.qr(A, mode=mode)

        @staticmethod
        def schur(A, output='real'):
            raise NotImplementedError('Torch has no native Schur decomposition; use the scipy backend')

        @staticmethod
        def norm(x, ord=None):
            return torch.linalg.norm(x, ord=ord)

        @staticmethod
        def dot(a, b):
            if a.layout != torch.strided:
                coo = a.to_sparse_coo().coalesce()
                if b.ndim == 1:
                    return torch.sparse.mm(coo, b.unsqueeze(1)).squeeze(1)
                return torch.sparse.mm(coo, b)
            if b.layout != torch.strided:
                if a.ndim == 1:
                    return torch.sparse.mm(b.transpose(0, 1).to_sparse_coo().coalesce(), a.unsqueeze(1)).squeeze(1)
                return torch.sparse.mm(b.transpose(0, 1).to_sparse_coo().coalesce(), a.mT).mT
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
            import warnings
            if A.shape[0] != A.shape[1]: return A
            origDtype = A.dtype
            A_work = A.to(torch.float64)
            A_work = (A_work + A_work.mT) / 2
            eigvals, eigvecs = torch.linalg.eigh(A_work)
            maxEig = torch.max(torch.abs(eigvals)).item()
            automaticTol = maxEig * eigvals.numel() * torch.finfo(torch.float64).eps
            if tol is None:
                tol = automaticTol
            elif isinstance(tol, torch.Tensor):
                tol = tol.detach().cpu().item()
            if tol < 0:
                raise ValueError("tol must be non-negative")

            negativeTolerance = 10 * automaticTol
            minEig = torch.min(eigvals).item()
            positiveEigvals = torch.clamp(eigvals, min=0)
            mask = positiveEigvals > tol
            discardedPositive = positiveEigvals[(positiveEigvals > 0) & ~mask]
            positiveTrace = torch.sum(positiveEigvals).item()
            discardedFraction = (
                torch.sum(discardedPositive).item() / positiveTrace if positiveTrace > 0 else 0.0
            )

            issues = []
            if minEig < -negativeTolerance:
                issues.append(
                    f"matrix is not positive semidefinite (minimum eigenvalue={minEig:.2e}, "
                    f"numerical tolerance={negativeTolerance:.2e})"
                )
            if discardedFraction > 1e-8:
                issues.append(
                    f"tol={tol:.2e} discarded {discardedFraction:.2e} of the positive spectral weight"
                )
            if issues:
                warnings.warn(f"[{name}] " + "; ".join(issues), RuntimeWarning)
            if not torch.any(mask):
                return torch.zeros((A.shape[0], 1), dtype=origDtype, device=A.device)
            res = eigvecs[:, mask] * torch.sqrt(positiveEigvals[mask]).unsqueeze(0)
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
            return a.transpose(0, 1)

    class decomposition(backendBase.decomposition):
        @staticmethod
        def svdDense(A, fullMatrices=False):
            if A.layout != torch.strided:
                raise NotImplementedError('Torch dense SVD requires an explicit dense tensor')
            origDtype = A.dtype
            if A.is_complex():
                workDtype = torch.complex128
                singularDtype = torch.float32 if origDtype == torch.complex64 else torch.float64
            else:
                workDtype = torch.float64
                singularDtype = origDtype
            U, S, Vh = torch.linalg.svd(A.to(workDtype), full_matrices=fullMatrices)
            return U.to(origDtype), S.to(singularDtype), Vh.to(origDtype)

        @staticmethod
        def svdSparse(A, k, which='LM'):
            raise NotImplementedError('Torch truncated sparse SVD is not supported; use the scipy backend')

        @staticmethod
        def qrOrthogonalize(B, backend):
            Q, _ = torchBackend.linalg.qr(B, mode='reduced')
            return Q

        @staticmethod
        def conj(a):
            return a.conj()

    class eigen(backendBase.eigen):
        @staticmethod
        def eig(A):
            return torch.linalg.eig(A)

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
        def toArrayLike(data, like):
            if isinstance(data, torch.Tensor):
                return data
            return torchBackend.array.toArray(data).to(device=like.device)

        @staticmethod
        def resultType(*values):
            dtype = values[0]
            for value in values[1:]:
                dtype = (torch.promote_types(dtype, value) if isinstance(value, torch.dtype)
                         else torch.result_type(torch.empty((), dtype=dtype), value))
            return dtype

        @staticmethod
        def astype(data, dtype):
            return data.to(dtype=dtype)

        @staticmethod
        def isDiagonal(data):
            if data.ndim != 2 or data.shape[0] != data.shape[1]:
                return False
            if data.layout != torch.strided:
                coo = data.to_sparse_coo().coalesce()
                indices = coo.indices()
                return not bool(torch.any((indices[0] != indices[1]) & (coo.values() != 0)))
            return int(torch.count_nonzero(data)) == int(torch.count_nonzero(data.diagonal()))

        @staticmethod
        def diagonal(data):
            if data.layout == torch.strided:
                return data.diagonal()
            coo = data.to_sparse_coo().coalesce()
            indices = coo.indices()
            mask = indices[0] == indices[1]
            result = torch.zeros(min(data.shape), dtype=data.dtype, device=data.device)
            return result.index_add(0, indices[0, mask], coo.values()[mask])

        @staticmethod
        def sparseFromTriplets(values, rows, columns, shape, like=None):
            device = like.device if like is not None else (values.device if isinstance(values, torch.Tensor) else 'cpu')
            entries = torch.as_tensor(values, device=device)
            indices = torch.stack((torch.as_tensor(rows, dtype=torch.int64, device=device),
                                   torch.as_tensor(columns, dtype=torch.int64, device=device)))
            return torch.sparse_coo_tensor(indices, entries, size=shape, device=device).coalesce()

        @staticmethod
        def imag(data):
            if not isinstance(data, torch.Tensor):
                return getattr(data, 'imag', 0.0)
            return data.imag if data.is_complex() else torch.zeros_like(data)

        @staticmethod
        def zeros(shape, dtype=None):
            return torch.zeros(shape, dtype=dtype)

        @staticmethod
        def eye(n, dtype=None):
            return torch.eye(n, dtype=dtype)

        @staticmethod
        def eyeLike(a):
            if a.layout != torch.strided:
                indices = torch.arange(a.shape[0], device=a.device)
                return torchBackend.array.sparseFromTriplets(
                    torch.ones(a.shape[0], dtype=a.dtype, device=a.device), indices, indices, a.shape, like=a)
            return torch.eye(a.shape[0], dtype=a.dtype, device=a.device)

        @staticmethod
        def hstack(arrays):
            return torch.hstack(arrays)

        @staticmethod
        def vstack(arrays):
            return torch.vstack(arrays)

        @staticmethod
        def toNumpy(data):
            data = data.detach()
            if data.layout != torch.strided:
                data = data.to_dense()
            return data.resolve_conj().resolve_neg().cpu().numpy()

        @staticmethod
        def toArray(data):
            if isinstance(data, torch.Tensor):
                return data
            from scipy.sparse import issparse
            if issparse(data):
                coo = data.tocoo()
                return torchBackend.array.sparseFromTriplets(coo.data, coo.row, coo.col, coo.shape)
            return torch.as_tensor(data)

        @staticmethod
        def diag(v, k=0):
            return torch.diag(v, diagonal=k)

        @staticmethod
        def trace(a):
            return torch.trace(a)

        @staticmethod
        def max(data, axis=None):
            if not isinstance(data, torch.Tensor): return np.max(data, axis=axis)
            return torch.max(data) if axis is None else torch.max(data, dim=axis).values

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
        def where(condition):
            if not isinstance(condition, torch.Tensor): return np.where(condition)
            return torch.where(condition)

        @staticmethod
        def array(data, dtype=None):
            return torch.tensor(data, dtype=dtype)

        @staticmethod
        def randn(shape, dtype=None):
            return torch.randn(shape, dtype=dtype)

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
            return data.layout in (torch.sparse_coo, torch.sparse_csr, torch.sparse_csc,
                                   torch.sparse_bsr, torch.sparse_bsc)

        @staticmethod
        def iscomplexobj(data):
            return torch.is_complex(data) if isinstance(data, torch.Tensor) else np.iscomplexobj(data)

        @staticmethod
        def eyeSparse(n, dtype=None):
            indices = torch.arange(n)
            return torchBackend.array.sparseFromTriplets(torch.ones(n, dtype=dtype), indices, indices, (n, n))

        @staticmethod
        def copy(data):
            return data.clone() if isinstance(data, torch.Tensor) else data.copy()

        @staticmethod
        def ndim(data):
            return data.ndim

        @staticmethod
        def argsort(data):
            return torch.argsort(data)

        @staticmethod
        def reshape(data, shape):
            return data.reshape(shape)

    @property
    def name(self):
        return 'torch'

    @property
    def arrayType(self):
        return torch.Tensor

    @property
    def deviceName(self):
        return self._device.type


class torchArrayForDevice(torchBackend.array):
    def __init__(self, device):
        self._device = torch.device(device)

    def zeros(self, shape, dtype=None):
        return torch.zeros(shape, dtype=dtype, device=self._device)

    def eye(self, n, dtype=None):
        return torch.eye(n, dtype=dtype, device=self._device)

    def toArray(self, data):
        if isinstance(data, torch.Tensor):
            return data
        from scipy.sparse import issparse
        if issparse(data):
            return torchBackend.array.toArray(data).to(self._device)
        return torch.as_tensor(data, device=self._device)

    def array(self, data, dtype=None):
        return torch.tensor(data, dtype=dtype, device=self._device)

    def randn(self, shape, dtype=None):
        return torch.randn(shape, dtype=dtype, device=self._device)

    def eyeSparse(self, n, dtype=None):
        indices = torch.arange(n, device=self._device)
        return self.sparseFromTriplets(torch.ones(n, dtype=dtype, device=self._device), indices, indices, (n, n))

    def sparseFromTriplets(self, values, rows, columns, shape, like=None):
        if like is None:
            like = torch.empty(0, device=self._device)
        return super().sparseFromTriplets(values, rows, columns, shape, like=like)
