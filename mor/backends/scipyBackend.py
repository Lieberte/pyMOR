import numpy as np
from scipy import linalg
from scipy.sparse import issparse, eye as speye, coo_matrix, csr_matrix
from scipy.sparse.linalg import svds, norm as spnorm, spsolve
from .backendsBase import backendBase
from .registry import registerBackend

_MAX_DENSE_DECOMPOSITION_BYTES = 64 * 1024 ** 2


def _denseDecompositionData(data):
    if not issparse(data):
        return data
    dtype = np.result_type(data.dtype, np.float32)
    denseBytes = int(data.shape[0]) * int(data.shape[1]) * dtype.itemsize
    if denseBytes > _MAX_DENSE_DECOMPOSITION_BYTES:
        raise ValueError(
            'Sparse full decomposition exceeds the dense conversion limit; '
            'use a truncated sparse SVD or explicitly supply dense data.'
        )
    return data.toarray()


@registerBackend('scipy', priority=50)
class scipyBackend(backendBase):
    class linalg(backendBase.linalg):
        @staticmethod
        def applyDirichlet(A, rhs, indices, values):
            dtype = np.result_type(A.dtype, rhs.dtype, values.dtype, np.float64)
            matrix = csr_matrix(A, dtype=dtype)
            result = np.asarray(rhs, dtype=dtype).reshape(-1).copy()
            result -= np.asarray(matrix[:, indices] @ values).reshape(-1)
            constrained = matrix.tolil()
            constrained[:, indices] = 0
            constrained[indices, :] = 0
            constrained[indices, indices] = 1
            result[indices] = values
            return constrained.tocsr(), result.reshape(-1, 1)

        @staticmethod
        def factorize(data):
            import warnings
            from scipy.sparse.linalg import splu
            data = data.astype(np.result_type(data.dtype, np.float64), copy=False)
            if issparse(data):
                factor = splu(data.tocsc())
                def solve(rhs):
                    if np.iscomplexobj(rhs) and not np.iscomplexobj(data):
                        return factor.solve(rhs.real) + 1j * factor.solve(rhs.imag)
                    return factor.solve(np.asarray(rhs, dtype=data.dtype))
                solve.factorBytes = factor.nnz * (data.dtype.itemsize + 8) + data.shape[0] * 16
                return solve
            with warnings.catch_warnings():
                warnings.simplefilter('error', linalg.LinAlgWarning)
                factor = linalg.lu_factor(data)
            def solve(rhs):
                return linalg.lu_solve(factor, rhs)
            solve.factorBytes = factor[0].nbytes + factor[1].nbytes
            return solve

        @staticmethod
        def solve(A, b):
            return spsolve(A, b) if issparse(A) else linalg.solve(A, b)

        @staticmethod
        def lstsq(A, b):
            return linalg.lstsq(A, b)[0]

        @staticmethod
        def solveTriangular(A, b, lower=False):
            return linalg.solve_triangular(A, b, lower=lower)

        @staticmethod
        def qr(A, mode='reduced'):
            return linalg.qr(_denseDecompositionData(A), mode='economic' if mode == 'reduced' else mode)

        @staticmethod
        def schur(A, output='real'):
            return linalg.schur(A, output=output)

        @staticmethod
        def norm(x, ord=None):
            if issparse(x):
                return spnorm(x, ord=ord)
            return linalg.norm(np.ravel(x, order='K')) if ord is None else linalg.norm(x, ord=ord)

        @staticmethod
        def dot(a, b):
            return a @ b if issparse(a) or issparse(b) else np.dot(a, b)

        @staticmethod
        def det(a):
            return linalg.det(a)

        @staticmethod
        def slogdet(a):
            return np.linalg.slogdet(a)

        @staticmethod
        def solveContinuousLyapunov(A, Q):
            return linalg.solve_continuous_lyapunov(A, -Q)

        @staticmethod
        def solveGeneralizedContinuousLyapunov(A, E, Q):
            if hasattr(linalg, 'solve_generalized_continuous_lyapunov'):
                return linalg.solve_generalized_continuous_lyapunov(A, E, -Q)
            def solveCorrection(rhs):
                eInvA = linalg.solve(E, A)
                eInvRhs = linalg.solve(E, rhs)
                transformedRhs = linalg.solve(E, eInvRhs.T).T
                return linalg.solve_continuous_lyapunov(eInvA, -transformedRhs)

            X = solveCorrection(Q)
            qNorm = max(linalg.norm(Q, ord=2), np.finfo(float).tiny)
            for _ in range(2):
                residual = A @ X @ E.T + E @ X @ A.T + Q
                if linalg.norm(residual, ord=2) <= qNorm * 1e-10:
                    break
                X += solveCorrection(residual)
                X = (X + X.T) / 2
            return X

        @staticmethod
        def robustSqrtFactor(A, tol=None, name="Matrix"):
            import warnings
            if A.shape[0] != A.shape[1]: return A
            A = (A + A.T) / 2
            eigvals, eigvecs = np.linalg.eigh(A)
            maxEig = np.max(np.abs(eigvals))
            automaticTol = maxEig * eigvals.size * np.finfo(eigvals.dtype).eps
            if tol is None:
                tol = automaticTol
            if tol < 0:
                raise ValueError("tol must be non-negative")

            negativeTolerance = 10 * automaticTol
            minEig = np.min(eigvals)
            positiveEigvals = np.clip(eigvals, a_min=0, a_max=None)
            mask = positiveEigvals > tol
            discardedPositive = positiveEigvals[(positiveEigvals > 0) & ~mask]
            positiveTrace = np.sum(positiveEigvals)
            discardedFraction = np.sum(discardedPositive) / positiveTrace if positiveTrace > 0 else 0.0

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
            if not np.any(mask):
                return np.zeros((A.shape[0], 1), dtype=A.dtype)
            return eigvecs[:, mask] * np.sqrt(positiveEigvals[mask])[np.newaxis, :]

        @staticmethod
        def balance(A):
            work = np.abs(A).astype(np.float64)
            n = work.shape[0]
            d = np.ones(n, dtype=np.float64)
            for _ in range(100):
                last_d = d.copy()
                for i in range(n):
                    r = np.sum(work[i, :]) - work[i, i]
                    c = np.sum(work[:, i]) - work[i, i]
                    if r == 0 or c == 0: continue
                    g, f, s = r / 2.0, 1.0, r + c
                    while c < g:
                        f, c = f * 2.0, c * 4.0
                        s = r + c
                    g = r * 2.0
                    while c > g:
                        f, c = f / 2.0, c / 4.0
                        s = r + c
                    if (c + r) / f < 0.95 * s:
                        d[i], work[i, :], work[:, i] = d[i] * f, work[i, :] * f, work[:, i] / f
                if np.allclose(d, last_d, rtol=1e-3): break
            return d.astype(A.dtype)

        @staticmethod
        def transpose(a):
            return a.T

        @staticmethod
        def conj(a):
            return a.conj()

    class decomposition(backendBase.decomposition):
        @staticmethod
        def svdDense(A, fullMatrices=False):
            return linalg.svd(_denseDecompositionData(A), full_matrices=fullMatrices)

        @staticmethod
        def randomizedSvd(A, nComponents=None, iterations=2, randomState=0, fullMatrices=False):
            from sklearn.utils.extmath import randomized_svd
            nComponents = min(A.shape) if nComponents is None else nComponents
            U, S, Vt = randomized_svd(A, n_components=nComponents, n_iter=iterations, random_state=randomState)
            return U, S, Vt

        @staticmethod
        def svdSparse(A, k, which='LM'):
            if A.dtype.kind not in 'fc':
                A = A.astype(np.float64)
            scale = 1.0
            if issparse(A):
                if 0 < k < min(A.shape) and A.count_nonzero() == 0:
                    return (np.eye(A.shape[0], k, dtype=A.dtype),
                            np.zeros(k, dtype=A.real.dtype),
                            np.eye(k, A.shape[1], dtype=A.dtype))
                scale = np.max(np.abs(A.data), initial=0.0)
                if scale > 0:
                    A = A / scale
            U, S, Vt = svds(A, k=k, which=which)
            S = S * scale
            idx = np.argsort(S)[::-1]
            return U[:, idx], S[idx], Vt[idx, :]

        @staticmethod
        def qrOrthogonalize(B, backend):
            # SciPy has no sparse QR API. Only bounded dense conversion is allowed.
            B = _denseDecompositionData(B)
            if B.shape[1] == 1:
                length = linalg.norm(B)
                if length != 0:
                    return B / length
            return linalg.qr(B, mode='economic')[0]

    class eigen(backendBase.eigen):
        @staticmethod
        def eig(A):
            return linalg.eig(A)

        @staticmethod
        def eigvalsGeneralized(A, B):
            return linalg.eigvals(A, B)

        @staticmethod
        def eigvals(A):
            return linalg.eigvals(A)

        @staticmethod
        def eigh(A):
            return linalg.eigh(_denseDecompositionData(A))

    class array(backendBase.array):
        @staticmethod
        def toArrayLike(data, like):
            return scipyBackend.array.toArray(data)

        @staticmethod
        def resultType(*values):
            return np.result_type(*values)

        @staticmethod
        def astype(data, dtype):
            return data.astype(dtype, copy=False)

        @staticmethod
        def isDiagonal(data):
            if data.ndim != 2 or data.shape[0] != data.shape[1]:
                return False
            if issparse(data):
                entries = data.tocoo(copy=True)
                entries.sum_duplicates()
                return not np.any((entries.row != entries.col) & (entries.data != 0))
            return np.count_nonzero(data) == np.count_nonzero(np.diagonal(data))

        @staticmethod
        def diagonal(data):
            return data.diagonal() if issparse(data) else np.diagonal(data)

        @staticmethod
        def sparseFromTriplets(values, rows, columns, shape, like=None):
            return coo_matrix((values, (rows, columns)), shape=shape).tocsr()

        @staticmethod
        def zeros(shape, dtype=None):
            return np.zeros(shape, dtype=dtype)

        @staticmethod
        def eye(n, dtype=None):
            return np.eye(n, dtype=dtype)

        @staticmethod
        def eyeLike(a):
            return speye(a.shape[0], dtype=a.dtype, format='csr') if issparse(a) else np.eye(a.shape[0], dtype=a.dtype)

        @staticmethod
        def hstack(arrays):
            return np.hstack(arrays)

        @staticmethod
        def vstack(arrays):
            return np.vstack(arrays)

        @staticmethod
        def toNumpy(data):
            return data.toarray() if issparse(data) else np.asarray(data)

        @staticmethod
        def toArray(data):
            if issparse(data):
                return data
            if type(data).__module__.startswith('torch'):
                detached = data.detach()
                if str(detached.layout) != 'torch.strided':
                    coo = detached.to_sparse_coo().coalesce()
                    indices = coo.indices().cpu().numpy()
                    values = coo.values().resolve_conj().resolve_neg().cpu().numpy()
                    return coo_matrix((values, (indices[0], indices[1])), shape=coo.shape).tocsr()
                return detached.resolve_conj().resolve_neg().cpu().numpy()
            return np.asarray(data)

        @staticmethod
        def diag(v, k=0):
            return np.diag(v, k=k)

        @staticmethod
        def trace(a):
            return np.trace(a)

        @staticmethod
        def max(data, axis=None):
            return np.max(data, axis=axis)

        @staticmethod
        def abs(data):
            return np.abs(data)

        @staticmethod
        def sqrt(data):
            return np.sqrt(data)

        @staticmethod
        def exp(data):
            return np.exp(data)

        @staticmethod
        def isfinite(data):
            return np.isfinite(data)

        @staticmethod
        def where(condition):
            return np.where(condition)

        @staticmethod
        def array(data, dtype=None):
            return np.array(data, dtype=dtype)

        @staticmethod
        def argsort(data):
            return np.argsort(data)

        @staticmethod
        def randn(shape, dtype=None):
            return np.random.randn(*shape).astype(dtype) if dtype else np.random.randn(*shape)

        @staticmethod
        def any(data):
            return np.any(data)

        @staticmethod
        def size(data):
            return np.size(data)

        @staticmethod
        def isSparse(data):
            return issparse(data)

        @staticmethod
        def eyeSparse(n, dtype=None):
            return speye(n, dtype=dtype)

    @property
    def name(self):
        return 'scipy'

    @property
    def arrayType(self):
        return np.ndarray
