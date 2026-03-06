import numpy as np
from scipy import linalg
from scipy.sparse import issparse, eye as speye
from scipy.sparse.linalg import svds, norm as spnorm, spsolve
from .backendsBase import backendBase
from .registry import registerBackend

@registerBackend('scipy', priority=50)
class scipyBackend(backendBase):
    class linalg(backendBase.linalg):
        @staticmethod
        def solve(A, b):
            return spsolve(A, b) if issparse(A) else linalg.solve(A, b)

        @staticmethod
        def solveTriangular(A, b, lower=False):
            return linalg.solve_triangular(A, b, lower=lower)

        @staticmethod
        def qr(A, mode='reduced'):
            return linalg.qr(A, mode='economic' if mode == 'reduced' else mode)

        @staticmethod
        def schur(A, output='real'):
            return linalg.schur(A, output=output)

        @staticmethod
        def norm(x, ord=None):
            return spnorm(x, ord=ord) if issparse(x) else linalg.norm(x, ord=ord)

        @staticmethod
        def dot(a, b):
            return a @ b if issparse(a) else np.dot(a, b)

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
            return linalg.solve_generalized_continuous_lyapunov(A, E, -Q)

        @staticmethod
        def robustSqrtFactor(A, tol=None, name="Matrix"):
            import warnings
            if A.shape[0] != A.shape[1]: return A
            A = (A + A.T) / 2
            eigvals, eigvecs = np.linalg.eigh(A)
            if tol is None:
                maxEig = np.max(np.abs(eigvals))
                tol = maxEig * eigvals.size * np.finfo(eigvals.dtype).eps
            eigvals = np.clip(eigvals, a_min=0, a_max=None)
            mask = eigvals > tol
            numTotal, numKeep = len(eigvals), np.sum(mask)
            if numTotal - numKeep > 0:
                warnings.warn(f"[{name}] {numTotal - numKeep}/{numTotal} eigenvalues were truncated (below tol={tol:.2e}). This may indicate a stiff or singular system.", RuntimeWarning)
            if not np.any(mask):
                return np.zeros((A.shape[0], 1), dtype=A.dtype)
            return eigvecs[:, mask] * np.sqrt(eigvals[mask])[np.newaxis, :]

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
            return linalg.svd(A, full_matrices=fullMatrices)

        @staticmethod
        def svdSparse(A, k, which='LM'):
            U, S, Vt = svds(A, k=k, which=which)
            idx = np.argsort(S)[::-1]
            return U[:, idx], S[idx], Vt[idx, :]

        @staticmethod
        def qrOrthogonalize(B, backend):
            if issparse(B):
                from scipy.sparse.linalg import qr as sparse_qr
                Q, _ = sparse_qr(B)
                return Q.toarray() if issparse(Q) else Q
            return B / linalg.norm(B) if B.shape[1] == 1 else linalg.qr(B, mode='economic')[0]

    class eigen(backendBase.eigen):
        @staticmethod
        def eigvalsGeneralized(A, B):
            return linalg.eigvals(A, B)

        @staticmethod
        def eigvals(A):
            return linalg.eigvals(A)

        @staticmethod
        def eigh(A):
            return linalg.eigh(A)

    class array(backendBase.array):
        @staticmethod
        def zeros(shape, dtype=None):
            return np.zeros(shape, dtype=dtype)

        @staticmethod
        def eye(n, dtype=None):
            return np.eye(n, dtype=dtype)

        @staticmethod
        def eyeLike(a):
            return np.eye(a.shape[0], dtype=a.dtype)

        @staticmethod
        def hstack(arrays):
            return np.hstack(arrays)

        @staticmethod
        def toNumpy(data):
            return np.asarray(data)

        @staticmethod
        def toArray(data):
            return np.asarray(data)

        @staticmethod
        def diag(v, k=0):
            return np.diag(v, k=k)

        @staticmethod
        def trace(a):
            return np.trace(a)

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
        def array(data, dtype=None):
            return np.array(data, dtype=dtype)

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
