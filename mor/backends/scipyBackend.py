import numpy as np
from scipy import linalg
from scipy.sparse import issparse
from scipy.sparse.linalg import svds, norm as spnorm
from .backendsBase import backendBase
from .registry import registerBackend

@registerBackend('scipy', priority=50)
class scipyBackend(backendBase):
    class linalg(backendBase.linalg):
        @staticmethod
        def solve(A, b):
            return linalg.solve(A, b)

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
            return linalg.solve_continuous_lyapunov(A, Q)

        @staticmethod
        def solveGeneralizedContinuousLyapunov(A, E, Q):
            return linalg.solve_generalized_continuous_lyapunov(A, E, Q)

        @staticmethod
        def robustSqrtFactor(A, tol=None, name="Matrix"):
            import warnings
            if A.shape[0] != A.shape[1]: return A
            A = (A + A.T) / 2
            eigvals, eigvecs = np.linalg.eigh(A)
            maxEig = np.max(eigvals)
            if tol is None: tol = max(maxEig, 1.0) * 1e-12
            mask = eigvals > tol
            numTotal, numKeep = len(eigvals), np.sum(mask)
            if numTotal - numKeep > 0:
                warnings.warn(f"[{name}] {numTotal - numKeep}/{numTotal} eigenvalues were truncated (below tol={tol:.2e}). This may indicate a stiff or singular system.", RuntimeWarning)
            return eigvecs[:, mask] * np.sqrt(eigvals[mask])[np.newaxis, :]

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
        def isfinite(data):
            return np.isfinite(data)

        @staticmethod
        def array(data, dtype=None):
            return np.array(data, dtype=dtype)

        @staticmethod
        def isSparse(data):
            return issparse(data)

    class specialized(backendBase.specialized):
        @staticmethod
        def gramMatrixNorm(w, backend):
            return backend.linalg.norm(backend.linalg.dot(w.T, w), ord=2)

    @property
    def name(self):
        return 'scipy'

    @property
    def arrayType(self):
        return np.ndarray
