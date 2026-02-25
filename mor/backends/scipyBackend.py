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
        def qr(A, mode='reduced'):
            return linalg.qr(A, mode='economic' if mode == 'reduced' else mode)

        @staticmethod
        def norm(x, ord=None):
            if issparse(x):
                return spnorm(x, ord=ord)
            return linalg.norm(x, ord=ord)

        @staticmethod
        def dot(a, b):
            if issparse(a):
                return a @ b
            return np.dot(a, b)

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
            if B.shape[1] == 1:
                return B / linalg.norm(B)
            return linalg.qr(B, mode='economic')[0]

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
        def hstack(arrays):
            return np.hstack(arrays)

        @staticmethod
        def toNumpy(data):
            return np.asarray(data)

        @staticmethod
        def toArray(data):
            return np.asarray(data)

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

    class specialized(backendBase.specialized):
        @staticmethod
        def gramMatrixNorm(w, backend):
            return backend.linalg.norm(backend.linalg.dot(w.T, w), ord=2)

    class lyapunov(backendBase.lyapunov):
        @staticmethod
        def solveContinuous(a, q):
            return linalg.solve_continuous_lyapunov(a, q)

        @staticmethod
        def solveDiscrete(a, q):
            return linalg.solve_discrete_lyapunov(a, q)

        @staticmethod
        def solveContinuousGeneralized(a, e, q):
            a_tilde = linalg.solve(e, a)
            q_tilde = linalg.solve(e, linalg.solve(e, q.T).T)
            return linalg.solve_continuous_lyapunov(a_tilde, q_tilde)

        @staticmethod
        def solveDiscreteGeneralized(a, e, q):
            a_tilde = linalg.solve(e, a)
            q_tilde = linalg.solve(e, linalg.solve(e, q.T).T)
            return linalg.solve_discrete_lyapunov(a_tilde, q_tilde)

    @property
    def name(self):
        return 'scipy'

    @property
    def arrayType(self):
        return np.ndarray

    @property
    def supportsLyapunov(self):
        return True
