import numpy as np
from .backendsBase import backendBase
from .registry import registerBackend

from scipy import linalg
from scipy.sparse.linalg import svds


@registerBackend('scipy', priority=50)
class scipyBackend(backendBase):
    class linalg(backendBase.linalg):
        @staticmethod
        def solve(A, b):
            return linalg.solve(A, b)

        @staticmethod
        def qr(A, mode='reduced'):
            if mode == 'reduced':
                mode = 'economic'
            return linalg.qr(A, mode=mode)

        @staticmethod
        def norm(x, ord=None):
            return linalg.norm(x, ord=ord)

        @staticmethod
        def dot(a, b):
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
            import scipy.sparse as sp
            if sp.issparse(B):
                from scipy.sparse.linalg import qr as sparse_qr
                Q, _ = sparse_qr(B)
                if sp.issparse(Q):
                    Q = Q.toarray()
                return Q
            if B.shape[1] == 1:
                return B / linalg.norm(B)
            n, k = B.shape
            if n > 1000 and k < n // 10:
                Q, _ = linalg.qr(B, mode='economic', pivoting=False, overwrite_a=False, check_finite=False)
            else:
                Q, _ = linalg.qr(B, mode='economic', check_finite=False)
            return Q

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

    specialized = backendBase.specialized

    class lyapunov(backendBase.lyapunov):

        @staticmethod
        def solveContinuous(a, q):
            return linalg.solve_continuous_lyapunov(a, q)

        @staticmethod
        def solveDiscrete(a, q):
            return linalg.solve_discrete_lyapunov(a, q)

        @staticmethod
        def solveContinuousGeneralized(a, e, q):
            a = np.asarray(a)
            e = np.asarray(e)
            q = np.asarray(q)
            aTilde = linalg.solve(e, a)
            qTilde = -linalg.solve(e, linalg.solve(e.T, q.T).T)
            return linalg.solve_continuous_lyapunov(aTilde, qTilde)

        @staticmethod
        def solveDiscreteGeneralized(a, e, q):
            a = np.asarray(a)
            e = np.asarray(e)
            q = np.asarray(q)
            aTilde = linalg.solve(e, a)
            qTilde = linalg.solve(e, linalg.solve(e.T, q.T).T)
            return linalg.solve_discrete_lyapunov(aTilde, qTilde)

    @property
    def name(self):
        return 'scipy'

    @property
    def arrayType(self):
        return np.ndarray

    @property
    def supportsLyapunov(self):
        return True
