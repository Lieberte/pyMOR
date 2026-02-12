import numpy as np
from .backendsBase import backendBase
from .registry import registerBackend

@registerBackend('numpy', priority=10)
class numpyBackend(backendBase):
    class linalg(backendBase.linalg):
        @staticmethod
        def solve(A, b):
            return np.linalg.solve(A, b)

        @staticmethod
        def qr(A, mode='reduced'):
            return np.linalg.qr(A, mode=mode)

        @staticmethod
        def norm(x, ord=None):
            return np.linalg.norm(x, ord=ord)

        @staticmethod
        def dot(a, b):
            return np.dot(a, b)

    class decomposition(backendBase.decomposition):

        @staticmethod
        def svdDense(A, fullMatrices=False):
            return np.linalg.svd(A, full_matrices=fullMatrices)

        @staticmethod
        def svdSparse(A, k, which='LM'):
            from scipy.sparse.linalg import svds
            U, S, Vt = svds(A, k=k, which=which)
            idx = np.argsort(S)[::-1]
            return U[:, idx], S[idx], Vt[idx, :]

        @staticmethod
        def qrOrthogonalize(B, backend):
            if B.shape[1] == 1:
                return B / np.linalg.norm(B)
            Q, _ = np.linalg.qr(B, mode='reduced')
            return Q
    eigen = backendBase.eigen

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

    @property
    def name(self):
        return 'numpy'

    @property
    def arrayType(self):
        return np.ndarray
