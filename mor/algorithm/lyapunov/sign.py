import numpy as np
from typing import Any
from mor.operators import matrixOperator
from mor.algorithm.registry import registerAlgorithm
from .hr import backendLyapunovAlgorithm

@registerAlgorithm('lyapunov', 'sign')
class signAlgorithm(backendLyapunovAlgorithm):
    # TODO: Implement robust numerical handling for stiff systems
    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator) -> Any:
        backend, n = self.localBackend, A.shape[0]
        maxIter, tol = self.options.get('maxIter', 80), self.options.get('tol', 1e-14)
        aj, zj = backend.array.copy(A.data), backend.array.copy(B.data)
        if backend.array.ndim(zj) == 1: zj = backend.array.reshape(zj, (-1, 1))
        for k in range(maxIter):
            _, logDet = backend.linalg.slogdet(aj)
            logGamma = logDet / n
            gamma = np.exp(logGamma) if (np.isfinite(logGamma) and abs(logGamma) < 30) else 1.0
            try: ajInv = backend.linalg.solve(aj, backend.array.eye(n, dtype=A.dtype))
            except Exception: break
            c1, c2 = backend.array.sqrt(gamma / 2.0), backend.array.sqrt(1.0 / (2.0 * gamma))
            zj = backend.array.hstack([c1 * zj, c2 * backend.linalg.dot(ajInv.T, zj)])
            ajNew = 0.5 * (gamma * aj + ajInv / gamma)
            diff = backend.linalg.norm(ajNew + backend.array.eye(n, dtype=A.dtype)) / n
            aj = ajNew
            if diff < tol: break
            if zj.shape[1] > 4 * n:
                q, r = backend.linalg.qr(zj, mode='reduced')
                u, s, vt = backend.decomposition.svdDense(r, fullMatrices=False)
                keep = s > (s[0] * n * 2.22e-16)
                zj = backend.linalg.dot(q, u[:, keep] * s[keep])
        zj = zj / backend.array.sqrt(2.0)
        q, r = backend.linalg.qr(zj, mode='reduced')
        u, s, vt = backend.decomposition.svdDense(r, fullMatrices=False)
        keep = s > (s[0] * n * 2.22e-16)
        return backend.linalg.dot(q, u[:, keep] * s[keep])
