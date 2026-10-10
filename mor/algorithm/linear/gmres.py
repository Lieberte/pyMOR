from numbers import Integral, Real
import math
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from mor.algorithm.registry import registerAlgorithm
from .linear import linear


@registerAlgorithm('linear', 'gmres')
class gmresLinear(linear):
    def solve(self, A, rhs, E=None, shift=0., trans=False, **kwargs):
        if self.localBackend.name != 'scipy':
            raise NotImplementedError('GMRES currently requires the scipy backend')
        options = dict(self.options, **kwargs)
        tolerance, absolute = options.get('tol', 1e-10), options.get('atol', 0.)
        budget, restart = options.get('maxIter', 100), options.get('restart', 30)
        for name, value in [('tol', tolerance), ('atol', absolute)]:
            if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and non-negative')
        for name, value in [('maxIter', budget), ('restart', restart)]:
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        n = A.shape[0]
        if A.shape != (n, n) or (E is not None and E.shape != A.shape):
            raise ValueError('A must be square and E must match A')
        values = np.asarray(rhs)
        vector = values.ndim == 1
        if vector: values = values[:, None]
        if values.ndim != 2 or values.shape[0] != n or not np.all(np.isfinite(values)) or not np.isfinite(shift):
            raise ValueError('rhs and shift must be finite and rhs must match A')
        dtype = np.result_type(A.dtype, E.dtype if E is not None else A.dtype, values.dtype, shift, np.float64)
        def apply(x):
            return A.apply(x, trans=trans) + shift * (E.apply(x, trans=trans) if E is not None else x)
        operator = LinearOperator((n, n), matvec=apply, dtype=dtype)
        result = np.empty(values.shape, dtype=dtype)
        self.lastIterations, self.lastResiduals, self.lastConverged = [], [], False
        for column in range(values.shape[1]):
            history = []
            solution, info = gmres(operator, values[:, column], rtol=tolerance, atol=absolute,
                restart=int(restart), maxiter=int(budget), callback=history.append, callback_type='pr_norm')
            residual = float(np.linalg.norm(apply(solution) - values[:, column]))
            self.lastIterations.append(len(history))
            self.lastResiduals.append(residual)
            limit = max(absolute, tolerance * np.linalg.norm(values[:, column]))
            if info != 0 or not np.isfinite(residual) or residual > limit * (1 + 1e-6):
                raise ValueError(f'GMRES failed to converge for RHS {column}: info={info}, residual={residual:.3g}')
            result[:, column] = solution
        self.lastConverged = True
        return result[:, 0] if vector else result
