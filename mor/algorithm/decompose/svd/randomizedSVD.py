from typing import Any
from numbers import Integral
from .svd import svd
from mor.algorithm.registry import registerAlgorithm
from ..truncation.truncation import truncate

from mor.operators.operatorsBase import operatorBase

@registerAlgorithm('svd', 'randomized')
class randomizedSVD(svd):
    def decompose(self, xOperator: operatorBase, rank: int, tol: float | None = None, oversampling: int = 10, iterations: int = 2, randomState: int = 0) -> tuple[Any, Any, Any]:
        backend, S = self.localBackend, xOperator.toBackendData()
        if backend.name != 'scipy':
            raise NotImplementedError('Randomized SVD currently supports the scipy backend')
        for name, value in [('rank', rank), ('oversampling', oversampling), ('iterations', iterations), ('randomState', randomState)]:
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
                raise ValueError(f'{name} must be a non-negative integer')
        if rank == 0:
            import numpy as np
            return np.empty((S.shape[0], 0)), np.empty(0), np.empty((0, S.shape[1]))
        l = min(rank + oversampling, min(S.shape))
        U, s, Vh = backend.decomposition.randomizedSvd(S, nComponents=l, iterations=int(iterations), randomState=int(randomState))
        r = truncate(s, rank=rank, tol=tol, backend=backend)
        return U[:, :r], s[:r], Vh[:r, :]
