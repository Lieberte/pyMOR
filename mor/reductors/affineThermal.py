import numpy as np

from mor.models.affineThermal import _array, affineThermalModel
from mor.operators import operatorBase


class affineThermalReductor:
    def reduce(self, mass, conductivity, boundaryMatrices, source, boundaryLoads, basis,
               referenceTemperature=0., interfaceNames=None):
        v = _array(basis, 'basis')
        if v.ndim != 2 or not v.shape[1] or not np.all(np.isfinite(v)):
            raise ValueError('basis must be a finite matrix with at least one column')
        if not np.allclose(v.T @ v, np.eye(v.shape[1]), atol=1e-9, rtol=1e-9):
            raise ValueError('basis columns must be orthonormal')
        n = v.shape[0]
        ones = np.ones((n, 1))
        if not np.allclose(v @ (v.T @ ones), ones, atol=1e-8):
            raise ValueError('basis must represent the constant temperature mode')

        def apply(operator, values):
            if operator.shape != (n, n):
                raise ValueError('full operators must match the basis rows')
            if isinstance(operator, operatorBase):
                result = operator.apply(values)
                return _array(operator.localBackend.array.toNumpy(result), 'operator application')
            return _array(operator @ values, 'operator application')

        source = _array(source, 'source')
        loads = _array(boundaryLoads, 'boundaryLoads')
        if source.shape != (n,) or loads.shape != (n, len(boundaryMatrices)):
            raise ValueError('full source and boundary load shapes must match the basis')
        mv, mo = apply(mass, v), apply(mass, ones).ravel()
        kv = apply(conductivity, v)
        boundaries = np.array([v.T @ apply(matrix, v) for matrix in boundaryMatrices])
        return affineThermalModel(v.T @ mv, v.T @ kv, boundaries, v.T @ source,
                                  v.T @ loads, loads.sum(axis=0), mo @ v, mo.sum(),
                                  source.sum(), referenceTemperature, interfaceNames)
