import numpy as np
from scipy import linalg

from mor.models.coupledThermal import _realArray, coupledThermalModel, reducedThermalComponent
from mor.operators import matrixOperator, operatorBase
from .pod import podReductor, podResult


class thermalCouplingReductor:
    def __init__(self, globalBackendName=None):
        if globalBackendName not in (None, 'scipy'):
            raise NotImplementedError('coupled thermal ROM assembly and solve use the scipy backend')

    def reduceComponent(self, name, model, *, snapshots=None, rank=None, basis=None, inputOperator=None, offset=None):
        if model.isNonlinear:
            raise NotImplementedError('coupled thermal ROM currently requires temperature-independent operators')
        if rank is not None and (isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or rank < 1):
            raise ValueError('rank must be a positive integer')
        n = model.order
        fixed = np.zeros(n)
        fixed[model.dirichletNodes] = 1.
        free = np.flatnonzero(fixed == 0)
        if not free.size:
            raise ValueError('a reduced component requires at least one free node')
        if snapshots is not None and basis is not None:
            raise ValueError('provide snapshots or basis, not both')
        if isinstance(basis, podResult):
            if offset is None and basis.snapshotMean is not None:
                offset = basis.V.localBackend.array.toNumpy(basis.snapshotMean).reshape(-1)
            basis = basis.V
        if offset is None:
            offset = np.full(n, model.referenceTemperature)
        else:
            offset = np.broadcast_to(_realArray(offset, 'offset'), (n,)).copy()
        offset[model.dirichletNodes] = model.dirichletValues
        constant = np.ones((len(free), 1)) / np.sqrt(len(free))
        if snapshots is not None:
            data = snapshots.toNumpy() if isinstance(snapshots, operatorBase) else snapshots
            data = _realArray(data, 'snapshots')
            if data.ndim != 2 or data.shape[0] != n or not data.shape[1]:
                raise ValueError('snapshots must have shape (fullOrder, snapshotCount)')
            if rank is None:
                raise ValueError('rank is required with snapshots and includes the constant mode')
            if rank > len(free):
                raise ValueError('rank exceeds the number of free nodes')
            centered = data[free] - offset[free, None]
            centered -= constant @ (constant.T @ centered)
            if rank > 1 and np.any(centered):
                pod = podReductor('scipy').reduce(matrixOperator(centered, 'scipy'), rank=min(rank - 1, centered.shape[1]))
                candidates = np.hstack((constant, pod.V.toNumpy()))
            else:
                candidates = constant
        elif basis is not None:
            values = basis.toNumpy() if isinstance(basis, operatorBase) else basis
            values = _realArray(values, 'basis')
            if values.ndim != 2 or values.shape[0] != n or not values.shape[1]:
                raise ValueError('basis must have shape (fullOrder, reducedOrder)')
            candidates = np.hstack((constant, values[free]))
        else:
            raise ValueError('a component requires snapshots or an explicit basis')
        residual = candidates[:, 1:] - constant @ (constant.T @ candidates[:, 1:])
        if residual.shape[1]:
            u, singularValues, _ = linalg.svd(residual, full_matrices=False)
            tolerance = max(candidates.shape) * np.finfo(float).eps * max(1., np.linalg.norm(candidates))
            keep = singularValues > tolerance
            freeBasis = np.hstack((constant, u[:, keep]))
        else:
            freeBasis = constant
        if rank is not None:
            freeBasis = freeBasis[:, :rank]
        V = np.zeros((n, freeBasis.shape[1]))
        V[free] = freeBasis

        def apply(operator, values):
            result = operator.apply(values)
            return _realArray(operator.localBackend.array.toNumpy(result), 'operator application')

        mv, kv = apply(model.M, V), apply(model.K, V)
        mo, ko = apply(model.M, offset[:, None]).ravel(), apply(model.K, offset[:, None]).ravel()
        q = _realArray(model.Q.toNumpy(), 'load').reshape(-1)
        if inputOperator is None:
            fullInput = np.zeros((n, 0))
        else:
            if not isinstance(inputOperator, operatorBase) or inputOperator.shape[0] != n:
                raise ValueError('inputOperator must be an operator with fullOrder rows')
            if inputOperator.backendName != model.backendName:
                raise ValueError('inputOperator backend must match the thermal model')
            fullInput = _realArray(inputOperator.toNumpy(), 'inputOperator')
            if fullInput.ndim == 1:
                fullInput = fullInput[:, None]
            if fullInput.ndim != 2:
                raise ValueError('inputOperator must be one- or two-dimensional')
        mr, kr = V.T @ mv, V.T @ kv
        projection = linalg.solve(mr, mv.T, assume_a='pos')
        return reducedThermalComponent(
            name, V, offset, projection, mr, kr, V.T @ (q-ko), V.T @ fullInput, fixed,
            mv.sum(axis=0), mo.sum(), kv.sum(axis=0), ko.sum(), q.sum(), fullInput.sum(axis=0),
            fixed @ mv, fixed @ kv, fixed @ ko, fixed @ q, fixed @ fullInput,
        )

    def assemble(self, components, interfaces=()):
        return coupledThermalModel(components, interfaces)
