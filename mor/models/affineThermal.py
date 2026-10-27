from dataclasses import dataclass

import numpy as np
from scipy import linalg


def _array(value, name):
    original = np.asarray(value)
    if np.iscomplexobj(original):
        raise ValueError(f'{name} must be real')
    result = np.array(original, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be finite')
    return result


@dataclass
class affineThermalResult:
    coordinates: np.ndarray
    meanTemperature: np.ndarray
    interfaceTemperatures: np.ndarray
    interfacePowers: np.ndarray
    stepInterfacePowers: np.ndarray
    energyResidualW: np.ndarray
    factorizationCount: int


class affineThermalModel:
    def __init__(self, M, K, boundaryMatrices, source, boundaryLoads, conductances,
                 capacityRow, totalCapacity, sourcePower, referenceTemperature=0., interfaceNames=None):
        self.M, self.K = _array(M, 'M'), _array(K, 'K')
        self.boundaryMatrices = _array(boundaryMatrices, 'boundaryMatrices')
        self.source = _array(source, 'source')
        self.boundaryLoads = _array(boundaryLoads, 'boundaryLoads')
        self.conductances = _array(conductances, 'conductances')
        self.capacityRow = _array(capacityRow, 'capacityRow')
        self.totalCapacity = float(totalCapacity)
        self.sourcePower = float(sourcePower)
        self.referenceTemperature = float(referenceTemperature)
        self.order = self.M.shape[0] if self.M.ndim == 2 else 0
        r, p = self.order, self.conductances.size
        shapes = ((self.M, (r, r)), (self.K, (r, r)), (self.boundaryMatrices, (p, r, r)),
                  (self.source, (r,)), (self.boundaryLoads, (r, p)),
                  (self.conductances, (p,)), (self.capacityRow, (r,)))
        if r == 0 or p == 0 or any(value.shape != shape for value, shape in shapes):
            raise ValueError('inconsistent reduced thermal array shapes')
        names = tuple(f'interface_{i}' for i in range(p)) if interfaceNames is None else tuple(interfaceNames)
        if len(names) != p or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != p:
            raise ValueError('interfaceNames must be unique nonempty strings, one per interface')
        self.interfaceNames = names
        scalars = [self.totalCapacity, self.sourcePower, self.referenceTemperature]
        if not np.all(np.isfinite(scalars)) or self.totalCapacity <= 0 or np.any(self.conductances <= 0):
            raise ValueError('capacity and interface conductances must be positive and scalars finite')
        for name, matrix in [('M', self.M), ('K', self.K), *[
                (f'boundaryMatrices[{i}]', value) for i, value in enumerate(self.boundaryMatrices)]]:
            if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10):
                raise ValueError(f'{name} must be symmetric')
            smallest = linalg.eigvalsh(matrix)[0]
            if name == 'M' and smallest <= 0:
                raise ValueError('M must be positive definite')
            if name != 'M' and smallest < -1e-10 * max(1., linalg.norm(matrix, 2)):
                raise ValueError(f'{name} must be positive semidefinite')
        self._constant = linalg.solve(self.M, self.capacityRow, assume_a='pos')
        tolerance = 1e-8
        if not np.isclose(self.capacityRow @ self._constant, self.totalCapacity, rtol=tolerance):
            raise ValueError('basis must represent the constant temperature mode')
        if not np.allclose(self.K @ self._constant, 0., atol=tolerance * max(1., linalg.norm(self.K))):
            raise ValueError('bulk conduction must conserve energy; fixed-temperature boundaries are unsupported')
        for matrix, load in zip(self.boundaryMatrices, self.boundaryLoads.T):
            if not np.allclose(matrix @ self._constant, load, rtol=tolerance, atol=tolerance):
                raise ValueError('boundary matrix and load must share the same Robin discretization')
        if not np.allclose(self._constant @ self.boundaryLoads, self.conductances, rtol=tolerance, atol=tolerance):
            raise ValueError('interface conductances disagree with boundary loads')
        if not np.isclose(self._constant @ self.source, self.sourcePower, rtol=tolerance, atol=tolerance):
            raise ValueError('source power disagrees with its projected load')
        for array, _ in shapes:
            array.setflags(write=False)

    def solveTransient(self, dt, sourceScales, boundaryScales, referenceTemperatures, initialRise=0.):
        dt = float(dt)
        heat = _array(sourceScales, 'sourceScales')
        scales = _array(boundaryScales, 'boundaryScales')
        refs = _array(referenceTemperatures, 'referenceTemperatures')
        if heat.ndim != 1 or not len(heat):
            raise ValueError('sourceScales must contain one value per step')
        steps, p = len(heat), len(self.conductances)
        if not np.isfinite(dt) or dt <= 0 or scales.shape != (steps, p) or refs.shape != scales.shape:
            raise ValueError('dt must be positive and boundary histories must have shape (steps, interfaces)')
        if np.any(scales < 0):
            raise ValueError('boundaryScales must be nonnegative')
        initialRise = float(initialRise)
        if not np.isfinite(initialRise):
            raise ValueError('initialRise must be finite')
        refs -= self.referenceTemperature
        coordinates = np.empty((steps + 1, self.order))
        coordinates[0] = self._constant * initialRise
        powers = np.empty((steps, p))
        stepPowers = np.empty_like(powers)
        lastScales, factor, old = None, None, None
        factorizations = 0
        for i in range(steps):
            if lastScales is None or not np.array_equal(scales[i], lastScales):
                k = self.K + np.einsum('p,pij->ij', scales[i], self.boundaryMatrices)
                factor = linalg.cho_factor(self.M / dt + k * .5, lower=True)
                old = self.M / dt - k * .5
                lastScales = scales[i].copy()
                factorizations += 1
            rhs = old @ coordinates[i] + self.source * heat[i] + self.boundaryLoads @ (scales[i] * refs[i])
            coordinates[i + 1] = linalg.cho_solve(factor, rhs)
            endTrace = self.boundaryLoads.T @ coordinates[i + 1]
            averageTrace = self.boundaryLoads.T @ ((coordinates[i] + coordinates[i + 1]) * .5)
            powers[i] = scales[i] * (endTrace - self.conductances * refs[i])
            stepPowers[i] = scales[i] * (averageTrace - self.conductances * refs[i])
        energy = coordinates @ self.capacityRow
        residual = np.diff(energy) / dt + stepPowers.sum(axis=1) - self.sourcePower * heat
        temperatures = coordinates @ self.boundaryLoads / self.conductances + self.referenceTemperature
        mean = energy / self.totalCapacity + self.referenceTemperature
        return affineThermalResult(coordinates, mean, temperatures, powers, stepPowers, residual, factorizations)

    def save(self, path):
        np.savez_compressed(path, format=np.array('affine-thermal-rom-v1'), M=self.M, K=self.K,
                            boundaryMatrices=self.boundaryMatrices, source=self.source,
                            boundaryLoads=self.boundaryLoads, conductances=self.conductances,
                            capacityRow=self.capacityRow, totalCapacity=self.totalCapacity,
                            sourcePower=self.sourcePower, referenceTemperature=self.referenceTemperature,
                            interfaceNames=np.array(self.interfaceNames))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if str(data['format']) != 'affine-thermal-rom-v1':
                raise ValueError('unsupported affine thermal ROM format')
            names = ('M', 'K', 'boundaryMatrices', 'source', 'boundaryLoads', 'conductances',
                     'capacityRow', 'totalCapacity', 'sourcePower', 'referenceTemperature')
            return cls(**{name: data[name] for name in names},
                       interfaceNames=data['interfaceNames'].tolist() if 'interfaceNames' in data else None)
