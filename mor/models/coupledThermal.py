from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy import linalg

from mor.assembly.coupling import addProjectedThermalInterface, projectThermalInterface, projectedThermalInterface
from .modelsBase import modelBase


def _realArray(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f'{name} must be real')
    result = np.array(value, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must contain finite values')
    return result


def _symmetric(value, name):
    result = _realArray(value, name)
    if result.ndim != 2 or result.shape[0] != result.shape[1] or not result.size:
        raise ValueError(f'{name} must be a nonempty square matrix')
    scale = max(float(np.linalg.norm(result)), np.finfo(float).tiny)
    if np.linalg.norm(result - result.T) > 1e-10 * scale:
        raise ValueError(f'{name} must be symmetric')
    return (result + result.T) / 2


@dataclass(frozen=True)
class reducedThermalComponent:
    name: str
    basis: np.ndarray
    offset: np.ndarray
    projection: np.ndarray
    capacity: np.ndarray
    conductivity: np.ndarray
    load: np.ndarray
    inputMatrix: np.ndarray
    fixedMask: np.ndarray
    energyWeights: np.ndarray
    energyOffset: float
    lossWeights: np.ndarray
    lossOffset: float
    sourcePower: float
    inputPowers: np.ndarray
    fixedCapacityWeights: np.ndarray
    fixedLossWeights: np.ndarray
    fixedLossOffset: float
    fixedSourcePower: float
    fixedInputPowers: np.ndarray

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('component name must be a nonempty string')
        for field in self.__dataclass_fields__:
            if field != 'name':
                value = _realArray(getattr(self, field), field)
                if value.ndim == 0:
                    value = float(value)
                else:
                    value.setflags(write=False)
                object.__setattr__(self, field, value)
        if self.basis.ndim != 2 or min(self.basis.shape) < 1:
            raise ValueError('basis must be a nonempty matrix')
        n, r = self.basis.shape
        m = self.inputMatrix.shape[1] if self.inputMatrix.ndim == 2 else -1
        shapes = {'offset': (n,), 'projection': (r, n), 'capacity': (r, r), 'conductivity': (r, r),
                  'load': (r,), 'inputMatrix': (r, m), 'fixedMask': (n,), 'energyWeights': (r,),
                  'lossWeights': (r,), 'inputPowers': (m,), 'fixedCapacityWeights': (r,),
                  'fixedLossWeights': (r,), 'fixedInputPowers': (m,)}
        for field, shape in shapes.items():
            if np.shape(getattr(self, field)) != shape:
                raise ValueError(f'{field} must have shape {shape}')
        linalg.cholesky(_symmetric(self.capacity, 'capacity'))
        k = _symmetric(self.conductivity, 'conductivity')
        if np.min(linalg.eigvalsh(k)) < -1e-10 * max(np.linalg.norm(k), np.finfo(float).tiny):
            raise ValueError('conductivity must be positive semidefinite')
        if not np.all((self.fixedMask == 0) | (self.fixedMask == 1)):
            raise ValueError('fixedMask must contain only zero and one')
        fixed = self.fixedMask.astype(bool)
        if not np.allclose(self.basis[fixed], 0., atol=1e-12):
            raise ValueError('fixed-temperature nodes must have zero basis rows')
        constant = 1. - self.fixedMask
        represented = self.basis @ linalg.lstsq(self.basis, constant)[0]
        if not np.allclose(represented, constant, atol=1e-10):
            raise ValueError('basis must preserve the free-node constant mode for energy conservation')

    @property
    def order(self):
        return self.basis.shape[1]

    @property
    def fullOrder(self):
        return self.basis.shape[0]

    @property
    def inputDim(self):
        return self.inputMatrix.shape[1]

    def project(self, temperature):
        value = _realArray(temperature, 'initial temperature')
        if value.ndim == 0:
            value = np.full(self.fullOrder, value)
        if value.shape != (self.fullOrder,):
            raise ValueError(f'initial temperature must have shape ({self.fullOrder},)')
        fixed = self.fixedMask.astype(bool)
        if not np.allclose(value[fixed], self.offset[fixed], rtol=0., atol=1e-10):
            raise ValueError('initial temperature violates fixed-temperature conditions')
        return self.projection @ (value - self.offset)

    def reconstruct(self, coordinates):
        values = _realArray(coordinates, 'coordinates')
        if values.ndim < 1 or values.shape[-1] != self.order:
            raise ValueError('coordinate last dimension must match component order')
        return values @ self.basis.T + self.offset


@dataclass(frozen=True)
class coupledThermalResult:
    model: object
    times: np.ndarray
    coordinates: np.ndarray
    diagnostics: dict

    def temperatures(self, componentName):
        component = self.model.components[componentName]
        return component.reconstruct(self.coordinates[:, self.model.componentSlices[componentName]])

    def interfacePowers(self):
        return {name: self.model.interfaceHeatFlow(name, self.coordinates).sum(axis=-1)
                for name in self.model.interfaces}


class coupledThermalModel(modelBase):
    def __init__(self, components, interfaces=()):
        super().__init__('scipy')
        values = list(components)
        self.components = {component.name: component for component in values}
        if not values or len(self.components) != len(values):
            raise ValueError('components must be nonempty and have unique names')
        self.componentSlices = {}
        self.inputSlices = {}
        offset = inputOffset = 0
        for component in values:
            self.componentSlices[component.name] = slice(offset, offset + component.order)
            self.inputSlices[component.name] = slice(inputOffset, inputOffset + component.inputDim)
            offset += component.order
            inputOffset += component.inputDim
        self.M = linalg.block_diag(*(component.capacity for component in values))
        self.K = linalg.block_diag(*(component.conductivity for component in values))
        self.Q = np.concatenate([component.load for component in values])
        self.B = np.zeros((offset, inputOffset))
        for component in values:
            self.B[self.componentSlices[component.name], self.inputSlices[component.name]] = component.inputMatrix
        self.interfaces = {}
        for interface in interfaces:
            if interface.name in self.interfaces:
                raise ValueError('interface names must be unique')
            if interface.primaryComponent not in self.components or interface.secondaryComponent not in self.components:
                raise ValueError('interface refers to an unknown component')
            primary, secondary = self.components[interface.primaryComponent], self.components[interface.secondaryComponent]
            projected = interface if isinstance(interface, projectedThermalInterface) else projectThermalInterface(interface, primary, secondary)
            self._validateInterface(projected, primary, secondary)
            self.interfaces[interface.name] = projected
            addProjectedThermalInterface(self.K, self.Q, projected,
                                         self.componentSlices[primary.name], self.componentSlices[secondary.name])
        for matrix in (self.M, self.K, self.Q, self.B):
            matrix.setflags(write=False)
        self._fixedInterfaces = tuple(name for name, interface in self.interfaces.items()
                                      if np.any(interface.primaryFixedWeights != interface.secondaryFixedWeights))
        self._stepFactor = None
        self._stepSize = None
        self.lastSolveInfo = {'factorizationCount': 0, 'method': 'implicitEuler'}

    @staticmethod
    def _validateInterface(interface, primary, secondary):
        if not isinstance(interface.name, str) or not interface.name:
            raise ValueError('interface name must be a nonempty string')
        g = _realArray(interface.conductance, 'conductance')
        if g.ndim != 1 or not len(g) or np.any(g < 0) or primary.name == secondary.name:
            raise ValueError('invalid projected interface conductance or endpoints')
        q = len(g)
        shapes = {'primaryTrace': (q, primary.order), 'secondaryTrace': (q, secondary.order),
                  'temperatureOffset': (q,), 'primaryFixedWeights': (q,), 'secondaryFixedWeights': (q,)}
        for name, shape in shapes.items():
            if _realArray(getattr(interface, name), name).shape != shape:
                raise ValueError(f'projected {name} must have shape {shape}')
        for trace, weights, component in ((interface.primaryTrace, interface.primaryFixedWeights, primary),
                                          (interface.secondaryTrace, interface.secondaryFixedWeights, secondary)):
            if np.any(weights < 0) or np.any(weights > 1):
                raise ValueError('projected fixed weights must be between zero and one')
            constantCoordinates = linalg.lstsq(component.basis, 1. - component.fixedMask)[0]
            if not np.allclose(trace @ constantCoordinates + weights, 1., atol=1e-10):
                raise ValueError('projected interface must preserve constant temperatures')

    @property
    def order(self):
        return len(self.Q)

    @property
    def fullOrder(self):
        return sum(component.fullOrder for component in self.components.values())

    def _state(self, value):
        state = _realArray(value, 'coordinates')
        if state.shape != (self.order,):
            raise ValueError(f'coordinates must have shape ({self.order},)')
        return state

    def _inputs(self, inputs):
        inputs = {} if inputs is None else inputs
        if not hasattr(inputs, 'keys') or set(inputs) - self.components.keys():
            raise ValueError('inputs must map known component names to input vectors')
        result = np.zeros(self.B.shape[1])
        for name, value in inputs.items():
            values = _realArray(value, f'{name} inputs').reshape(-1)
            target = self.inputSlices[name]
            if values.size != target.stop - target.start:
                raise ValueError(f'{name} input size does not match its input operator')
            result[target] = values
        return result

    def projectTemperatures(self, temperatures):
        if set(temperatures) != self.components.keys():
            raise ValueError('initial temperatures must specify every component exactly once')
        return np.concatenate([component.project(temperatures[name]) for name, component in self.components.items()])

    def reconstruct(self, coordinates):
        values = _realArray(coordinates, 'coordinates')
        if values.ndim < 1 or values.shape[-1] != self.order:
            raise ValueError('coordinate last dimension must match model order')
        return {name: component.reconstruct(values[..., self.componentSlices[name]])
                for name, component in self.components.items()}

    def interfaceHeatFlow(self, name, coordinates):
        interface = self.interfaces[name]
        values = _realArray(coordinates, 'coordinates')
        if values.ndim < 1 or values.shape[-1] != self.order:
            raise ValueError('coordinate last dimension must match model order')
        primary = values[..., self.componentSlices[interface.primaryComponent]] @ interface.primaryTrace.T
        secondary = values[..., self.componentSlices[interface.secondaryComponent]] @ interface.secondaryTrace.T
        return (primary - secondary + interface.temperatureOffset) * interface.conductance

    def solveSteady(self, inputs=None):
        try:
            factor = linalg.cho_factor(self.K, lower=True)
        except linalg.LinAlgError as error:
            raise ValueError('steady system is singular or not positive definite; provide a thermal sink') from error
        return linalg.cho_solve(factor, self.Q + self.B @ self._inputs(inputs))

    def implicitEulerStep(self, previousCoordinates, timeStep, inputs=None):
        dt = float(timeStep)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('timeStep must be positive and finite')
        previous = self._state(previousCoordinates)
        if self._stepSize != dt:
            self._stepFactor = linalg.cho_factor(self.M + dt * self.K, lower=True)
            self._stepSize = dt
            self.lastSolveInfo['factorizationCount'] += 1
        return linalg.cho_solve(self._stepFactor, self.M @ previous + dt * (self.Q + self.B @ self._inputs(inputs)))

    def energyBalance(self, coordinates, previousCoordinates, timeStep, inputs=None):
        dt = float(timeStep)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('timeStep must be positive and finite')
        current, previous = self._state(coordinates), self._state(previousCoordinates)
        rate, u = (current - previous) / dt, self._inputs(inputs)
        energy = previousEnergy = external = fixedPower = 0.
        for name, component in self.components.items():
            state, old, derivative = current[self.componentSlices[name]], previous[self.componentSlices[name]], rate[self.componentSlices[name]]
            localInput = u[self.inputSlices[name]]
            energy += component.energyWeights @ state + component.energyOffset
            previousEnergy += component.energyWeights @ old + component.energyOffset
            external += component.sourcePower + component.inputPowers @ localInput - component.lossWeights @ state - component.lossOffset
            fixedPower += (component.fixedCapacityWeights @ derivative + component.fixedLossWeights @ state
                           + component.fixedLossOffset - component.fixedSourcePower - component.fixedInputPowers @ localInput)
        for name in self._fixedInterfaces:
            interface = self.interfaces[name]
            flux = self.interfaceHeatFlow(name, current)
            fixedPower += (interface.primaryFixedWeights - interface.secondaryFixedWeights) @ flux
        rateW = (energy - previousEnergy) / dt
        return {'energyJ': float(energy), 'energyRateW': float(rateW), 'externalPowerW': float(external),
                'fixedTemperaturePowerW': float(fixedPower), 'residualW': float(rateW - external - fixedPower)}

    def solveTransient(self, initialTemperatures, times, inputs=None):
        times = _realArray(times, 'times')
        if times.ndim != 1 or not len(times) or np.any(np.diff(times) <= 0):
            raise ValueError('times must be a nonempty strictly increasing vector')
        coordinates = np.empty((len(times), self.order))
        coordinates[0] = self.projectTemperatures(initialTemperatures)
        diagnostics = {name: [] for name in ('energyJ', 'energyRateW', 'externalPowerW', 'fixedTemperaturePowerW', 'residualW')}
        for index, dt in enumerate(np.diff(times), start=1):
            currentInputs = inputs(float(times[index])) if callable(inputs) else inputs
            coordinates[index] = self.implicitEulerStep(coordinates[index-1], dt, currentInputs)
            balance = self.energyBalance(coordinates[index], coordinates[index-1], dt, currentInputs)
            for name, value in balance.items():
                diagnostics[name].append(value)
        return coupledThermalResult(self, times, coordinates, {key: np.asarray(value) for key, value in diagnostics.items()})

    def toNumpy(self):
        return {'M': self.M.copy(), 'K': self.K.copy(), 'Q': self.Q.copy(), 'B': self.B.copy()}

    def save(self, path):
        arrays, manifest = {}, {'formatVersion': 'coupled-thermal-rom-v1', 'components': [], 'interfaces': []}
        for category, values in (('components', self.components.values()), ('interfaces', self.interfaces.values())):
            for index, value in enumerate(values):
                record = {}
                for name in value.__dataclass_fields__:
                    item = getattr(value, name)
                    if isinstance(item, str):
                        record[name] = item
                    else:
                        key = f'{category}_{index}_{name}'
                        arrays[key] = item
                        record[name] = {'array': key}
                manifest[category].append(record)
        arrays['manifest'] = np.asarray(json.dumps(manifest))
        with Path(path).open('wb') as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            manifest = json.loads(str(archive['manifest']))
            if manifest.get('formatVersion') != 'coupled-thermal-rom-v1':
                raise ValueError('unsupported coupled thermal ROM format')
            def restore(record, constructor):
                return constructor(**{name: archive[value['array']].copy() if isinstance(value, dict) else value
                                      for name, value in record.items()})
            components = [restore(record, reducedThermalComponent) for record in manifest['components']]
            interfaces = [restore(record, projectedThermalInterface) for record in manifest['interfaces']]
        return cls(components, interfaces)
