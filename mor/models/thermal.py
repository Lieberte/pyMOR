from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, cg, gmres, spsolve

from mor.operators import applyDirichletConditions, thermalOperatorSet

from .modelsBase import modelBase


def _jsonReady(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonReady(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonReady(item) for item in value]
    return value


def _finiteArray(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must contain finite values')
    return array


@dataclass(frozen=True)
class tabulatedThermalProperty:
    temperatures: Any
    values: Any
    extrapolation: str = 'clamp'

    def __post_init__(self):
        temperatures = _finiteArray(self.temperatures, 'property temperatures').reshape(-1)
        values = _finiteArray(self.values, 'property values')
        if temperatures.size < 2:
            raise ValueError('tabulated property requires at least two temperatures')
        if np.any(np.diff(temperatures) <= 0.0):
            raise ValueError('property temperatures must be strictly increasing')
        if values.ndim == 0 or values.shape[0] != temperatures.size:
            raise ValueError('property values must have one entry per temperature')
        if self.extrapolation != 'clamp':
            raise ValueError("tabulated property extrapolation currently supports only 'clamp'")
        object.__setattr__(self, 'temperatures', temperatures)
        object.__setattr__(self, 'values', values.copy())

    @property
    def valueShape(self) -> tuple[int, ...]:
        return self.values.shape[1:]

    def evaluate(self, temperature: Any) -> Any:
        query = _finiteArray(temperature, 'temperature')
        flattenedQuery = query.reshape(-1)
        flattenedValues = self.values.reshape(self.temperatures.size, -1)
        interpolated = np.column_stack(
            [
                np.interp(flattenedQuery, self.temperatures, flattenedValues[:, component])
                for component in range(flattenedValues.shape[1])
            ]
        )
        result = interpolated.reshape(query.shape + self.valueShape)
        if query.ndim == 0:
            if not self.valueShape:
                return float(result)
            return result.reshape(self.valueShape)
        return result

    def toDict(self) -> dict[str, Any]:
        return {
            'type': 'table',
            'temperatures': self.temperatures.tolist(),
            'values': self.values.tolist(),
            'extrapolation': self.extrapolation,
        }

    @classmethod
    def fromDict(cls, data: Mapping[str, Any]) -> 'tabulatedThermalProperty':
        temperatures = data.get('temperatures', data.get('temperature'))
        values = data.get('values', data.get('value'))
        if temperatures is None or values is None:
            raise ValueError('tabulated property requires temperatures and values')
        return cls(
            temperatures=temperatures,
            values=values,
            extrapolation=data.get('extrapolation', 'clamp'),
        )


def _propertyValue(value: Any) -> Any:
    if isinstance(value, tabulatedThermalProperty):
        return value
    if isinstance(value, Mapping):
        return tabulatedThermalProperty.fromDict(value)
    return value


def _positiveScalarProperty(value: Any, name: str) -> float | tabulatedThermalProperty:
    normalized = _propertyValue(value)
    if isinstance(normalized, tabulatedThermalProperty):
        if normalized.valueShape:
            raise ValueError(f'{name} table values must be scalar')
        if np.any(normalized.values <= 0.0):
            raise ValueError(f'{name} values must be positive')
        return normalized
    scalar = float(normalized)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f'{name} must be positive and finite')
    return scalar


def _conductivityProperty(value: Any) -> Any:
    normalized = _propertyValue(value)
    values = normalized.values if isinstance(normalized, tabulatedThermalProperty) else _finiteArray(normalized, 'conductivity')
    if values.ndim == 0:
        if float(values) <= 0.0:
            raise ValueError('conductivity must be positive')
        return normalized if isinstance(normalized, tabulatedThermalProperty) else float(values)
    if isinstance(normalized, tabulatedThermalProperty) and values.ndim == 1:
        if np.any(values <= 0.0):
            raise ValueError('conductivity values must be positive')
        return normalized
    tensors = values if isinstance(normalized, tabulatedThermalProperty) else values[None, :, :]
    if tensors.ndim != 3 or tensors.shape[1:] != (3, 3):
        raise ValueError('conductivity must be a scalar, 3x3 tensor, or temperature table of either')
    symmetric = 0.5 * (tensors + np.swapaxes(tensors, 1, 2))
    if not np.allclose(tensors, symmetric):
        raise ValueError('anisotropic conductivity must be symmetric')
    if np.any(np.linalg.eigvalsh(symmetric) <= 0.0):
        raise ValueError('anisotropic conductivity must be positive definite')
    if isinstance(normalized, tabulatedThermalProperty):
        return tabulatedThermalProperty(
            normalized.temperatures,
            symmetric,
            extrapolation=normalized.extrapolation,
        )
    return symmetric[0].copy()


def _evaluateProperty(value: Any, temperature: Any) -> Any:
    if isinstance(value, tabulatedThermalProperty):
        return value.evaluate(temperature)
    query = np.asarray(temperature, dtype=float)
    propertyArray = np.asarray(value, dtype=float)
    if query.ndim == 0:
        return value
    if propertyArray.ndim == 0:
        return np.full(query.shape, float(propertyArray), dtype=float)
    return np.broadcast_to(propertyArray, query.shape + propertyArray.shape).copy()


@dataclass(frozen=True)
class thermalMaterial:
    name: str
    conductivity: Any
    density: Any
    specificHeat: Any

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('material name must be a non-empty string')
        object.__setattr__(self, 'name', self.name.strip())
        object.__setattr__(self, 'conductivity', _conductivityProperty(self.conductivity))
        object.__setattr__(self, 'density', _positiveScalarProperty(self.density, 'density'))
        object.__setattr__(self, 'specificHeat', _positiveScalarProperty(self.specificHeat, 'specificHeat'))

    @property
    def isTemperatureDependent(self) -> bool:
        return any(
            isinstance(value, tabulatedThermalProperty)
            for value in (self.conductivity, self.density, self.specificHeat)
        )

    def conductivityAt(self, temperature: Any) -> Any:
        return _evaluateProperty(self.conductivity, temperature)

    def densityAt(self, temperature: Any) -> Any:
        return _evaluateProperty(self.density, temperature)

    def specificHeatAt(self, temperature: Any) -> Any:
        return _evaluateProperty(self.specificHeat, temperature)

    def toDict(self) -> dict[str, Any]:
        return {
            'conductivity': self.conductivity.toDict()
            if isinstance(self.conductivity, tabulatedThermalProperty)
            else _jsonReady(self.conductivity),
            'density': self.density.toDict()
            if isinstance(self.density, tabulatedThermalProperty)
            else self.density,
            'specificHeat': self.specificHeat.toDict()
            if isinstance(self.specificHeat, tabulatedThermalProperty)
            else self.specificHeat,
        }

    @classmethod
    def fromDict(cls, name: str, data: Mapping[str, Any]) -> 'thermalMaterial':
        return cls(
            name=name,
            conductivity=data['conductivity'],
            density=data['density'],
            specificHeat=data['specificHeat'],
        )


class thermalMaterialLibrary:
    def __init__(self, materials: Mapping[str, thermalMaterial | Mapping[str, Any]]):
        resolved: dict[str, thermalMaterial] = {}
        for name, value in materials.items():
            material = value if isinstance(value, thermalMaterial) else thermalMaterial.fromDict(name, value)
            if material.name != name:
                raise ValueError(f'material key {name!r} does not match material name {material.name!r}')
            if name in resolved:
                raise ValueError(f'duplicate material name: {name!r}')
            resolved[name] = material
        if not resolved:
            raise ValueError('at least one thermal material is required')
        self._materials = resolved

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._materials)

    @property
    def isTemperatureDependent(self) -> bool:
        return any(material.isTemperatureDependent for material in self._materials.values())

    def __contains__(self, name: str) -> bool:
        return name in self._materials

    def __getitem__(self, name: str) -> thermalMaterial:
        try:
            return self._materials[name]
        except KeyError as exc:
            raise KeyError(f'unknown thermal material: {name!r}') from exc

    def resolveRegionProperties(self, regionMaterials: Mapping[str, str]) -> dict[str, dict[str, Any]]:
        if not regionMaterials:
            raise ValueError('regionMaterials cannot be empty')
        conductivity: dict[str, Any] = {}
        density: dict[str, float] = {}
        specificHeat: dict[str, float] = {}
        for regionName, materialName in regionMaterials.items():
            material = self[materialName]
            conductivity[regionName] = material.conductivity
            density[regionName] = material.density
            specificHeat[regionName] = material.specificHeat
        return {
            'conductivity': conductivity,
            'density': density,
            'specificHeat': specificHeat,
        }

    def toDict(self) -> dict[str, dict[str, Any]]:
        return {name: material.toDict() for name, material in self._materials.items()}

    @classmethod
    def fromDict(cls, data: Mapping[str, Mapping[str, Any]]) -> 'thermalMaterialLibrary':
        return cls(data)


@dataclass(frozen=True)
class thermalBoundaryField:
    values: Any
    location: str = 'node'

    def __post_init__(self):
        if self.location not in ('uniform', 'face', 'node', 'globalNode'):
            raise ValueError("boundary field location must be 'uniform', 'face', 'node', or 'globalNode'")
        values = _finiteArray(self.values, 'boundary field values')
        if self.location == 'uniform' and values.ndim != 0:
            raise ValueError('uniform boundary field requires a scalar value')
        object.__setattr__(self, 'values', float(values) if values.ndim == 0 else values.copy())

    def toDict(self) -> dict[str, Any]:
        return {'location': self.location, 'values': _jsonReady(self.values)}

    @classmethod
    def fromDict(cls, data: Mapping[str, Any]) -> 'thermalBoundaryField':
        return cls(values=data['values'], location=data.get('location', 'node'))


def _boundaryFieldValue(value: Any, name: str) -> Any:
    if isinstance(value, thermalBoundaryField):
        return value
    if isinstance(value, Mapping) and 'location' in value:
        return thermalBoundaryField.fromDict(value)
    array = _finiteArray(value, name)
    return float(array) if array.ndim == 0 else array.copy()


@dataclass(frozen=True)
class thermalConvectionBoundary:
    coefficient: Any
    ambientTemperature: Any

    def __post_init__(self):
        coefficient = _boundaryFieldValue(self.coefficient, 'convection coefficient')
        ambient = _boundaryFieldValue(self.ambientTemperature, 'ambientTemperature')
        coefficientValues = coefficient.values if isinstance(coefficient, thermalBoundaryField) else coefficient
        if np.any(np.asarray(coefficientValues) < 0.0):
            raise ValueError('convection coefficient must be non-negative')
        object.__setattr__(self, 'coefficient', coefficient)
        object.__setattr__(self, 'ambientTemperature', ambient)

    def toDict(self) -> dict[str, Any]:
        return {
            'coefficient': self.coefficient.toDict()
            if isinstance(self.coefficient, thermalBoundaryField)
            else _jsonReady(self.coefficient),
            'ambientTemperature': self.ambientTemperature.toDict()
            if isinstance(self.ambientTemperature, thermalBoundaryField)
            else _jsonReady(self.ambientTemperature),
        }

    @classmethod
    def fromDict(cls, data: Mapping[str, Any]) -> 'thermalConvectionBoundary':
        return cls(
            coefficient=data['coefficient'],
            ambientTemperature=data['ambientTemperature'],
        )


@dataclass(frozen=True)
class thermalContactInterface:
    primaryBoundary: str
    secondaryBoundary: str
    coefficient: Any
    mapping: str = 'matching'
    maxDistance: float = 1e-8
    requireUniqueSecondary: bool = True

    def __post_init__(self):
        primary = str(self.primaryBoundary).strip()
        secondary = str(self.secondaryBoundary).strip()
        mapping = str(self.mapping).lower()
        coefficient = _boundaryFieldValue(self.coefficient, 'contact coefficient')
        coefficientValues = coefficient.values if isinstance(coefficient, thermalBoundaryField) else coefficient
        maxDistance = float(self.maxDistance)
        if not primary or not secondary or primary == secondary:
            raise ValueError('contact interfaces require two different non-empty boundary names')
        if mapping not in ('matching', 'nearest'):
            raise ValueError("contact mapping must be 'matching' or 'nearest'")
        if np.any(np.asarray(coefficientValues) < 0.0):
            raise ValueError('contact coefficient must be non-negative')
        if not np.isfinite(maxDistance) or maxDistance <= 0.0:
            raise ValueError('contact maxDistance must be positive and finite')
        object.__setattr__(self, 'primaryBoundary', primary)
        object.__setattr__(self, 'secondaryBoundary', secondary)
        object.__setattr__(self, 'coefficient', coefficient)
        object.__setattr__(self, 'mapping', mapping)
        object.__setattr__(self, 'maxDistance', maxDistance)
        object.__setattr__(self, 'requireUniqueSecondary', bool(self.requireUniqueSecondary))

    def toDict(self) -> dict[str, Any]:
        return {
            'primaryBoundary': self.primaryBoundary,
            'secondaryBoundary': self.secondaryBoundary,
            'coefficient': self.coefficient.toDict()
            if isinstance(self.coefficient, thermalBoundaryField)
            else _jsonReady(self.coefficient),
            'mapping': self.mapping,
            'maxDistance': self.maxDistance,
            'requireUniqueSecondary': self.requireUniqueSecondary,
        }

    @classmethod
    def fromDict(cls, data: Mapping[str, Any]) -> 'thermalContactInterface':
        return cls(
            primaryBoundary=data['primaryBoundary'],
            secondaryBoundary=data['secondaryBoundary'],
            coefficient=data['coefficient'],
            mapping=data.get('mapping', 'matching'),
            maxDistance=data.get('maxDistance', 1e-8),
            requireUniqueSecondary=data.get('requireUniqueSecondary', True),
        )


@dataclass(frozen=True)
class thermalLoadCase:
    heatSources: Mapping[str, Any] = field(default_factory=dict)
    convection: Mapping[str, thermalConvectionBoundary] = field(default_factory=dict)
    heatFlux: Mapping[str, Any] = field(default_factory=dict)
    dirichletTemperatures: Mapping[str, Any] = field(default_factory=dict)
    contacts: Mapping[str, thermalContactInterface] = field(default_factory=dict)
    operatingMetadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        heatSources = dict(self.heatSources)
        convection = {
            name: value
            if isinstance(value, thermalConvectionBoundary)
            else thermalConvectionBoundary.fromDict(value)
            for name, value in self.convection.items()
        }
        heatFlux = {
            name: _boundaryFieldValue(value, f'heatFlux[{name!r}]')
            for name, value in self.heatFlux.items()
        }
        dirichlet = dict(self.dirichletTemperatures)
        contacts = {
            name: value
            if isinstance(value, thermalContactInterface)
            else thermalContactInterface.fromDict(value)
            for name, value in self.contacts.items()
        }
        for name, value in heatSources.items():
            _finiteArray(value, f'heatSources[{name!r}]')
        for name, value in dirichlet.items():
            _finiteArray(value, f'dirichletTemperatures[{name!r}]')
        object.__setattr__(self, 'heatSources', heatSources)
        object.__setattr__(self, 'convection', convection)
        object.__setattr__(self, 'heatFlux', heatFlux)
        object.__setattr__(self, 'dirichletTemperatures', dirichlet)
        object.__setattr__(self, 'contacts', contacts)
        object.__setattr__(self, 'operatingMetadata', dict(self.operatingMetadata))

    def toDict(self) -> dict[str, Any]:
        return {
            'heatSources': _jsonReady(self.heatSources),
            'convection': {name: value.toDict() for name, value in self.convection.items()},
            'heatFlux': {
                name: value.toDict() if isinstance(value, thermalBoundaryField) else _jsonReady(value)
                for name, value in self.heatFlux.items()
            },
            'dirichletTemperatures': _jsonReady(self.dirichletTemperatures),
            'contacts': {name: value.toDict() for name, value in self.contacts.items()},
            'operatingMetadata': _jsonReady(self.operatingMetadata),
        }

    @classmethod
    def fromDict(cls, data: Mapping[str, Any]) -> 'thermalLoadCase':
        return cls(
            heatSources=data.get('heatSources', {}),
            convection=data.get('convection', {}),
            heatFlux=data.get('heatFlux', {}),
            dirichletTemperatures=data.get('dirichletTemperatures', {}),
            contacts=data.get('contacts', {}),
            operatingMetadata=data.get('operatingMetadata', {}),
        )


@dataclass(frozen=True)
class thermalNonlinearOptions:
    tolerance: float = 1e-6
    residualTolerance: float = 1e-6
    maxIterations: int = 50
    relaxation: float = 1.0

    def __post_init__(self):
        tolerance = float(self.tolerance)
        residualTolerance = float(self.residualTolerance)
        maxIterations = int(self.maxIterations)
        relaxation = float(self.relaxation)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError('nonlinear tolerance must be positive and finite')
        if not np.isfinite(residualTolerance) or residualTolerance <= 0.0:
            raise ValueError('nonlinear residualTolerance must be positive and finite')
        if maxIterations <= 0:
            raise ValueError('nonlinear maxIterations must be positive')
        if not np.isfinite(relaxation) or relaxation <= 0.0 or relaxation > 1.0:
            raise ValueError('nonlinear relaxation must be in (0, 1]')
        object.__setattr__(self, 'tolerance', tolerance)
        object.__setattr__(self, 'residualTolerance', residualTolerance)
        object.__setattr__(self, 'maxIterations', maxIterations)
        object.__setattr__(self, 'relaxation', relaxation)

    def toDict(self) -> dict[str, Any]:
        return {
            'tolerance': self.tolerance,
            'residualTolerance': self.residualTolerance,
            'maxIterations': self.maxIterations,
            'relaxation': self.relaxation,
        }

    @classmethod
    def fromDict(cls, data: Mapping[str, Any]) -> 'thermalNonlinearOptions':
        return cls(
            tolerance=data.get('tolerance', 1e-6),
            residualTolerance=data.get('residualTolerance', data.get('tolerance', 1e-6)),
            maxIterations=data.get('maxIterations', 50),
            relaxation=data.get('relaxation', 1.0),
        )


@dataclass(frozen=True)
class thermalLinearSolverOptions:
    method: str = 'auto'
    relativeTolerance: float = 1e-9
    absoluteTolerance: float = 0.0
    maxIterations: int | None = None
    preconditioner: str = 'jacobi'
    iterativeSizeThreshold: int = 2000
    fallbackToDirect: bool = True

    def __post_init__(self):
        method = str(self.method).lower()
        preconditioner = str(self.preconditioner).lower()
        relativeTolerance = float(self.relativeTolerance)
        absoluteTolerance = float(self.absoluteTolerance)
        maxIterations = None if self.maxIterations is None else int(self.maxIterations)
        iterativeSizeThreshold = int(self.iterativeSizeThreshold)
        if method not in ('auto', 'direct', 'cg', 'gmres'):
            raise ValueError("linear solver method must be 'auto', 'direct', 'cg', or 'gmres'")
        if preconditioner not in ('none', 'jacobi'):
            raise ValueError("linear solver preconditioner must be 'none' or 'jacobi'")
        if not np.isfinite(relativeTolerance) or relativeTolerance <= 0.0:
            raise ValueError('linear solver relativeTolerance must be positive and finite')
        if not np.isfinite(absoluteTolerance) or absoluteTolerance < 0.0:
            raise ValueError('linear solver absoluteTolerance must be non-negative and finite')
        if maxIterations is not None and maxIterations <= 0:
            raise ValueError('linear solver maxIterations must be positive when provided')
        if iterativeSizeThreshold <= 0:
            raise ValueError('linear solver iterativeSizeThreshold must be positive')
        object.__setattr__(self, 'method', method)
        object.__setattr__(self, 'preconditioner', preconditioner)
        object.__setattr__(self, 'relativeTolerance', relativeTolerance)
        object.__setattr__(self, 'absoluteTolerance', absoluteTolerance)
        object.__setattr__(self, 'maxIterations', maxIterations)
        object.__setattr__(self, 'iterativeSizeThreshold', iterativeSizeThreshold)
        object.__setattr__(self, 'fallbackToDirect', bool(self.fallbackToDirect))

    def toDict(self) -> dict[str, Any]:
        return {
            'method': self.method,
            'relativeTolerance': self.relativeTolerance,
            'absoluteTolerance': self.absoluteTolerance,
            'maxIterations': self.maxIterations,
            'preconditioner': self.preconditioner,
            'iterativeSizeThreshold': self.iterativeSizeThreshold,
            'fallbackToDirect': self.fallbackToDirect,
        }

    @classmethod
    def fromDict(cls, data: Mapping[str, Any]) -> 'thermalLinearSolverOptions':
        return cls(
            method=data.get('method', 'auto'),
            relativeTolerance=data.get('relativeTolerance', 1e-9),
            absoluteTolerance=data.get('absoluteTolerance', 0.0),
            maxIterations=data.get('maxIterations'),
            preconditioner=data.get('preconditioner', 'jacobi'),
            iterativeSizeThreshold=data.get('iterativeSizeThreshold', 2000),
            fallbackToDirect=data.get('fallbackToDirect', True),
        )


class thermalModel(modelBase):
    def __init__(
        self,
        operators: thermalOperatorSet,
        *,
        dirichletNodes: Any = (),
        dirichletValues: Any = (),
        metadata: Mapping[str, Any] | None = None,
        operatorUpdater: Callable[[Any], thermalOperatorSet] | None = None,
        referenceTemperature: float = 293.15,
        nonlinearOptions: thermalNonlinearOptions | Mapping[str, Any] | None = None,
        linearSolverOptions: thermalLinearSolverOptions | Mapping[str, Any] | None = None,
        backendName: str | None = None,
    ):
        super().__init__(backendName or operators.K.backendName)
        self.operators = operators
        self.K = operators.K
        self.M = operators.M
        self.Q = operators.Q
        self.dirichletNodes, self.dirichletValues = self._normalizeDirichlet(
            dirichletNodes,
            dirichletValues,
        )
        self.metadata = dict(metadata or {})
        self.operatorUpdater = operatorUpdater
        self.referenceTemperature = float(referenceTemperature)
        if not np.isfinite(self.referenceTemperature):
            raise ValueError('referenceTemperature must be finite')
        self.nonlinearOptions = (
            nonlinearOptions
            if isinstance(nonlinearOptions, thermalNonlinearOptions)
            else thermalNonlinearOptions.fromDict(nonlinearOptions or {})
        )
        self.linearSolverOptions = (
            linearSolverOptions
            if isinstance(linearSolverOptions, thermalLinearSolverOptions)
            else thermalLinearSolverOptions.fromDict(linearSolverOptions or {})
        )
        self.lastLinearSolveInfo: dict[str, Any] = {
            'method': None,
            'converged': False,
            'iterations': 0,
            'residualNorm': None,
            'relativeResidual': None,
            'usedFallback': False,
        }
        self.lastNonlinearInfo: dict[str, Any] = {
            'converged': not self.isNonlinear,
            'iterations': 0,
            'relativeChange': 0.0,
            'relativeResidual': 0.0,
        }
        self._validate()

    @property
    def order(self) -> int:
        return self.operators.nodeCount

    @property
    def isNonlinear(self) -> bool:
        return self.operatorUpdater is not None

    def _setOperators(self, operators: thermalOperatorSet) -> None:
        self.operators = operators
        self.K = operators.K
        self.M = operators.M
        self.Q = operators.Q
        self._validate()

    def _validate(self) -> None:
        n = self.order
        if self.K.shape != (n, n) or self.M.shape != (n, n):
            raise ValueError('thermal K and M operators must be square and have the same shape')
        if self.Q.shape not in ((n,), (n, 1)):
            raise ValueError('thermal Q operator must have shape (n,) or (n, 1)')
        if self.dirichletNodes.size and (
            self.dirichletNodes[0] < 0 or self.dirichletNodes[-1] >= n
        ):
            raise ValueError('Dirichlet node index out of range')

    @staticmethod
    def _normalizeDirichlet(nodeIndices: Any, values: Any) -> tuple[np.ndarray, np.ndarray]:
        rawIndices = np.asarray(nodeIndices, dtype=int).reshape(-1)
        if rawIndices.size == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        rawValues = np.asarray(values, dtype=float)
        if rawValues.ndim == 0:
            rawValues = np.full(rawIndices.size, float(rawValues), dtype=float)
        else:
            rawValues = rawValues.reshape(-1)
        if rawValues.size != rawIndices.size:
            raise ValueError('Dirichlet values must be scalar or match nodeIndices')
        prescribed: dict[int, float] = {}
        for nodeIndex, value in zip(rawIndices, rawValues):
            node = int(nodeIndex)
            temperature = float(value)
            if node in prescribed and not np.isclose(prescribed[node], temperature):
                raise ValueError(f'conflicting Dirichlet values for node {node}')
            prescribed[node] = temperature
        indices = np.array(sorted(prescribed), dtype=int)
        temperatures = np.array([prescribed[int(index)] for index in indices], dtype=float)
        return indices, temperatures

    def _temperatureColumn(self, value: Any, name: str) -> Any:
        data = value.toBackendData() if hasattr(value, 'toBackendData') else value
        array = self.localBackend.array.toArray(data)
        if getattr(array, 'ndim', 0) == 1:
            array = array.reshape(-1, 1)
        if getattr(array, 'shape', ()) != (self.order, 1):
            raise ValueError(f'{name} must have shape ({self.order},) or ({self.order}, 1)')
        return array

    def _solve(self, matrix: Any, rhs: Any) -> Any:
        system = matrix
        load = rhs
        backend = self.localBackend
        if self.dirichletNodes.size:
            systemOperator, loadOperator = applyDirichletConditions(
                matrix,
                rhs,
                self.dirichletNodes,
                self.dirichletValues,
                backendName=self.backendName or 'scipy',
            )
            system = systemOperator.toBackendData()
            load = loadOperator.toBackendData()
            backend = systemOperator.localBackend
        if self.backendName == 'scipy' or backend.__class__.__name__ == 'scipyBackend':
            solution = self._solveScipy(system, load)
        else:
            if self.linearSolverOptions.method not in ('auto', 'direct'):
                raise ValueError('iterative thermal solvers currently require the scipy backend')
            solution = backend.linalg.solve(system, load)
            solutionArray = np.asarray(backend.array.toArray(solution), dtype=float).reshape(-1)
            rhsArray = np.asarray(backend.array.toArray(load), dtype=float).reshape(-1)
            residual = np.asarray(system @ solutionArray).reshape(-1) - rhsArray
            residualNorm = float(np.linalg.norm(residual, ord=np.inf))
            rhsNorm = max(float(np.linalg.norm(rhsArray, ord=np.inf)), 1.0)
            self.lastLinearSolveInfo = {
                'method': 'direct',
                'converged': True,
                'iterations': 1,
                'residualNorm': residualNorm,
                'relativeResidual': residualNorm / rhsNorm,
                'usedFallback': False,
            }
        solution = backend.array.toArray(solution)
        return solution.reshape(self.order, 1)

    def _solveScipy(self, matrix: Any, rhs: Any) -> np.ndarray:
        options = self.linearSolverOptions
        rhsArray = np.asarray(rhs, dtype=float).reshape(-1)
        method = options.method
        if method == 'auto':
            method = 'cg' if issparse(matrix) and matrix.shape[0] >= options.iterativeSizeThreshold else 'direct'
        usedFallback = False
        iterations = 1
        if method == 'direct':
            solution = np.asarray(spsolve(matrix, rhsArray) if issparse(matrix) else np.linalg.solve(matrix, rhsArray))
            info = 0
        else:
            preconditioner = None
            if options.preconditioner == 'jacobi':
                diagonal = np.asarray(matrix.diagonal() if issparse(matrix) else np.diag(matrix), dtype=float)
                if np.any(~np.isfinite(diagonal)) or np.any(np.abs(diagonal) <= np.finfo(float).eps):
                    raise RuntimeError('Jacobi preconditioner requires a finite non-zero diagonal')
                inverseDiagonal = 1.0 / diagonal
                preconditioner = LinearOperator(matrix.shape, matvec=lambda value: inverseDiagonal * value)
            callbackCount = 0

            def countIteration(_):
                nonlocal callbackCount
                callbackCount += 1

            solver = cg if method == 'cg' else gmres
            kwargs = {
                'rtol': options.relativeTolerance,
                'atol': options.absoluteTolerance,
                'maxiter': options.maxIterations,
                'M': preconditioner,
                'callback': countIteration,
            }
            if method == 'gmres':
                kwargs['callback_type'] = 'legacy'
            solution, info = solver(matrix, rhsArray, **kwargs)
            solution = np.asarray(solution)
            iterations = callbackCount
            if info != 0 and options.fallbackToDirect:
                solution = np.asarray(spsolve(matrix, rhsArray) if issparse(matrix) else np.linalg.solve(matrix, rhsArray))
                usedFallback = True
                method = f'{method}->direct'
                info = 0
                iterations += 1
        residual = np.asarray(matrix @ solution).reshape(-1) - rhsArray
        residualNorm = float(np.linalg.norm(residual, ord=np.inf))
        rhsNorm = max(float(np.linalg.norm(rhsArray, ord=np.inf)), 1.0)
        self.lastLinearSolveInfo = {
            'method': method,
            'converged': info == 0,
            'iterations': iterations,
            'residualNorm': residualNorm,
            'relativeResidual': residualNorm / rhsNorm,
            'usedFallback': usedFallback,
        }
        if info != 0:
            raise RuntimeError(f'{method} thermal linear solve failed with solver info {info}')
        return solution.reshape(-1, 1)

    def _initialTemperature(self, value: Any = None) -> Any:
        if value is None:
            temperature = self.localBackend.array.toArray(
                np.full((self.order, 1), self.referenceTemperature, dtype=float)
            )
        else:
            temperature = self._temperatureColumn(value, 'initialTemperature')
        if self.dirichletNodes.size:
            temperature = temperature.copy()
            temperature[self.dirichletNodes, 0] = self.dirichletValues
        return temperature

    def _resolvedNonlinearOptions(
        self,
        value: thermalNonlinearOptions | Mapping[str, Any] | None,
    ) -> thermalNonlinearOptions:
        if value is None:
            return self.nonlinearOptions
        if isinstance(value, thermalNonlinearOptions):
            return value
        return thermalNonlinearOptions.fromDict(value)

    def _picardSolve(
        self,
        solveIteration: Callable[[], Any],
        residualIteration: Callable[[Any], tuple[Any, Any]],
        initialTemperature: Any,
        nonlinearOptions: thermalNonlinearOptions | Mapping[str, Any] | None,
    ) -> Any:
        options = self._resolvedNonlinearOptions(nonlinearOptions)
        estimate = self._initialTemperature(initialTemperature)
        for iteration in range(1, options.maxIterations + 1):
            self._setOperators(self.operatorUpdater(estimate))
            candidate = solveIteration()
            updated = estimate + options.relaxation * (candidate - estimate)
            if self.dirichletNodes.size:
                updated[self.dirichletNodes, 0] = self.dirichletValues
            change = float(np.max(np.abs(updated - estimate)))
            scale = max(float(np.max(np.abs(updated))), 1.0)
            relativeChange = change / scale
            estimate = updated
            self._setOperators(self.operatorUpdater(estimate))
            residual, reference = residualIteration(estimate)
            relativeResidual = self._relativeFreeResidual(residual, reference)
            if relativeChange <= options.tolerance and relativeResidual <= options.residualTolerance:
                self.lastNonlinearInfo = {
                    'converged': True,
                    'iterations': iteration,
                    'relativeChange': relativeChange,
                    'relativeResidual': relativeResidual,
                }
                return estimate
        self.lastNonlinearInfo = {
            'converged': False,
            'iterations': options.maxIterations,
            'relativeChange': relativeChange,
            'relativeResidual': relativeResidual,
        }
        raise RuntimeError(
            f'nonlinear thermal solve did not converge in {options.maxIterations} iterations '
            f'(relative change {relativeChange:.3e}, relative residual {relativeResidual:.3e})'
        )

    def _relativeFreeResidual(self, residual: Any, reference: Any) -> float:
        residualArray = np.asarray(residual, dtype=float).reshape(-1)
        referenceArray = np.asarray(reference, dtype=float).reshape(-1)
        if residualArray.size != self.order or referenceArray.size != self.order:
            raise ValueError('nonlinear residual and reference must match model order')
        if self.dirichletNodes.size:
            free = np.ones(self.order, dtype=bool)
            free[self.dirichletNodes] = False
            residualArray = residualArray[free]
            referenceArray = referenceArray[free]
        if residualArray.size == 0:
            return 0.0
        residualNorm = float(np.linalg.norm(residualArray, ord=np.inf))
        referenceNorm = max(float(np.linalg.norm(referenceArray, ord=np.inf)), 1.0)
        return residualNorm / referenceNorm

    def solveSteady(
        self,
        initialTemperature: Any = None,
        *,
        nonlinearOptions: thermalNonlinearOptions | Mapping[str, Any] | None = None,
    ) -> Any:
        if not self.isNonlinear:
            return self._solve(self.K.toBackendData(), self.Q.toBackendData())
        return self._picardSolve(
            lambda: self._solve(self.K.toBackendData(), self.Q.toBackendData()),
            lambda temperature: (
                self.operators.steadyResidual(temperature),
                self.Q.toBackendData(),
            ),
            initialTemperature,
            nonlinearOptions,
        )

    def implicitEulerStep(
        self,
        previousTemperature: Any,
        timeStep: float,
        *,
        initialTemperature: Any = None,
        nonlinearOptions: thermalNonlinearOptions | Mapping[str, Any] | None = None,
    ) -> Any:
        dt = float(timeStep)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError('timeStep must be positive and finite')
        previous = self._temperatureColumn(previousTemperature, 'previousTemperature')
        def solveIteration():
            matrix = self.M.toBackendData() / dt + self.K.toBackendData()
            rhs = self.Q.toBackendData() + self.M.apply(previous) / dt
            return self._solve(matrix, rhs)
        def residualIteration(temperature):
            rate = (temperature - previous) / dt
            reference = self.Q.toBackendData() + self.M.apply(previous) / dt
            return self.operators.transientResidual(temperature, rate), reference
        if not self.isNonlinear:
            return solveIteration()
        return self._picardSolve(
            solveIteration,
            residualIteration,
            previous if initialTemperature is None else initialTemperature,
            nonlinearOptions,
        )

    def solveTransient(
        self,
        initialTemperature: Any,
        *,
        timeStep: float,
        stepCount: int,
        nonlinearOptions: thermalNonlinearOptions | Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        dt = float(timeStep)
        count = int(stepCount)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError('timeStep must be positive and finite')
        if count <= 0:
            raise ValueError('stepCount must be positive')
        current = self._initialTemperature(initialTemperature)
        history = np.empty((count + 1, self.order, 1), dtype=float)
        history[0] = np.asarray(current, dtype=float)
        for step in range(1, count + 1):
            current = self.implicitEulerStep(
                current,
                dt,
                nonlinearOptions=nonlinearOptions,
            )
            history[step] = np.asarray(current, dtype=float)
        return np.arange(count + 1, dtype=float) * dt, history

    def steadyResidual(self, temperature: Any) -> Any:
        current = self._temperatureColumn(temperature, 'temperature')
        operators = self.operatorUpdater(current) if self.isNonlinear else self.operators
        return operators.steadyResidual(current)

    def transientResidual(self, temperature: Any, previousTemperature: Any, timeStep: float) -> Any:
        dt = float(timeStep)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError('timeStep must be positive and finite')
        current = self._temperatureColumn(temperature, 'temperature')
        previous = self._temperatureColumn(previousTemperature, 'previousTemperature')
        operators = self.operatorUpdater(current) if self.isNonlinear else self.operators
        return operators.transientResidual(current, (current - previous) / dt)

    def toNumpy(self) -> dict[str, Any]:
        return {
            'K': self.K.toNumpy(),
            'M': self.M.toNumpy(),
            'Q': self.Q.toNumpy(),
            'dirichletNodes': self.dirichletNodes.copy(),
            'dirichletValues': self.dirichletValues.copy(),
            'metadata': _jsonReady(self.metadata),
            'referenceTemperature': self.referenceTemperature,
            'isNonlinear': self.isNonlinear,
            'lastNonlinearInfo': dict(self.lastNonlinearInfo),
            'linearSolverOptions': self.linearSolverOptions.toDict(),
            'lastLinearSolveInfo': dict(self.lastLinearSolveInfo),
        }
