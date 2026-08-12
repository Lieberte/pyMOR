from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .thermalOperatorCase import thermalOperatorCase


FORMAT_VERSION = 'thermal-component-case-v3'
SUPPORTED_FORMAT_VERSIONS = frozenset(
    ('thermal-component-case-v1', 'thermal-component-case-v2', 'thermal-component-case-v3')
)


def _finiteMatrix(value: Any, columns: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != columns:
        raise ValueError(f'{name} must have shape (n, {columns}) with n > 0')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must contain finite values')
    return array


def _finiteVector(value: Any, count: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != count or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must contain {count} finite values')
    return array


def _positiveVector(value: Any, count: int, name: str) -> np.ndarray:
    array = _finiteVector(value, count, name)
    if np.any(array <= 0.0):
        raise ValueError(f'{name} must be positive')
    return array


def _extraFeatures(value: Any | None, count: int, name: str) -> np.ndarray:
    if value is None:
        return np.empty((count, 0), dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] != count or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must have shape ({count}, d) and contain finite values')
    return array


def _conductivityTensors(value: Any, count: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        tensors = np.repeat((np.eye(3) * float(array))[None], count, axis=0)
    elif array.shape == (count,):
        tensors = array[:, None, None] * np.eye(3)[None]
    elif array.shape == (3,):
        tensors = np.repeat(np.diag(array)[None], count, axis=0)
    elif array.shape == (count, 3):
        tensors = np.zeros((count, 3, 3), dtype=np.float64)
        tensors[:, np.arange(3), np.arange(3)] = array
    elif array.shape == (3, 3):
        tensors = np.repeat(array[None], count, axis=0)
    elif array.shape == (count, 3, 3):
        tensors = array.copy()
    else:
        raise ValueError('conductivity must be scalar, principal values, or 3x3 tensors')
    if not np.all(np.isfinite(tensors)):
        raise ValueError('conductivity must contain finite values')
    symmetric = 0.5 * (tensors + np.swapaxes(tensors, 1, 2))
    if not np.allclose(tensors, symmetric):
        raise ValueError('conductivity tensors must be symmetric')
    if np.any(np.linalg.eigvalsh(symmetric) <= 0.0):
        raise ValueError('conductivity tensors must be symmetric positive definite')
    return symmetric


def _jsonReady(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonReady(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonReady(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError('JSON metadata numbers must be finite')
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'unsupported JSON metadata value: {type(value).__name__}')


@dataclass(frozen=True)
class thermalCoordinateFrame:
    centerMeters: Any
    halfExtentMeters: Any

    def __post_init__(self):
        center = np.asarray(self.centerMeters, dtype=np.float64).reshape(-1)
        halfExtent = np.asarray(self.halfExtentMeters, dtype=np.float64).reshape(-1)
        if center.shape != (3,) or halfExtent.shape != (3,):
            raise ValueError('coordinate frame center and half extent must contain three values')
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(halfExtent)):
            raise ValueError('coordinate frame must contain finite values')
        if np.any(halfExtent <= 0.0):
            raise ValueError('coordinate frame half extents must be positive')
        object.__setattr__(self, 'centerMeters', center)
        object.__setattr__(self, 'halfExtentMeters', halfExtent)

    @classmethod
    def fromCoordinates(cls, coordinatesMeters: Any) -> 'thermalCoordinateFrame':
        coordinates = _finiteMatrix(coordinatesMeters, 3, 'coordinatesMeters')
        minimum = coordinates.min(axis=0)
        maximum = coordinates.max(axis=0)
        return cls(
            centerMeters=0.5 * (minimum + maximum),
            halfExtentMeters=0.5 * (maximum - minimum),
        )

    def normalize(self, coordinatesMeters: Any) -> np.ndarray:
        coordinates = _finiteMatrix(coordinatesMeters, 3, 'coordinatesMeters')
        return (coordinates - self.centerMeters) / self.halfExtentMeters

    def denormalize(self, normalizedCoordinates: Any) -> np.ndarray:
        normalized = _finiteMatrix(normalizedCoordinates, 3, 'normalizedCoordinates')
        return self.centerMeters + normalized * self.halfExtentMeters


@dataclass(frozen=True)
class thermalVolumeSamples:
    coordinatesMeters: Any
    conductivityWPerMK: Any
    densityKgPerM3: Any
    specificHeatJPerKgK: Any
    heatSourceWPerM3: Any
    weightsM3: Any
    extraFeatures: Any | None = None

    def __post_init__(self):
        coordinates = _finiteMatrix(self.coordinatesMeters, 3, 'volume coordinatesMeters')
        count = coordinates.shape[0]
        density = _positiveVector(self.densityKgPerM3, count, 'densityKgPerM3')
        specificHeat = _positiveVector(self.specificHeatJPerKgK, count, 'specificHeatJPerKgK')
        object.__setattr__(self, 'coordinatesMeters', coordinates)
        object.__setattr__(
            self, 'conductivityWPerMK', _conductivityTensors(self.conductivityWPerMK, count)
        )
        object.__setattr__(self, 'densityKgPerM3', density)
        object.__setattr__(self, 'specificHeatJPerKgK', specificHeat)
        object.__setattr__(
            self, 'heatSourceWPerM3', _finiteVector(self.heatSourceWPerM3, count, 'heatSourceWPerM3')
        )
        object.__setattr__(self, 'weightsM3', _positiveVector(self.weightsM3, count, 'weightsM3'))
        object.__setattr__(
            self, 'extraFeatures', _extraFeatures(self.extraFeatures, count, 'volume extraFeatures')
        )

    @property
    def totalHeatSourcePowerW(self) -> float:
        """Return the volume integral of the pointwise heat-source density."""

        return float(np.dot(self.heatSourceWPerM3, self.weightsM3))


@dataclass(frozen=True)
class thermalBoundarySamples:
    coordinatesMeters: Any
    normals: Any
    htcWPerM2K: Any
    ambientTemperatureK: Any
    weightsM2: Any
    boundaryTypeIds: Any | None = None
    heatFluxWPerM2: Any | None = None
    patchIds: Any | None = None
    extraFeatures: Any | None = None

    def __post_init__(self):
        coordinates = _finiteMatrix(self.coordinatesMeters, 3, 'boundary coordinatesMeters')
        count = coordinates.shape[0]
        normals = _finiteMatrix(self.normals, 3, 'boundary normals')
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths <= np.finfo(float).eps):
            raise ValueError('boundary normals must be non-zero')
        normals = normals / lengths[:, None]
        htc = _finiteVector(self.htcWPerM2K, count, 'htcWPerM2K')
        if np.any(htc < 0.0):
            raise ValueError('htcWPerM2K must be non-negative')
        boundaryTypes = (
            np.zeros(count, dtype=np.int64)
            if self.boundaryTypeIds is None
            else np.asarray(self.boundaryTypeIds, dtype=np.int64).reshape(-1)
        )
        if boundaryTypes.size != count or np.any(boundaryTypes < 0):
            raise ValueError('boundaryTypeIds must contain one non-negative id per boundary point')
        heatFlux = (
            np.zeros(count, dtype=np.float64)
            if self.heatFluxWPerM2 is None
            else _finiteVector(self.heatFluxWPerM2, count, 'heatFluxWPerM2')
        )
        patchIds = (
            np.zeros(count, dtype=np.int64)
            if self.patchIds is None
            else np.asarray(self.patchIds, dtype=np.int64).reshape(-1)
        )
        if patchIds.size != count or np.any(patchIds < 0):
            raise ValueError('patchIds must contain one non-negative id per boundary point')
        object.__setattr__(self, 'coordinatesMeters', coordinates)
        object.__setattr__(self, 'normals', normals)
        object.__setattr__(self, 'htcWPerM2K', htc)
        object.__setattr__(
            self,
            'ambientTemperatureK',
            _finiteVector(self.ambientTemperatureK, count, 'ambientTemperatureK'),
        )
        object.__setattr__(self, 'weightsM2', _positiveVector(self.weightsM2, count, 'weightsM2'))
        object.__setattr__(self, 'boundaryTypeIds', boundaryTypes)
        object.__setattr__(self, 'heatFluxWPerM2', heatFlux)
        object.__setattr__(self, 'patchIds', patchIds)
        object.__setattr__(
            self, 'extraFeatures', _extraFeatures(self.extraFeatures, count, 'boundary extraFeatures')
        )

    @property
    def totalHeatFlowRateW(self) -> float:
        """Return the area integral of the pointwise normal heat-flux density."""

        return float(np.dot(self.heatFluxWPerM2, self.weightsM2))

    def heatFlowRateWByPatch(self) -> dict[int, float]:
        """Return area-integrated heat-flow rate for every surface patch."""

        return {
            int(patchId): float(
                np.dot(
                    self.heatFluxWPerM2[self.patchIds == patchId],
                    self.weightsM2[self.patchIds == patchId],
                )
            )
            for patchId in np.unique(self.patchIds)
        }


@dataclass(frozen=True)
class thermalTemperatureSamples:
    coordinatesMeters: Any
    temperatureK: Any
    weightsM3: Any

    def __post_init__(self):
        coordinates = _finiteMatrix(self.coordinatesMeters, 3, 'temperature coordinatesMeters')
        count = coordinates.shape[0]
        object.__setattr__(self, 'coordinatesMeters', coordinates)
        object.__setattr__(
            self, 'temperatureK', _finiteVector(self.temperatureK, count, 'temperatureK')
        )
        object.__setattr__(self, 'weightsM3', _positiveVector(self.weightsM3, count, 'weightsM3'))


@dataclass(frozen=True)
class thermalOperatorFeatureScaling:
    lengthMeters: float = 1.0
    conductivityWPerMK: float = 100.0
    densityKgPerM3: float = 10000.0
    specificHeatJPerKgK: float = 1000.0
    heatSourceWPerM3: float = 1e6
    htcWPerM2K: float = 1000.0
    heatFluxWPerM2: float = 1e5
    temperatureReferenceK: float = 293.15
    temperatureScaleK: float = 100.0
    operatingScales: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        positiveNames = (
            'lengthMeters',
            'conductivityWPerMK',
            'densityKgPerM3',
            'specificHeatJPerKgK',
            'heatSourceWPerM3',
            'htcWPerM2K',
            'heatFluxWPerM2',
            'temperatureScaleK',
        )
        for name in positiveNames:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be positive and finite')
            object.__setattr__(self, name, value)
        reference = float(self.temperatureReferenceK)
        if not np.isfinite(reference):
            raise ValueError('temperatureReferenceK must be finite')
        object.__setattr__(self, 'temperatureReferenceK', reference)
        scales = {str(name): float(value) for name, value in self.operatingScales.items()}
        if any(not np.isfinite(value) or value <= 0.0 for value in scales.values()):
            raise ValueError('operating feature scales must be positive and finite')
        object.__setattr__(self, 'operatingScales', scales)


@dataclass(frozen=True)
class thermalComponentFeatureSchema:
    operatingFeatureNames: Sequence[str] = ()
    boundaryTypeCount: int = 1
    temperatureBoundaryTypeIds: Sequence[int] = ()
    fixedTemperatureBoundaryTypeIds: Sequence[int] = ()

    def __post_init__(self):
        names = tuple(str(name).strip() for name in self.operatingFeatureNames)
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError('operatingFeatureNames must be unique non-empty strings')
        boundaryTypeCount = int(self.boundaryTypeCount)
        if boundaryTypeCount <= 0:
            raise ValueError('boundaryTypeCount must be positive')
        temperatureBoundaryTypeIds = tuple(
            int(value) for value in self.temperatureBoundaryTypeIds
        )
        if (
            len(set(temperatureBoundaryTypeIds)) != len(temperatureBoundaryTypeIds)
            or any(
                value < 0 or value >= boundaryTypeCount
                for value in temperatureBoundaryTypeIds
            )
        ):
            raise ValueError(
                'temperatureBoundaryTypeIds must contain unique ids within '
                'boundaryTypeCount'
            )
        fixedTemperatureBoundaryTypeIds = tuple(
            int(value) for value in self.fixedTemperatureBoundaryTypeIds
        )
        if (
            len(set(fixedTemperatureBoundaryTypeIds))
            != len(fixedTemperatureBoundaryTypeIds)
            or any(
                value < 0 or value >= boundaryTypeCount
                for value in fixedTemperatureBoundaryTypeIds
            )
            or any(
                value not in temperatureBoundaryTypeIds
                for value in fixedTemperatureBoundaryTypeIds
            )
        ):
            raise ValueError(
                'fixedTemperatureBoundaryTypeIds must be a unique subset of '
                'temperatureBoundaryTypeIds'
            )
        object.__setattr__(self, 'operatingFeatureNames', names)
        object.__setattr__(self, 'boundaryTypeCount', boundaryTypeCount)
        object.__setattr__(
            self, 'temperatureBoundaryTypeIds', temperatureBoundaryTypeIds
        )
        object.__setattr__(
            self,
            'fixedTemperatureBoundaryTypeIds',
            fixedTemperatureBoundaryTypeIds,
        )


@dataclass(frozen=True)
class thermalComponentCase:
    caseId: str
    geometryId: str
    componentType: str
    volume: thermalVolumeSamples
    boundary: thermalBoundarySamples
    temperature: thermalTemperatureSamples
    operatingFeatures: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        identifiers = tuple(str(value).strip() for value in (self.caseId, self.geometryId, self.componentType))
        if any(not value for value in identifiers):
            raise ValueError('caseId, geometryId, and componentType must be non-empty')
        if any(value in ('.', '..') or '/' in value or '\\' in value for value in identifiers):
            raise ValueError('case identifiers cannot contain path separators or traversal names')
        operating = {str(name): float(value) for name, value in self.operatingFeatures.items()}
        if any(not name or not np.isfinite(value) for name, value in operating.items()):
            raise ValueError('operatingFeatures must contain finite named values')
        object.__setattr__(self, 'caseId', identifiers[0])
        object.__setattr__(self, 'geometryId', identifiers[1])
        object.__setattr__(self, 'componentType', identifiers[2])
        object.__setattr__(self, 'operatingFeatures', operating)
        object.__setattr__(self, 'metadata', _jsonReady(dict(self.metadata)))

    @property
    def coordinateFrame(self) -> thermalCoordinateFrame:
        return thermalCoordinateFrame.fromCoordinates(self.volume.coordinatesMeters)

    def toOperatorCase(
        self,
        *,
        schema: thermalComponentFeatureSchema | None = None,
        scaling: thermalOperatorFeatureScaling | None = None,
    ) -> thermalOperatorCase:
        resolvedSchema = schema or thermalComponentFeatureSchema()
        resolvedScaling = scaling or thermalOperatorFeatureScaling()
        if np.any(self.boundary.boundaryTypeIds >= resolvedSchema.boundaryTypeCount):
            raise ValueError('boundaryTypeIds exceed schema boundaryTypeCount')
        frame = self.coordinateFrame
        volumeCoordinates = frame.normalize(self.volume.coordinatesMeters)
        boundaryCoordinates = frame.normalize(self.boundary.coordinatesMeters)
        queryCoordinates = frame.normalize(self.temperature.coordinatesMeters)
        conductivity = self.volume.conductivityWPerMK.reshape(-1, 9) / resolvedScaling.conductivityWPerMK
        volumeFeatures = np.column_stack(
            (
                volumeCoordinates,
                conductivity,
                self.volume.densityKgPerM3 / resolvedScaling.densityKgPerM3,
                self.volume.specificHeatJPerKgK / resolvedScaling.specificHeatJPerKgK,
                self.volume.heatSourceWPerM3 / resolvedScaling.heatSourceWPerM3,
                self.volume.extraFeatures,
            )
        )
        boundaryTypeOneHot = np.eye(resolvedSchema.boundaryTypeCount, dtype=np.float64)[
            self.boundary.boundaryTypeIds
        ]
        temperatureActive = self.boundary.htcWPerM2K > 0.0
        if resolvedSchema.temperatureBoundaryTypeIds:
            temperatureActive |= np.isin(
                self.boundary.boundaryTypeIds,
                resolvedSchema.temperatureBoundaryTypeIds,
            )
        normalizedBoundaryTemperature = np.where(
            temperatureActive,
            (
                self.boundary.ambientTemperatureK
                - resolvedScaling.temperatureReferenceK
            )
            / resolvedScaling.temperatureScaleK,
            0.0,
        )
        boundaryFeatures = np.column_stack(
            (
                boundaryCoordinates,
                self.boundary.normals,
                self.boundary.htcWPerM2K / resolvedScaling.htcWPerM2K,
                normalizedBoundaryTemperature,
                self.boundary.heatFluxWPerM2 / resolvedScaling.heatFluxWPerM2,
                boundaryTypeOneHot,
                self.boundary.extraFeatures,
            )
        )
        operating = []
        for name in resolvedSchema.operatingFeatureNames:
            if name not in self.operatingFeatures:
                raise KeyError(f'operatingFeatures does not contain required feature {name!r}')
            operating.append(
                self.operatingFeatures[name]
                / resolvedScaling.operatingScales.get(name, 1.0)
            )
        globalFeatures = np.concatenate(
            (
                frame.halfExtentMeters / resolvedScaling.lengthMeters,
                np.array(
                    [
                        self.volume.weightsM3.sum() / resolvedScaling.lengthMeters**3,
                        self.boundary.weightsM2.sum() / resolvedScaling.lengthMeters**2,
                    ]
                ),
                np.asarray(operating, dtype=np.float64),
            )
        )
        return thermalOperatorCase(
            caseId=self.caseId,
            geometryId=self.geometryId,
            volumeFeatures=volumeFeatures,
            volumeWeights=self.volume.weightsM3 / resolvedScaling.lengthMeters**3,
            boundaryFeatures=boundaryFeatures,
            boundaryWeights=self.boundary.weightsM2 / resolvedScaling.lengthMeters**2,
            globalFeatures=globalFeatures,
            queryCoordinates=queryCoordinates,
            queryTemperatures=(
                self.temperature.temperatureK - resolvedScaling.temperatureReferenceK
            )
            / resolvedScaling.temperatureScaleK,
            queryWeights=self.temperature.weightsM3 / resolvedScaling.lengthMeters**3,
        )


def saveThermalComponentCase(
    case: thermalComponentCase,
    directory: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    root = Path(directory)
    if not overwrite and ((root / 'manifest.json').exists() or (root / 'fields.npz').exists()):
        raise FileExistsError(f'thermal component case already exists: {root}')
    root.mkdir(parents=True, exist_ok=True)
    arraysPath = root / 'fields.npz'
    manifestPath = root / 'manifest.json'
    np.savez_compressed(
        arraysPath,
        volumeCoordinatesMeters=case.volume.coordinatesMeters,
        volumeConductivityWPerMK=case.volume.conductivityWPerMK,
        volumeDensityKgPerM3=case.volume.densityKgPerM3,
        volumeSpecificHeatJPerKgK=case.volume.specificHeatJPerKgK,
        volumeHeatSourceWPerM3=case.volume.heatSourceWPerM3,
        volumeWeightsM3=case.volume.weightsM3,
        volumeExtraFeatures=case.volume.extraFeatures,
        boundaryCoordinatesMeters=case.boundary.coordinatesMeters,
        boundaryNormals=case.boundary.normals,
        boundaryHtcWPerM2K=case.boundary.htcWPerM2K,
        boundaryAmbientTemperatureK=case.boundary.ambientTemperatureK,
        boundaryWeightsM2=case.boundary.weightsM2,
        boundaryTypeIds=case.boundary.boundaryTypeIds,
        boundaryHeatFluxWPerM2=case.boundary.heatFluxWPerM2,
        boundaryPatchIds=case.boundary.patchIds,
        boundaryExtraFeatures=case.boundary.extraFeatures,
        queryCoordinatesMeters=case.temperature.coordinatesMeters,
        queryTemperatureK=case.temperature.temperatureK,
        queryWeightsM3=case.temperature.weightsM3,
    )
    manifestPath.write_text(
        json.dumps(
            {
                'formatVersion': FORMAT_VERSION,
                'caseId': case.caseId,
                'geometryId': case.geometryId,
                'componentType': case.componentType,
                'arrayFile': arraysPath.name,
                'operatingFeatures': case.operatingFeatures,
                'metadata': case.metadata,
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    return manifestPath


def loadThermalComponentCase(directory: str | Path) -> thermalComponentCase:
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('formatVersion') not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(f"unsupported thermal component format: {manifest.get('formatVersion')!r}")
    with np.load(root / manifest['arrayFile'], allow_pickle=False) as arrays:
        return thermalComponentCase(
            caseId=manifest['caseId'],
            geometryId=manifest['geometryId'],
            componentType=manifest['componentType'],
            volume=thermalVolumeSamples(
                coordinatesMeters=arrays['volumeCoordinatesMeters'],
                conductivityWPerMK=arrays['volumeConductivityWPerMK'],
                densityKgPerM3=arrays['volumeDensityKgPerM3'],
                specificHeatJPerKgK=arrays['volumeSpecificHeatJPerKgK'],
                heatSourceWPerM3=arrays['volumeHeatSourceWPerM3'],
                weightsM3=arrays['volumeWeightsM3'],
                extraFeatures=arrays['volumeExtraFeatures'],
            ),
            boundary=thermalBoundarySamples(
                coordinatesMeters=arrays['boundaryCoordinatesMeters'],
                normals=arrays['boundaryNormals'],
                htcWPerM2K=arrays['boundaryHtcWPerM2K'],
                ambientTemperatureK=arrays['boundaryAmbientTemperatureK'],
                weightsM2=arrays['boundaryWeightsM2'],
                boundaryTypeIds=arrays['boundaryTypeIds'],
                heatFluxWPerM2=(
                    arrays['boundaryHeatFluxWPerM2']
                    if 'boundaryHeatFluxWPerM2' in arrays
                    else None
                ),
                patchIds=(
                    arrays['boundaryPatchIds']
                    if 'boundaryPatchIds' in arrays
                    else None
                ),
                extraFeatures=arrays['boundaryExtraFeatures'],
            ),
            temperature=thermalTemperatureSamples(
                coordinatesMeters=arrays['queryCoordinatesMeters'],
                temperatureK=arrays['queryTemperatureK'],
                weightsM3=arrays['queryWeightsM3'],
            ),
            operatingFeatures=manifest.get('operatingFeatures', {}),
            metadata=manifest.get('metadata', {}),
        )


def saveThermalComponentDataset(
    cases: Sequence[thermalComponentCase],
    directory: str | Path,
    *,
    boundaryTypeNames: Mapping[int, str] | None = None,
    datasetMetadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    root = Path(directory)
    if not overwrite and (root / 'dataset.json').exists():
        raise FileExistsError(f'thermal component dataset already exists: {root}')
    casesDirectory = root / 'cases'
    entries = []
    seen = set()
    resolvedCases = tuple(cases)
    boundaryNames = {
        int(identifier): str(name).strip()
        for identifier, name in (boundaryTypeNames or {}).items()
    }
    if any(identifier < 0 or not name for identifier, name in boundaryNames.items()):
        raise ValueError('boundaryTypeNames requires non-negative ids and non-empty names')
    for case in resolvedCases:
        if case.caseId in seen:
            raise ValueError(f'duplicate thermal component caseId: {case.caseId!r}')
        seen.add(case.caseId)
        relativeDirectory = Path('cases') / case.caseId
        saveThermalComponentCase(case, root / relativeDirectory, overwrite=overwrite)
        entries.append(
            {
                'caseId': case.caseId,
                'geometryId': case.geometryId,
                'componentType': case.componentType,
                'path': relativeDirectory.as_posix(),
            }
        )
    casesDirectory.mkdir(parents=True, exist_ok=True)
    indexPath = root / 'dataset.json'
    indexPath.write_text(
        json.dumps(
            {
                'formatVersion': FORMAT_VERSION,
                'featureSchema': {
                    'operatingFeatureNames': sorted(
                        {name for case in resolvedCases for name in case.operatingFeatures}
                    ),
                    'boundaryTypeNames': {
                        str(identifier): name for identifier, name in sorted(boundaryNames.items())
                    },
                },
                'metadata': _jsonReady(datasetMetadata or {}),
                'cases': entries,
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    return indexPath


def loadThermalComponentDataset(directory: str | Path) -> tuple[thermalComponentCase, ...]:
    root = Path(directory)
    index = json.loads((root / 'dataset.json').read_text(encoding='utf-8'))
    if index.get('formatVersion') not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(f"unsupported thermal component dataset: {index.get('formatVersion')!r}")
    return tuple(loadThermalComponentCase(root / entry['path']) for entry in index['cases'])
