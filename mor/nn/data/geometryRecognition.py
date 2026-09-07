from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class geometryRecognitionRecord:
    caseId: str
    geometryId: str
    meshUuid: str
    componentType: str | None
    componentInstance: str | None
    caseDirectory: Path

    def load(self) -> 'geometryRecognitionCase':
        dataPath = self.caseDirectory / 'data.npz'
        if not dataPath.is_file():
            raise FileNotFoundError(f'geometry data file is missing: {dataPath}')
        with np.load(dataPath, allow_pickle=False) as data:
            required = ('nodesMeters', 'surfaceTriangles')
            missing = tuple(name for name in required if name not in data)
            if missing:
                raise KeyError(f'graph geometry arrays are missing from {dataPath}: {missing}')
            volumeCoordinates = (
                np.asarray(data['elementCentersMeters'], dtype=np.float64)
                if 'elementCentersMeters' in data
                else None
            )
            volumeWeights = (
                np.asarray(data['elementVolumesM3'], dtype=np.float64)
                if 'elementVolumesM3' in data
                else None
            )
            return geometryRecognitionCase(
                sourceId=self.caseId,
                geometryId=self.geometryId,
                meshUuid=self.meshUuid,
                componentType=self.componentType,
                componentInstance=self.componentInstance,
                nodesMeters=np.asarray(data['nodesMeters'], dtype=np.float64),
                surfaceTriangles=np.asarray(data['surfaceTriangles'], dtype=np.int64),
                volumeCoordinatesMeters=volumeCoordinates,
                volumeWeightsM3=volumeWeights,
            )


@dataclass(frozen=True)
class geometryRecognitionCase:
    sourceId: str
    geometryId: str
    meshUuid: str | None
    componentType: str | None
    componentInstance: str | None
    nodesMeters: np.ndarray
    surfaceTriangles: np.ndarray
    volumeCoordinatesMeters: np.ndarray | None = None
    volumeWeightsM3: np.ndarray | None = None

    def __post_init__(self):
        sourceId = str(self.sourceId).strip()
        geometryId = str(self.geometryId).strip()
        if not sourceId or not geometryId:
            raise ValueError('sourceId and geometryId must be non-empty')
        nodes = np.asarray(self.nodesMeters, dtype=np.float64)
        rawTriangles = np.asarray(self.surfaceTriangles)
        if not np.issubdtype(rawTriangles.dtype, np.integer):
            raise ValueError('surfaceTriangles must contain integer node indices')
        triangles = np.asarray(rawTriangles, dtype=np.int64)
        if nodes.ndim != 2 or nodes.shape[1] != 3 or nodes.shape[0] < 3:
            raise ValueError('nodesMeters must have shape (nodeCount, 3)')
        if not np.all(np.isfinite(nodes)):
            raise ValueError('nodesMeters must contain finite values')
        if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.shape[0] == 0:
            raise ValueError('surfaceTriangles must have shape (triangleCount, 3)')
        if np.any(triangles < 0) or np.any(triangles >= nodes.shape[0]):
            raise ValueError('surfaceTriangles contains invalid node indices')
        coordinates = self.volumeCoordinatesMeters
        weights = self.volumeWeightsM3
        if (coordinates is None) != (weights is None):
            raise ValueError('volume coordinates and weights must either both be set or both be absent')
        if coordinates is not None:
            coordinates = np.asarray(coordinates, dtype=np.float64)
            weights = np.asarray(weights, dtype=np.float64).reshape(-1)
            if coordinates.ndim != 2 or coordinates.shape[1] != 3:
                raise ValueError('volumeCoordinatesMeters must have shape (elementCount, 3)')
            if weights.size != coordinates.shape[0] or np.any(weights <= 0.0):
                raise ValueError('volumeWeightsM3 must contain one positive value per volume point')
            if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(weights)):
                raise ValueError('volume geometry must contain finite values')
        object.__setattr__(self, 'sourceId', sourceId)
        object.__setattr__(self, 'geometryId', geometryId)
        object.__setattr__(self, 'meshUuid', _optionalIdentifier(self.meshUuid))
        object.__setattr__(self, 'componentType', _optionalIdentifier(self.componentType))
        object.__setattr__(self, 'componentInstance', _optionalIdentifier(self.componentInstance))
        object.__setattr__(self, 'nodesMeters', nodes)
        object.__setattr__(self, 'surfaceTriangles', triangles)
        object.__setattr__(self, 'volumeCoordinatesMeters', coordinates)
        object.__setattr__(self, 'volumeWeightsM3', weights)


def _optionalIdentifier(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def loadGeometryRecognitionDatasetRecords(
    dataset: str | Path,
    *,
    deduplicateByMeshUuid: bool = True,
) -> tuple[geometryRecognitionRecord, ...]:
    path = Path(dataset)
    indexPath = path / 'dataset.json' if path.is_dir() else path
    if not indexPath.is_file():
        raise FileNotFoundError(f'geometry dataset index is missing: {indexPath}')
    index = json.loads(indexPath.read_text(encoding='utf-8'))
    if not isinstance(index, dict):
        raise ValueError('geometry dataset index must contain a JSON object')
    if not index.get('connectivityStored', False):
        raise ValueError('geometry recognition requires the graph dataset view')
    tasks = index.get('tasks', {})
    if not isinstance(tasks, dict):
        raise ValueError('geometry dataset tasks must contain a JSON object')
    classificationTask = tasks.get('geometryComponentClassification')
    declaredClasses = None
    if classificationTask is not None:
        if not isinstance(classificationTask, dict):
            raise ValueError('geometry component classification task must be a JSON object')
        if classificationTask.get('targetField') != 'componentType':
            raise ValueError(
                'geometry component classification targetField must be componentType'
            )
        rawClasses = classificationTask.get('classes')
        if not isinstance(rawClasses, list) or not rawClasses:
            raise ValueError(
                'geometry component classification classes must be a non-empty list'
            )
        declaredClasses = {
            str(value).strip() for value in rawClasses if str(value).strip()
        }
        if len(declaredClasses) != len(rawClasses):
            raise ValueError(
                'geometry component classification classes must be unique non-empty names'
            )
    entries = index.get('cases')
    if not isinstance(entries, list) or not entries:
        raise ValueError('geometry dataset index must contain a non-empty cases list')
    records = []
    labelsByMesh: dict[str, str | None] = {}
    seenMeshes = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('every geometry dataset entry must be a JSON object')
        caseId = str(entry.get('caseId', '')).strip()
        geometryId = str(entry.get('geometryId', '')).strip()
        meshUuid = str(entry.get('meshUuid', '')).strip()
        relativePath = str(entry.get('path', '')).strip()
        if not caseId or not geometryId or not meshUuid or not relativePath:
            raise ValueError('every geometry dataset entry must contain caseId, geometryId, meshUuid, and path')
        componentType = _optionalIdentifier(entry.get('componentType'))
        if declaredClasses is not None and componentType not in declaredClasses:
            raise ValueError(
                f'componentType {componentType!r} is not declared in the geometry '
                'component classification classes'
            )
        previousLabel = labelsByMesh.setdefault(meshUuid, componentType)
        if previousLabel != componentType:
            raise ValueError(
                f'meshUuid {meshUuid!r} has conflicting component types: '
                f'{previousLabel!r} and {componentType!r}'
            )
        if deduplicateByMeshUuid and meshUuid in seenMeshes:
            continue
        seenMeshes.add(meshUuid)
        casePath = indexPath.parent / relativePath
        records.append(
            geometryRecognitionRecord(
                caseId=caseId,
                geometryId=geometryId,
                meshUuid=meshUuid,
                componentType=componentType,
                componentInstance=_optionalIdentifier(entry.get('componentInstance')),
                caseDirectory=casePath,
            )
        )
    return tuple(records)


def iterGeometryRecognitionCases(
    dataset: str | Path,
    *,
    deduplicateByMeshUuid: bool = True,
) -> Iterator[geometryRecognitionCase]:
    for record in loadGeometryRecognitionDatasetRecords(
        dataset, deduplicateByMeshUuid=deduplicateByMeshUuid
    ):
        yield record.load()


def loadGeometryRecognitionCase(path: str | Path) -> geometryRecognitionCase:
    source = Path(path)
    caseDirectory = source if source.is_dir() else source.parent
    dataPath = caseDirectory / 'data.npz' if source.is_dir() else source
    manifestPath = caseDirectory / 'manifest.json'
    manifest = (
        json.loads(manifestPath.read_text(encoding='utf-8'))
        if manifestPath.is_file()
        else {}
    )
    record = geometryRecognitionRecord(
        caseId=str(manifest.get('caseId', dataPath.stem)),
        geometryId=str(manifest.get('geometryId', dataPath.stem)),
        meshUuid=str(manifest.get('meshUuid', '')).strip(),
        componentType=_optionalIdentifier(manifest.get('componentType')),
        componentInstance=_optionalIdentifier(manifest.get('componentInstance')),
        caseDirectory=caseDirectory,
    )
    if dataPath.name != 'data.npz':
        with np.load(dataPath, allow_pickle=False) as data:
            return geometryRecognitionCase(
                sourceId=record.caseId,
                geometryId=record.geometryId,
                meshUuid=record.meshUuid,
                componentType=record.componentType,
                componentInstance=record.componentInstance,
                nodesMeters=np.asarray(data['nodesMeters'], dtype=np.float64),
                surfaceTriangles=np.asarray(data['surfaceTriangles'], dtype=np.int64),
                volumeCoordinatesMeters=(
                    np.asarray(data['elementCentersMeters'], dtype=np.float64)
                    if 'elementCentersMeters' in data
                    else None
                ),
                volumeWeightsM3=(
                    np.asarray(data['elementVolumesM3'], dtype=np.float64)
                    if 'elementVolumesM3' in data
                    else None
                ),
            )
    return record.load()
