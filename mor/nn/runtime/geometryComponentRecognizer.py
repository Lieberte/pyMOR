from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from numbers import Integral
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, QhullError

from mor.nn.data.geometryRecognition import (
    geometryRecognitionCase,
    geometryRecognitionRecord,
)


FORMAT_VERSION = 'geometry-component-recognizer-v2'


@dataclass(frozen=True)
class geometryDescriptorConfig:
    surfaceSampleCount: int = 8192
    pairSampleCount: int = 32768
    triangleChunkSize: int = 131072
    radialBins: int = 16
    pairDistanceBins: int = 16
    normalBins: int = 12
    axisBins: int = 10
    componentAreaSlots: int = 8

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError('geometry descriptor configuration values must be positive integers')
            object.__setattr__(self, field.name, int(value))


@dataclass(frozen=True)
class geometryDescriptor:
    values: np.ndarray
    featureNames: tuple[str, ...]
    centerMeters: np.ndarray
    scaleMeters: float
    surfaceAreaM2: float
    volumeM3: float | None
    sourceNodeCount: int
    sourceTriangleCount: int
    sampledPointCount: int
    surfaceComponentCount: int


@dataclass(frozen=True)
class geometryComponentCandidate:
    componentType: str
    score: float
    distance: float


@dataclass(frozen=True)
class geometryComponentPrediction:
    predictedType: str
    confidence: float
    unknown: bool
    candidates: tuple[geometryComponentCandidate, ...]
    geometryId: str
    centerMeters: tuple[float, float, float]
    scaleMeters: float
    sourceNodeCount: int
    sampledPointCount: int
    surfaceComponentCount: int
    modelVersion: str = FORMAT_VERSION

    def toDict(self) -> dict:
        return {
            'predictedType': self.predictedType,
            'confidence': self.confidence,
            'unknown': self.unknown,
            'candidates': [asdict(candidate) for candidate in self.candidates],
            'geometryId': self.geometryId,
            'geometry': {
                'centerMeters': list(self.centerMeters),
                'scaleMeters': self.scaleMeters,
                'sourceNodeCount': self.sourceNodeCount,
                'sampledPointCount': self.sampledPointCount,
                'surfaceComponentCount': self.surfaceComponentCount,
            },
            'modelVersion': self.modelVersion,
        }


def _histogram(values: np.ndarray, bins: int, upper: float) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, np.nextafter(upper, 0.0))
    counts, _ = np.histogram(clipped, bins=bins, range=(0.0, upper))
    return counts.astype(np.float64) / max(1, clipped.size)


def _quantiles(values: np.ndarray) -> np.ndarray:
    return np.quantile(values, np.linspace(0.1, 0.9, 9))


def _triangleStatistics(case: geometryRecognitionCase, config: geometryDescriptorConfig):
    triangles = case.surfaceTriangles
    nodes = case.nodesMeters
    areas = np.empty(triangles.shape[0], dtype=np.float64)
    weightedCenter = np.zeros(3, dtype=np.float64)
    totalArea = 0.0
    for start in range(0, triangles.shape[0], config.triangleChunkSize):
        selection = triangles[start : start + config.triangleChunkSize]
        a = nodes[selection[:, 0]]
        b = nodes[selection[:, 1]]
        c = nodes[selection[:, 2]]
        chunkAreas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        areas[start : start + selection.shape[0]] = chunkAreas
        centers = (a + b + c) / 3.0
        weightedCenter += np.sum(centers * chunkAreas[:, None], axis=0)
        totalArea += float(chunkAreas.sum())
    if not np.isfinite(totalArea) or totalArea <= 0.0:
        raise ValueError('surface mesh must contain positive finite triangle area')
    positive = areas > np.finfo(np.float64).eps * totalArea
    if not np.any(positive):
        raise ValueError('surface mesh contains only degenerate triangles')
    return areas, positive, weightedCenter / totalArea, totalArea


def _surfaceSamples(
    case: geometryRecognitionCase,
    config: geometryDescriptorConfig,
    areas: np.ndarray,
    positive: np.ndarray,
):
    triangleIndices = np.flatnonzero(positive)
    positiveAreas = areas[positive]
    count = config.surfaceSampleCount
    cumulative = np.cumsum(positiveAreas)
    positions = (np.arange(count, dtype=np.float64) + 0.5) * cumulative[-1] / count
    chosen = triangleIndices[np.searchsorted(cumulative, positions, side='left')]
    triangles = case.surfaceTriangles[chosen]
    a = case.nodesMeters[triangles[:, 0]]
    b = case.nodesMeters[triangles[:, 1]]
    c = case.nodesMeters[triangles[:, 2]]
    localIndex = np.arange(count, dtype=np.float64)
    u = np.mod(localIndex * 0.7548776662466927 + 0.5, 1.0)
    v = np.mod(localIndex * 0.5698402909980532 + 0.25, 1.0)
    rootU = np.sqrt(u)
    points = (1.0 - rootU)[:, None] * a + (rootU * (1.0 - v))[:, None] * b + (rootU * v)[:, None] * c
    rawNormals = np.cross(b - a, c - a)
    normalNorm = np.linalg.norm(rawNormals, axis=1)
    normals = rawNormals / normalNorm[:, None]
    return points, normals


def _pairValues(points: np.ndarray, normals: np.ndarray, count: int):
    sampleCount = count
    indices = np.arange(sampleCount, dtype=np.int64)
    first = (indices * 104729 + 17) % points.shape[0]
    second = (indices * 130363 + 53) % points.shape[0]
    identical = first == second
    second[identical] = (second[identical] + 1) % points.shape[0]
    distances = np.linalg.norm(points[first] - points[second], axis=1)
    normalDots = np.abs(np.sum(normals[first] * normals[second], axis=1))
    return distances, normalDots


def _surfaceComponentFeatures(
    case: geometryRecognitionCase,
    areas: np.ndarray,
    positive: np.ndarray,
    slots: int,
):
    triangles = case.surfaceTriangles[positive]
    componentAreas = areas[positive]
    _, inverse = np.unique(triangles, return_inverse=True)
    inverse = inverse.reshape(-1, 3)
    rows = np.concatenate((inverse[:, 0], inverse[:, 1], inverse[:, 2]))
    columns = np.concatenate((inverse[:, 1], inverse[:, 2], inverse[:, 0]))
    graph = coo_matrix(
        (np.ones(rows.size, dtype=np.uint8), (rows, columns)),
        shape=(int(inverse.max()) + 1, int(inverse.max()) + 1),
    )
    componentCount, labels = connected_components(
        graph, directed=False, return_labels=True
    )
    areasByComponent = np.bincount(
        labels[inverse[:, 0]], weights=componentAreas, minlength=componentCount
    )
    fractions = np.sort(areasByComponent / areasByComponent.sum())[::-1]
    largestFractions = np.zeros(slots, dtype=np.float64)
    copied = min(slots, componentCount)
    largestFractions[:copied] = fractions[:copied]
    epsilon = np.finfo(np.float64).eps
    entropy = -float(np.sum(fractions * np.log(np.maximum(fractions, epsilon))))
    normalizedEntropy = entropy / max(float(np.log(max(componentCount, 2))), epsilon)
    effectiveCount = 1.0 / float(np.sum(fractions * fractions))
    tailFraction = max(0.0, 1.0 - float(largestFractions.sum()))
    return (
        componentCount,
        np.concatenate(
            (
                [np.log1p(componentCount)],
                largestFractions,
                [normalizedEntropy, np.log1p(effectiveCount), tailFraction],
            )
        ),
    )


def _convexityFeatures(
    normalized: np.ndarray,
    projected: np.ndarray,
    surfaceArea: float,
    scale: float,
    volume: float | None,
) -> np.ndarray:
    epsilon = np.finfo(np.float64).eps
    normalizedSurfaceArea = surfaceArea / (scale * scale)
    normalizedVolume = volume / (scale**3) if volume is not None else None
    surfaceHullRatio = 0.0
    volumeHullRatio = 0.0
    try:
        hull = ConvexHull(normalized)
        surfaceHullRatio = np.log1p(
            normalizedSurfaceArea / max(float(hull.area), epsilon)
        )
        if normalizedVolume is not None:
            volumeHullRatio = np.log1p(
                normalizedVolume / max(float(hull.volume), epsilon)
            )
    except QhullError:
        pass
    extents = np.ptp(projected, axis=0)
    boundingBoxVolume = float(np.prod(extents))
    volumeBoundingBoxRatio = (
        np.log1p(normalizedVolume / boundingBoxVolume)
        if normalizedVolume is not None and boundingBoxVolume > epsilon
        else 0.0
    )
    return np.asarray(
        [surfaceHullRatio, volumeHullRatio, volumeBoundingBoxRatio],
        dtype=np.float64,
    )


def extractGeometryDescriptor(
    case: geometryRecognitionCase,
    config: geometryDescriptorConfig | None = None,
) -> geometryDescriptor:
    config = config or geometryDescriptorConfig()
    areas, positive, surfaceCenter, surfaceArea = _triangleStatistics(case, config)
    points, normals = _surfaceSamples(case, config, areas, positive)
    volume = None
    if case.volumeCoordinatesMeters is not None:
        weights = case.volumeWeightsM3
        volume = float(weights.sum())
        center = np.average(case.volumeCoordinatesMeters, axis=0, weights=weights)
    else:
        center = surfaceCenter
    centered = points - center
    scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        raise ValueError('geometry scale must be positive and finite')
    normalized = centered / scale
    covariance = normalized.T @ normalized / normalized.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    eigenRatios = eigenvalues / max(float(eigenvalues.sum()), np.finfo(float).eps)
    projected = normalized @ eigenvectors
    extents = np.ptp(projected, axis=0)
    extentRatios = extents / max(float(extents.max()), np.finfo(float).eps)
    radial = np.linalg.norm(normalized, axis=1)
    radialDirection = np.divide(
        normalized,
        radial[:, None],
        out=np.zeros_like(normalized),
        where=radial[:, None] > np.finfo(float).eps,
    )
    normalRadial = np.abs(np.sum(normals * radialDirection, axis=1))
    pairDistances, pairNormalDots = _pairValues(
        normalized, normals, config.pairSampleCount
    )
    surfaceComponentCount, surfaceComponentFeatures = _surfaceComponentFeatures(
        case, areas, positive, config.componentAreaSlots
    )
    convexityFeatures = _convexityFeatures(
        normalized, projected, surfaceArea, scale, volume
    )
    values = []
    names = []

    def add(prefix: str, data):
        array = np.asarray(data, dtype=np.float64).reshape(-1)
        values.extend(array.tolist())
        names.extend(f'{prefix}_{index}' for index in range(array.size))

    add('inertia_ratio', eigenRatios)
    add('extent_ratio', extentRatios)
    add('radial_hist', _histogram(radial, config.radialBins, 3.0))
    add('radial_quantile', _quantiles(radial))
    add('pair_distance_hist', _histogram(pairDistances, config.pairDistanceBins, 4.0))
    add('pair_distance_quantile', _quantiles(pairDistances))
    add('normal_pair_hist', _histogram(pairNormalDots, config.normalBins, 1.0))
    add('normal_radial_hist', _histogram(normalRadial, config.normalBins, 1.0))
    for axis in range(3):
        add(
            f'axis_{axis}_absolute_hist',
            _histogram(np.abs(projected[:, axis]), config.axisBins, 3.0),
        )
    add('log_surface_area_ratio', [np.log1p(surfaceArea / (scale * scale))])
    add(
        'log_volume_ratio',
        [np.log1p(volume / (scale**3)) if volume is not None else 0.0],
    )
    add('volume_available', [1.0 if volume is not None else 0.0])
    add('surface_component', surfaceComponentFeatures)
    add('convexity', convexityFeatures)
    descriptorValues = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(descriptorValues)):
        raise ValueError('geometry descriptor contains non-finite values')
    return geometryDescriptor(
        values=descriptorValues,
        featureNames=tuple(names),
        centerMeters=np.asarray(center, dtype=np.float64),
        scaleMeters=scale,
        surfaceAreaM2=surfaceArea,
        volumeM3=volume,
        sourceNodeCount=int(case.nodesMeters.shape[0]),
        sourceTriangleCount=int(case.surfaceTriangles.shape[0]),
        sampledPointCount=int(points.shape[0]),
        surfaceComponentCount=int(surfaceComponentCount),
    )


class geometryComponentRecognizer:
    def __init__(
        self,
        *,
        config: geometryDescriptorConfig | None = None,
        minimumMargin: float = 0.02,
        rejectionScale: float = 0.65,
    ):
        self.config = config or geometryDescriptorConfig()
        self.minimumMargin = float(minimumMargin)
        self.rejectionScale = float(rejectionScale)
        if self.minimumMargin < 0.0 or not 0.0 < self.rejectionScale < 1.0:
            raise ValueError('recognizer margin and rejection scale are invalid')
        self.featureNames: tuple[str, ...] = ()
        self.featureMean: np.ndarray | None = None
        self.featureScale: np.ndarray | None = None
        self.prototypes: dict[str, tuple[dict, ...]] = {}
        self.thresholds: dict[str, float] = {}

    @property
    def isFitted(self) -> bool:
        return bool(self.prototypes)

    def _standardize(self, values: np.ndarray) -> np.ndarray:
        standardized = (values - self.featureMean) / self.featureScale
        norm = np.linalg.norm(standardized)
        return standardized / max(float(norm), np.finfo(float).eps)

    def fit(self, cases: Iterable[geometryRecognitionCase]) -> 'geometryComponentRecognizer':
        extracted = []
        labelsByMesh = {}
        seenMeshes = set()
        for case in cases:
            if case.componentType is None:
                raise ValueError(f'training geometry {case.geometryId!r} has no componentType')
            identity = case.meshUuid or case.geometryId
            previousLabel = labelsByMesh.setdefault(identity, case.componentType)
            if previousLabel != case.componentType:
                raise ValueError(
                    f'geometry identity {identity!r} has conflicting component types: '
                    f'{previousLabel!r} and {case.componentType!r}'
                )
            if identity in seenMeshes:
                continue
            seenMeshes.add(identity)
            descriptor = extractGeometryDescriptor(case, self.config)
            extracted.append((case, descriptor))
        if len(extracted) < 2:
            raise ValueError('at least two independent geometries are required')
        names = extracted[0][1].featureNames
        if any(descriptor.featureNames != names for _, descriptor in extracted):
            raise ValueError('geometry descriptor schemas do not match')
        matrix = np.stack([descriptor.values for _, descriptor in extracted])
        self.featureNames = names
        self.featureMean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        self.featureScale = np.where(scale > 1e-8, scale, 1.0)
        encoded = [self._standardize(row) for row in matrix]
        grouped: dict[str, list[dict]] = {}
        for (case, _), vector in zip(extracted, encoded):
            grouped.setdefault(case.componentType, []).append(
                {
                    'geometryId': case.geometryId,
                    'meshUuid': case.meshUuid,
                    'embedding': vector,
                }
            )
        self.prototypes = {label: tuple(items) for label, items in sorted(grouped.items())}
        self.thresholds = self._calibrateThresholds()
        return self

    def fitRecords(
        self, records: Iterable[geometryRecognitionRecord]
    ) -> 'geometryComponentRecognizer':
        def loadedCases():
            for record in records:
                yield record.load()

        return self.fit(loadedCases())

    def _calibrateThresholds(self) -> dict[str, float]:
        allItems = [
            (label, item['embedding'])
            for label, items in self.prototypes.items()
            for item in items
        ]
        thresholds = {}
        for label, items in self.prototypes.items():
            between = []
            within = []
            for item in items:
                vector = item['embedding']
                between.extend(
                    1.0 - float(np.dot(vector, other))
                    for otherLabel, other in allItems
                    if otherLabel != label
                )
                within.extend(
                    1.0 - float(np.dot(vector, other['embedding']))
                    for other in items
                    if other is not item
                )
            nearestOther = min(between) if between else 1.0
            withinRadius = max(within) if within else 0.0
            thresholds[label] = max(
                0.03,
                min(
                    nearestOther * self.rejectionScale,
                    max(0.12, withinRadius * 1.5 + 0.05),
                ),
            )
        return thresholds

    def predict(
        self,
        case: geometryRecognitionCase,
        *,
        topK: int = 3,
    ) -> geometryComponentPrediction:
        if not self.isFitted:
            raise RuntimeError('geometry component recognizer is not fitted')
        if not isinstance(topK, Integral) or isinstance(topK, bool) or topK <= 0:
            raise ValueError('topK must be a positive integer')
        descriptor = extractGeometryDescriptor(case, self.config)
        if descriptor.featureNames != self.featureNames:
            raise ValueError('geometry descriptor schema differs from the fitted model')
        embedding = self._standardize(descriptor.values)
        distances = []
        for label, items in self.prototypes.items():
            distance = min(
                1.0 - float(np.dot(embedding, item['embedding'])) for item in items
            )
            distances.append((label, max(0.0, distance)))
        distances.sort(key=lambda item: item[1])
        raw = np.asarray([-distance / 0.1 for _, distance in distances])
        probabilities = np.exp(raw - raw.max())
        probabilities /= probabilities.sum()
        candidates = tuple(
            geometryComponentCandidate(label, float(score), float(distance))
            for (label, distance), score in zip(distances[:topK], probabilities[:topK])
        )
        bestLabel, bestDistance = distances[0]
        margin = distances[1][1] - bestDistance if len(distances) > 1 else np.inf
        threshold = self.thresholds[bestLabel]
        distanceAcceptance = float(np.clip(1.0 - bestDistance / threshold, 0.0, 1.0))
        marginAcceptance = (
            1.0
            if self.minimumMargin == 0.0
            else float(np.clip(margin / self.minimumMargin, 0.0, 1.0))
        )
        knownConfidence = float(probabilities[0]) * min(
            distanceAcceptance, marginAcceptance
        )
        unknown = bestDistance > threshold or margin < self.minimumMargin
        confidence = 1.0 - knownConfidence if unknown else knownConfidence
        return geometryComponentPrediction(
            predictedType='unknown' if unknown else bestLabel,
            confidence=confidence,
            unknown=unknown,
            candidates=candidates,
            geometryId=case.geometryId,
            centerMeters=tuple(float(value) for value in descriptor.centerMeters),
            scaleMeters=float(descriptor.scaleMeters),
            sourceNodeCount=descriptor.sourceNodeCount,
            sampledPointCount=descriptor.sampledPointCount,
            surfaceComponentCount=descriptor.surfaceComponentCount,
        )

    def toDict(self) -> dict:
        if not self.isFitted:
            raise RuntimeError('geometry component recognizer is not fitted')
        return {
            'formatVersion': FORMAT_VERSION,
            'descriptorConfig': asdict(self.config),
            'minimumMargin': self.minimumMargin,
            'rejectionScale': self.rejectionScale,
            'featureNames': list(self.featureNames),
            'featureMean': self.featureMean.tolist(),
            'featureScale': self.featureScale.tolist(),
            'classes': {
                label: {
                    'distanceThreshold': self.thresholds[label],
                    'prototypes': [
                        {
                            'geometryId': item['geometryId'],
                            'meshUuid': item['meshUuid'],
                            'embedding': item['embedding'].tolist(),
                        }
                        for item in items
                    ],
                }
                for label, items in self.prototypes.items()
            },
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.toDict(), indent=2), encoding='utf-8')
        return output

    @classmethod
    def load(cls, path: str | Path) -> 'geometryComponentRecognizer':
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('formatVersion') != FORMAT_VERSION:
            raise ValueError(f'unsupported geometry recognizer format: {data.get("formatVersion")!r}')
        recognizer = cls(
            config=geometryDescriptorConfig(**data['descriptorConfig']),
            minimumMargin=data['minimumMargin'],
            rejectionScale=data['rejectionScale'],
        )
        recognizer.featureNames = tuple(data['featureNames'])
        recognizer.featureMean = np.asarray(data['featureMean'], dtype=np.float64)
        recognizer.featureScale = np.asarray(data['featureScale'], dtype=np.float64)
        recognizer.prototypes = {
            label: tuple(
                {
                    'geometryId': item['geometryId'],
                    'meshUuid': item.get('meshUuid'),
                    'embedding': np.asarray(item['embedding'], dtype=np.float64),
                }
                for item in classData['prototypes']
            )
            for label, classData in data['classes'].items()
        }
        recognizer.thresholds = {
            label: float(classData['distanceThreshold'])
            for label, classData in data['classes'].items()
        }
        recognizer._validateLoadedState()
        return recognizer

    def _validateLoadedState(self) -> None:
        featureCount = len(self.featureNames)
        if featureCount == 0 or len(set(self.featureNames)) != featureCount:
            raise ValueError('geometry recognizer feature names must be non-empty and unique')
        if self.featureMean.shape != (featureCount,) or self.featureScale.shape != (featureCount,):
            raise ValueError('geometry recognizer feature statistics have incompatible shapes')
        if (
            not np.all(np.isfinite(self.featureMean))
            or not np.all(np.isfinite(self.featureScale))
            or np.any(self.featureScale <= 0.0)
        ):
            raise ValueError('geometry recognizer feature statistics must be finite with positive scales')
        if not self.prototypes or set(self.prototypes) != set(self.thresholds):
            raise ValueError('geometry recognizer classes and thresholds do not match')
        for label, items in self.prototypes.items():
            threshold = self.thresholds[label]
            if not label or not items or not np.isfinite(threshold) or threshold <= 0.0:
                raise ValueError('geometry recognizer class metadata is invalid')
            for item in items:
                embedding = item['embedding']
                if embedding.shape != (featureCount,) or not np.all(np.isfinite(embedding)):
                    raise ValueError('geometry recognizer prototype embedding is invalid')
