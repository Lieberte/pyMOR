from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .geometryRecognition import geometryRecognitionCase, geometryRecognitionRecord


@dataclass(frozen=True)
class geometryPointCloudSample:
    geometryId: str
    meshUuid: str
    componentType: str
    pointsAndNormals: np.ndarray

    def __post_init__(self):
        values = np.asarray(self.pointsAndNormals, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 6 or values.shape[0] < 2:
            raise ValueError('geometry point cloud must have shape (pointCount, 6)')
        if not np.all(np.isfinite(values)):
            raise ValueError('geometry point cloud must contain finite values')
        object.__setattr__(self, 'pointsAndNormals', values)


def _positiveTriangleGeometry(case: geometryRecognitionCase):
    triangles = case.surfaceTriangles
    nodes = case.nodesMeters
    a = nodes[triangles[:, 0]]
    b = nodes[triangles[:, 1]]
    c = nodes[triangles[:, 2]]
    rawNormals = np.cross(b - a, c - a)
    doubleAreas = np.linalg.norm(rawNormals, axis=1)
    positive = doubleAreas > np.finfo(np.float64).eps * max(
        float(doubleAreas.sum()), 1.0
    )
    if not np.any(positive):
        raise ValueError('surface mesh contains no positive-area triangles')
    return a[positive], b[positive], c[positive], rawNormals[positive], doubleAreas[positive]


def sampleGeometryPointCloud(
    case: geometryRecognitionCase,
    pointCount: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    if not isinstance(pointCount, Integral) or isinstance(pointCount, bool) or pointCount < 2:
        raise ValueError('pointCount must be an integer greater than one')
    pointCount = int(pointCount)
    rng = np.random.default_rng(seed)
    a, b, c, rawNormals, doubleAreas = _positiveTriangleGeometry(case)
    chosen = rng.choice(
        a.shape[0], size=pointCount, replace=True, p=doubleAreas / doubleAreas.sum()
    )
    u = rng.random(pointCount)
    v = rng.random(pointCount)
    rootU = np.sqrt(u)
    points = (
        (1.0 - rootU)[:, None] * a[chosen]
        + (rootU * (1.0 - v))[:, None] * b[chosen]
        + (rootU * v)[:, None] * c[chosen]
    )
    normals = rawNormals[chosen] / doubleAreas[chosen, None]
    if case.volumeCoordinatesMeters is not None:
        center = np.average(
            case.volumeCoordinatesMeters,
            axis=0,
            weights=case.volumeWeightsM3,
        )
    else:
        triangleCenters = (a + b + c) / 3.0
        center = np.average(triangleCenters, axis=0, weights=doubleAreas)
    centered = points - center
    scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        raise ValueError('geometry point-cloud scale must be positive and finite')
    return np.concatenate((centered / scale, normals), axis=1).astype(np.float32)


def buildGeometryPointCloudSamples(
    records: Iterable[geometryRecognitionRecord],
    *,
    pointCount: int = 2048,
    seed: int = 0,
) -> tuple[geometryPointCloudSample, ...]:
    samples = []
    seenMeshes = set()
    for index, record in enumerate(records):
        if record.meshUuid in seenMeshes:
            continue
        seenMeshes.add(record.meshUuid)
        if record.componentType is None:
            raise ValueError(f'geometry {record.geometryId!r} has no componentType')
        samples.append(
            geometryPointCloudSample(
                geometryId=record.geometryId,
                meshUuid=record.meshUuid,
                componentType=record.componentType,
                pointsAndNormals=sampleGeometryPointCloud(
                    record.load(), pointCount, seed=seed + index * 104729
                ),
            )
        )
    if not samples:
        raise ValueError('at least one geometry point cloud is required')
    return tuple(samples)


def stratifiedGeometryPointCloudSplit(
    samples: Sequence[geometryPointCloudSample],
    *,
    minimumClassCount: int = 2,
    validationPerClass: int = 1,
    seed: int = 0,
):
    if minimumClassCount < 2 or validationPerClass < 1:
        raise ValueError('split counts must be positive and minimumClassCount at least two')
    grouped: dict[str, list[geometryPointCloudSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.componentType, []).append(sample)
    rng = np.random.default_rng(seed)
    train = []
    validation = []
    excluded = []
    for label in sorted(grouped):
        items = grouped[label]
        if len(items) < minimumClassCount or len(items) <= validationPerClass:
            excluded.extend(items)
            continue
        order = rng.permutation(len(items))
        validation.extend(items[index] for index in order[:validationPerClass])
        train.extend(items[index] for index in order[validationPerClass:])
    if not train or not validation:
        raise ValueError('stratified split produced an empty train or validation set')
    trainMeshes = {sample.meshUuid for sample in train}
    validationMeshes = {sample.meshUuid for sample in validation}
    if trainMeshes & validationMeshes:
        raise ValueError('meshUuid leakage detected between train and validation sets')
    return tuple(train), tuple(validation), tuple(excluded)


def randomRotationMatrix(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = rng.random(3)
    quaternion = np.array(
        [
            np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
            np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
            np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
            np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
        ]
    )
    x, y, z, w = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


class geometryPointCloudDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[geometryPointCloudSample],
        classNames: Sequence[str],
        *,
        pointCount: int = 1024,
        repeatFactor: int = 1,
        augment: bool = False,
        rotationAugmentation: bool = True,
        jitterStandardDeviation: float = 0.005,
        seed: int = 0,
    ):
        self.samples = tuple(samples)
        self.classNames = tuple(classNames)
        self.labelToIndex = {label: index for index, label in enumerate(self.classNames)}
        if not self.samples or len(self.labelToIndex) != len(self.classNames):
            raise ValueError('point-cloud dataset samples and unique class names are required')
        if any(sample.componentType not in self.labelToIndex for sample in self.samples):
            raise ValueError('point-cloud sample contains a component type outside classNames')
        if not isinstance(pointCount, Integral) or pointCount < 2:
            raise ValueError('pointCount must be an integer greater than one')
        if any(sample.pointsAndNormals.shape[0] < pointCount for sample in self.samples):
            raise ValueError('pointCount exceeds a cached geometry point cloud')
        if not isinstance(repeatFactor, Integral) or repeatFactor < 1:
            raise ValueError('repeatFactor must be a positive integer')
        if jitterStandardDeviation < 0.0:
            raise ValueError('jitterStandardDeviation cannot be negative')
        self.pointCount = int(pointCount)
        self.repeatFactor = int(repeatFactor)
        self.augment = bool(augment)
        self.rotationAugmentation = bool(rotationAugmentation)
        self.jitterStandardDeviation = float(jitterStandardDeviation)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples) * self.repeatFactor

    def __getitem__(self, index: int):
        sample = self.samples[index % len(self.samples)]
        source = sample.pointsAndNormals
        if self.augment:
            selected = self.rng.choice(source.shape[0], self.pointCount, replace=False)
        else:
            selected = np.linspace(0, source.shape[0] - 1, self.pointCount, dtype=np.int64)
        values = source[selected].copy()
        if self.augment and self.rotationAugmentation:
            rotation = randomRotationMatrix(self.rng)
            values[:, :3] = values[:, :3] @ rotation.T
            values[:, 3:] = values[:, 3:] @ rotation.T
        if self.augment and self.jitterStandardDeviation > 0.0:
            jitter = self.rng.normal(
                0.0, self.jitterStandardDeviation, size=values[:, :3].shape
            )
            values[:, :3] += np.clip(
                jitter, -4.0 * self.jitterStandardDeviation, 4.0 * self.jitterStandardDeviation
            )
        return (
            torch.from_numpy(values.astype(np.float32, copy=False)),
            torch.tensor(self.labelToIndex[sample.componentType], dtype=torch.long),
            sample.meshUuid,
        )

    def sampleWeights(self) -> torch.Tensor:
        counts = Counter(sample.componentType for sample in self.samples)
        weights = [
            1.0 / counts[self.samples[index % len(self.samples)].componentType]
            for index in range(len(self))
        ]
        return torch.tensor(weights, dtype=torch.double)
