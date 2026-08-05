from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _featureMatrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f'{name} must have shape (n, d) with n, d > 0')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must contain finite values')
    return array


def _positiveWeights(value: Any, count: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != count or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f'{name} must contain {count} positive finite values')
    return array


@dataclass(frozen=True)
class thermalOperatorCase:
    caseId: str
    geometryId: str
    volumeFeatures: Any
    volumeWeights: Any
    boundaryFeatures: Any
    boundaryWeights: Any
    globalFeatures: Any
    queryCoordinates: Any
    queryTemperatures: Any
    queryWeights: Any | None = None

    def __post_init__(self):
        caseId = str(self.caseId).strip()
        geometryId = str(self.geometryId).strip()
        if not caseId or not geometryId:
            raise ValueError('caseId and geometryId must be non-empty strings')
        volume = _featureMatrix(self.volumeFeatures, 'volumeFeatures')
        boundary = _featureMatrix(self.boundaryFeatures, 'boundaryFeatures')
        queries = _featureMatrix(self.queryCoordinates, 'queryCoordinates')
        globals_ = np.asarray(self.globalFeatures, dtype=np.float32).reshape(-1)
        temperatures = np.asarray(self.queryTemperatures, dtype=np.float32)
        if globals_.size == 0 or not np.all(np.isfinite(globals_)):
            raise ValueError('globalFeatures must contain finite values')
        if temperatures.ndim == 1:
            temperatures = temperatures[:, None]
        if temperatures.shape != (queries.shape[0], 1) or not np.all(np.isfinite(temperatures)):
            raise ValueError('queryTemperatures must have shape (number of queries, 1)')
        queryWeights = (
            np.ones(queries.shape[0], dtype=np.float32)
            if self.queryWeights is None
            else _positiveWeights(self.queryWeights, queries.shape[0], 'queryWeights')
        )
        object.__setattr__(self, 'caseId', caseId)
        object.__setattr__(self, 'geometryId', geometryId)
        object.__setattr__(self, 'volumeFeatures', volume)
        object.__setattr__(
            self,
            'volumeWeights',
            _positiveWeights(self.volumeWeights, volume.shape[0], 'volumeWeights'),
        )
        object.__setattr__(self, 'boundaryFeatures', boundary)
        object.__setattr__(
            self,
            'boundaryWeights',
            _positiveWeights(self.boundaryWeights, boundary.shape[0], 'boundaryWeights'),
        )
        object.__setattr__(self, 'globalFeatures', globals_)
        object.__setattr__(self, 'queryCoordinates', queries)
        object.__setattr__(self, 'queryTemperatures', temperatures)
        object.__setattr__(self, 'queryWeights', queryWeights)

    @property
    def volumeFeatureDim(self) -> int:
        return int(self.volumeFeatures.shape[1])

    @property
    def boundaryFeatureDim(self) -> int:
        return int(self.boundaryFeatures.shape[1])

    @property
    def globalFeatureDim(self) -> int:
        return int(self.globalFeatures.size)

    @property
    def queryDim(self) -> int:
        return int(self.queryCoordinates.shape[1])
