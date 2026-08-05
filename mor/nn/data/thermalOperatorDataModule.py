from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mor.nn.registry import nnRegistry

from .baseDataModule import baseDataModule
from .thermalOperatorCase import thermalOperatorCase


class _thermalCaseDataset(Dataset):
    def __init__(self, cases: Sequence[thermalOperatorCase]):
        self.cases = tuple(cases)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> thermalOperatorCase:
        return self.cases[index]


def _paddedFeatureBatch(cases: Sequence[thermalOperatorCase], featureName: str, weightName: str):
    featureDim = int(getattr(cases[0], featureName).shape[1])
    maximumCount = max(int(getattr(case, featureName).shape[0]) for case in cases)
    features = torch.zeros((len(cases), maximumCount, featureDim), dtype=torch.float32)
    weights = torch.zeros((len(cases), maximumCount), dtype=torch.float32)
    mask = torch.zeros((len(cases), maximumCount), dtype=torch.bool)
    for batchIndex, case in enumerate(cases):
        caseFeatures = torch.from_numpy(getattr(case, featureName))
        count = caseFeatures.shape[0]
        features[batchIndex, :count] = caseFeatures
        weights[batchIndex, :count] = torch.from_numpy(getattr(case, weightName))
        mask[batchIndex, :count] = True
    return features, weights, mask


def collateThermalOperatorCases(cases: Sequence[thermalOperatorCase]) -> dict[str, Any]:
    if not cases:
        raise ValueError('cannot collate an empty thermal operator batch')
    volumeFeatures, volumeWeights, volumeMask = _paddedFeatureBatch(
        cases, 'volumeFeatures', 'volumeWeights'
    )
    boundaryFeatures, boundaryWeights, boundaryMask = _paddedFeatureBatch(
        cases, 'boundaryFeatures', 'boundaryWeights'
    )
    queryFeatures, queryWeights, queryMask = _paddedFeatureBatch(
        cases, 'queryCoordinates', 'queryWeights'
    )
    maximumQueryCount = queryFeatures.shape[1]
    targets = torch.zeros((len(cases), maximumQueryCount, 1), dtype=torch.float32)
    for batchIndex, case in enumerate(cases):
        count = case.queryTemperatures.shape[0]
        targets[batchIndex, :count] = torch.from_numpy(case.queryTemperatures)
    return {
        'volumeFeatures': volumeFeatures,
        'volumeWeights': volumeWeights,
        'volumeMask': volumeMask,
        'boundaryFeatures': boundaryFeatures,
        'boundaryWeights': boundaryWeights,
        'boundaryMask': boundaryMask,
        'globalFeatures': torch.stack(
            [torch.from_numpy(case.globalFeatures) for case in cases], dim=0
        ),
        'queryCoordinates': queryFeatures,
        'queryWeights': queryWeights,
        'queryMask': queryMask,
        'targets': targets,
        'caseIds': tuple(case.caseId for case in cases),
        'geometryIds': tuple(case.geometryId for case in cases),
    }


class thermalOperatorDataModule(baseDataModule):
    def __init__(
        self,
        cases: Sequence[thermalOperatorCase],
        *,
        validationGeometryIds: Sequence[str],
        batchSize: int = 4,
        shuffle: bool = True,
        randomSeed: int = 0,
        numWorkers: int = 0,
        **kwargs,
    ):
        super().__init__(dataName='thermalOperatorDataModule', **kwargs)
        resolvedCases = tuple(cases)
        if not resolvedCases:
            raise ValueError('thermal operator data requires at least one case')
        self._validateDimensions(resolvedCases)
        validationIds = frozenset(str(value) for value in validationGeometryIds)
        self.trainCases = tuple(case for case in resolvedCases if case.geometryId not in validationIds)
        self.validationCases = tuple(case for case in resolvedCases if case.geometryId in validationIds)
        if not self.trainCases or not self.validationCases:
            raise ValueError('geometry split must leave at least one train and one validation case')
        self.batchSize = int(batchSize)
        self.shuffle = bool(shuffle)
        self.randomSeed = int(randomSeed)
        self.numWorkers = int(numWorkers)
        if self.batchSize <= 0 or self.numWorkers < 0:
            raise ValueError('batchSize must be positive and numWorkers non-negative')
        self.trainDataset = _thermalCaseDataset(self.trainCases)
        self.validationDataset = _thermalCaseDataset(self.validationCases)

    @staticmethod
    def _validateDimensions(cases: Sequence[thermalOperatorCase]) -> None:
        dimensions = {
            (
                case.volumeFeatureDim,
                case.boundaryFeatureDim,
                case.globalFeatureDim,
                case.queryDim,
            )
            for case in cases
        }
        if len(dimensions) != 1:
            raise ValueError('all thermal operator cases must use the same feature dimensions')

    @property
    def featureDimensions(self) -> dict[str, int]:
        case = self.trainCases[0]
        return {
            'volume': case.volumeFeatureDim,
            'boundary': case.boundaryFeatureDim,
            'global': case.globalFeatureDim,
            'query': case.queryDim,
        }

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        generator = torch.Generator().manual_seed(self.randomSeed)
        return DataLoader(
            dataset,
            batch_size=self.batchSize,
            shuffle=shuffle,
            num_workers=self.numWorkers,
            collate_fn=collateThermalOperatorCases,
            generator=generator,
        )

    def getTrainData(self) -> tuple[thermalOperatorCase, ...]:
        return self.trainCases

    def getValidationData(self) -> tuple[thermalOperatorCase, ...]:
        return self.validationCases

    def getTrainLoader(self) -> DataLoader:
        return self._loader(self.trainDataset, self.shuffle)

    def getValidationLoader(self) -> DataLoader:
        return self._loader(self.validationDataset, False)


nnRegistry.register('data', 'thermalOperatorDataModule', thermalOperatorDataModule)
