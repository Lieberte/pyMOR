from typing import Any
from numbers import Real
import torch
from torch.utils.data import DataLoader, TensorDataset
from mor.nn.data.baseDataModule import baseDataModule
from mor.nn.registry import nnRegistry

class snapshotDataModule(baseDataModule):
    def __init__(self, trainInputs: Any, validationInputs: Any, trainTargets: Any | None = None, validationTargets: Any | None = None, batchSize: int = 32, shuffle: bool = True, numWorkers: int = 0, pinMemory: bool = False, dropLast: bool = False, persistentWorkers: bool = False, **kwargs):
        super().__init__(dataName='snapshotDataModule', **kwargs)
        self.batchSize = batchSize
        self.shuffle = shuffle
        self.numWorkers = numWorkers
        self.pinMemory = pinMemory
        self.dropLast = dropLast
        self.persistentWorkers = persistentWorkers and numWorkers > 0
        self.trainInputs, self.trainTargets = self._preparePair(trainInputs, trainTargets)
        self.validationInputs, self.validationTargets = self._preparePair(validationInputs, validationTargets)
        self.trainDataset = TensorDataset(self.trainInputs, self.trainTargets)
        self.validationDataset = TensorDataset(self.validationInputs, self.validationTargets)

    @classmethod
    def fromSnapshots(cls, inputs: Any, targets: Any | None = None, validationSplit: float = 0.2, shuffle: bool = True, randomSeed: int | None = None, **kwargs):
        """Split samples with a ratio in [0, 1); zero evaluates on the training data."""
        inputsTensor, targetsTensor = cls._preparePair(inputs, targets)
        sampleCount = int(inputsTensor.shape[0])
        trainIndices, validationIndices = cls._splitIndices(sampleCount, validationSplit, shuffle, randomSeed)
        if trainIndices.numel() == 0:
            trainIndices = validationIndices
        trainInputs, trainTargets = cls._slicePair(inputsTensor, targetsTensor, trainIndices)
        validationInputs, validationTargets = cls._slicePair(inputsTensor, targetsTensor, validationIndices)
        if validationIndices.numel() == 0:
            validationInputs, validationTargets = trainInputs, trainTargets
        return cls(trainInputs=trainInputs, validationInputs=validationInputs, trainTargets=trainTargets, validationTargets=validationTargets, shuffle=shuffle, **kwargs)

    @staticmethod
    def _toFloatTensor(data: Any) -> torch.Tensor:
        tensor = torch.as_tensor(data)
        return tensor if tensor.dtype.is_floating_point else tensor.float()

    @classmethod
    def _preparePair(cls, inputs: Any, targets: Any | None) -> tuple[torch.Tensor, torch.Tensor]:
        inputsTensor = cls._toFloatTensor(inputs)
        targetsTensor = cls._toFloatTensor(targets if targets is not None else inputs)
        if inputsTensor.ndim == 0 or targetsTensor.ndim == 0:
            raise ValueError('inputs and targets must have a sample dimension')
        if inputsTensor.shape[0] != targetsTensor.shape[0]:
            raise ValueError('inputs and targets must have the same sample count')
        return inputsTensor, targetsTensor

    @staticmethod
    def _slicePair(inputsTensor: torch.Tensor, targetsTensor: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return inputsTensor[indices], targetsTensor[indices]

    @staticmethod
    def _splitIndices(sampleCount: int, validationSplit: float, shuffle: bool, randomSeed: int | None) -> tuple[torch.Tensor, torch.Tensor]:
        if (not isinstance(validationSplit, Real) or isinstance(validationSplit, bool)
                or not 0 <= validationSplit < 1):
            raise ValueError('validationSplit must be a real number in [0, 1)')
        validationCount = int(round(sampleCount * validationSplit))
        validationCount = (max(1, min(validationCount, sampleCount - 1))
                           if sampleCount > 1 and validationSplit > 0 else 0)
        if shuffle:
            generator = torch.Generator()
            if randomSeed is not None:
                generator.manual_seed(randomSeed)
            indices = torch.randperm(sampleCount, generator=generator)
        else:
            indices = torch.arange(sampleCount)
        validationIndices = indices[:validationCount]
        trainIndices = indices[validationCount:]
        return trainIndices, validationIndices

    def _buildLoader(self, dataset: TensorDataset, shuffle: bool, dropLast: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=self.batchSize, shuffle=shuffle, num_workers=self.numWorkers, pin_memory=self.pinMemory, drop_last=dropLast, persistent_workers=self.persistentWorkers)

    def getTrainData(self) -> Any:
        return self.trainInputs, self.trainTargets

    def getValidationData(self) -> Any:
        return self.validationInputs, self.validationTargets

    def getTrainLoader(self) -> DataLoader:
        return self._buildLoader(self.trainDataset, shuffle=self.shuffle, dropLast=self.dropLast)

    def getValidationLoader(self) -> DataLoader:
        return self._buildLoader(self.validationDataset, shuffle=False, dropLast=False)

nnRegistry.register('data', 'snapshotDataModule', snapshotDataModule)
