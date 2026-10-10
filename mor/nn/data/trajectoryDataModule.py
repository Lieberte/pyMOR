import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Any
from mor.nn.data.baseDataModule import baseDataModule
from mor.nn.registry import nnRegistry
from mor.nn.data.snapshotDataModule import snapshotDataModule

class trajectoryDataModule(baseDataModule):
    def __init__(self, trainInputs: Any, validationInputs: Any, trainTargets: Any, validationTargets: Any, batchSize: int = 32, shuffle: bool = True, **kwargs):
        super().__init__(dataName='trajectoryDataModule', **kwargs)
        self.batchSize, self.shuffle = batchSize, shuffle
        self.trainInputs, self.trainTargets = snapshotDataModule._preparePair(trainInputs, trainTargets)
        self.validationInputs, self.validationTargets = snapshotDataModule._preparePair(validationInputs, validationTargets)
        self.trainDataset = TensorDataset(self.trainInputs, self.trainTargets)
        self.validationDataset = TensorDataset(self.validationInputs, self.validationTargets)

    def getTrainData(self) -> Any: return self.trainInputs, self.trainTargets
    def getValidationData(self) -> Any: return self.validationInputs, self.validationTargets
    def getTrainLoader(self) -> DataLoader: return DataLoader(self.trainDataset, batch_size=self.batchSize, shuffle=self.shuffle)
    def getValidationLoader(self) -> DataLoader: return DataLoader(self.validationDataset, batch_size=self.batchSize, shuffle=False)

nnRegistry.register('data', 'trajectoryDataModule', trajectoryDataModule)
