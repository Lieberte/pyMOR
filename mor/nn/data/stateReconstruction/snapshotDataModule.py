from typing import Any
from mor.nn.data.baseDataModule import baseDataModule
from mor.nn.registry import nnRegistry

class snapshotDataModule(baseDataModule):
    def __init__(self, trainInputs: Any, validationInputs: Any, trainTargets: Any | None = None, validationTargets: Any | None = None, **kwargs):
        super().__init__(dataName='snapshotDataModule', **kwargs)
        self.trainInputs = trainInputs
        self.validationInputs = validationInputs
        self.trainTargets = trainTargets if trainTargets is not None else trainInputs
        self.validationTargets = validationTargets if validationTargets is not None else validationInputs

    def getTrainData(self) -> Any:
        return self.trainInputs, self.trainTargets

    def getValidationData(self) -> Any:
        return self.validationInputs, self.validationTargets

nnRegistry.register('data.stateReconstruction', 'snapshotDataModule', snapshotDataModule)
