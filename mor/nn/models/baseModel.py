from abc import ABC, abstractmethod
from typing import Any

class baseModel(ABC):
    def __init__(self, modelName: str = 'baseModel', **kwargs):
        self.modelName = modelName
        self.options = kwargs
        self.deviceName = kwargs.get('deviceName', 'cpu')
        self.trainingState = {
            'currentEpoch': 0,
            'currentStep': 0,
            'isTraining': False
        }

    @abstractmethod
    def forward(self, inputs: Any) -> Any:
        pass

    def toDevice(self, deviceName: str) -> str:
        self.deviceName = deviceName
        return self.deviceName

    def trainMode(self):
        self.trainingState['isTraining'] = True

    def evalMode(self):
        self.trainingState['isTraining'] = False

    def saveState(self, **kwargs) -> dict:
        raise NotImplementedError()

    def loadState(self, state: dict, **kwargs):
        raise NotImplementedError()

    def updateTrainingState(self, **kwargs):
        self.trainingState.update(kwargs)

    def getTrainingState(self) -> dict:
        return dict(self.trainingState)
