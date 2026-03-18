from abc import ABC, abstractmethod
from typing import Any

class baseDataModule(ABC):
    def __init__(self, dataName: str = 'baseDataModule', **kwargs):
        self.dataName = dataName
        self.options = kwargs

    @abstractmethod
    def getTrainData(self) -> Any:
        pass

    @abstractmethod
    def getValidationData(self) -> Any:
        pass
