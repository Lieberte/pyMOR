from abc import ABC, abstractmethod
from typing import Any

class baseModel(ABC):
    def __init__(self, modelName: str = 'baseModel', **kwargs):
        self.modelName = modelName
        self.options = kwargs

    @abstractmethod
    def forward(self, inputs: Any) -> Any:
        pass
