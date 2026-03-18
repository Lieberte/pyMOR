from abc import ABC, abstractmethod
from typing import Any

class baseLoss(ABC):
    def __init__(self, lossName: str = 'baseLoss', **kwargs):
        self.lossName = lossName
        self.options = kwargs

    @abstractmethod
    def compute(self, predictions: Any, targets: Any) -> Any:
        pass
