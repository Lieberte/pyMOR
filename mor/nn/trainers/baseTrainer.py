from abc import ABC, abstractmethod
from typing import Any

class baseTrainer(ABC):
    def __init__(self, trainerName: str = 'baseTrainer', **kwargs):
        self.trainerName = trainerName
        self.options = kwargs

    @abstractmethod
    def fit(self, model: Any, dataModule: Any) -> dict:
        pass
