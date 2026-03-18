from abc import ABC, abstractmethod
from typing import Any

class baseValidation(ABC):
    def __init__(self, validationName: str = 'baseValidation', **kwargs):
        self.validationName = validationName
        self.options = kwargs

    @abstractmethod
    def evaluate(self, model: Any, dataModule: Any) -> dict:
        pass
