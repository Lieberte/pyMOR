from abc import ABC, abstractmethod
from typing import Any

class baseHpo(ABC):
    def __init__(self, hpoName: str = 'baseHpo', **kwargs):
        self.hpoName = hpoName
        self.options = kwargs

    @abstractmethod
    def optimize(self, objective: Any) -> dict:
        pass
