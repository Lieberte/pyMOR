from abc import ABC, abstractmethod
from typing import Any


class svdBase(ABC):

    def __init__(self, backendName: str = 'numpy'):
        self.setBackend(backendName)

    @abstractmethod
    def decompose(self, xOperator, **kwargs) -> tuple[Any, Any, Any]:
        pass

    def setBackend(self, backendName: str):
        from mor.backends import backendRegistry
        self.backend = backendRegistry.get(backendName)
        self._backendName = backendName

    @property
    def backendName(self) -> str:
        return self._backendName