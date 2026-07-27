from abc import ABC, abstractmethod
from typing import Any, Optional
from mor.backends import backendRegistry

class modelBase(ABC):
    def __init__(self, backendName: str | None = None):
        self.localBackend = backendRegistry.get(backendName)
        self._backendName = self.localBackend.name

    @property
    def backendName(self) -> str | None:
        return self._backendName

    @abstractmethod
    def toNumpy(self) -> Any:
        pass
