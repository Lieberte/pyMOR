from abc import ABC, abstractmethod
from typing import Tuple, Any

from mor.backends import backendRegistry

class operatorBase(ABC):
    def __init__(self, backendName: str | None = None):
        self.localBackend = backendRegistry.get(backendName)
        self._backendName = backendName

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def dtype(self) -> Any:
        pass

    @property
    @abstractmethod
    def isSparse(self) -> bool:
        pass

    @abstractmethod
    def apply(self, x: Any, trans: bool = False) -> Any:
        pass

    @property
    def backendName(self) -> str | None:
        return self._backendName
