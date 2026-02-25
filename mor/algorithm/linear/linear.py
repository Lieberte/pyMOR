from abc import ABC, abstractmethod
from typing import Any
from mor.backends import backendRegistry

class linearAlgorithm(ABC):
    def __init__(self, backendName: str | None = None, **kwargs):
        self.localBackend = backendRegistry.get(backendName)
        self.options = kwargs

    @abstractmethod
    def solve(self, A: Any, rhs: Any, **kwargs) -> Any:
        pass
