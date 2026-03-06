from abc import ABC, abstractmethod
from typing import Any

from mor.backends import backendRegistry

class svd(ABC):
    def __init__(self, backendName: str | None = None, **kwargs):
        self.localBackend = backendRegistry.get(backendName)
        self.options = kwargs

    @abstractmethod
    def decompose(self, xOperator: Any, **kwargs) -> tuple[Any, Any, Any]:
        pass
