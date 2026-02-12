from abc import ABC, abstractmethod
from typing import Tuple, Any


class operatorBase(ABC):

    def __init__(self, backendName: str = 'numpy'):
        from mor.backends import backendRegistry
        self.backend = backendRegistry.get(backendName)
        self._backendName = backendName

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def dtype(self):
        pass

    @property
    @abstractmethod
    def format(self) -> str:
        pass

    @abstractmethod
    def apply(self, x: Any) -> Any:
        pass

    @abstractmethod
    def svd(self, fullMatrices: bool = False, rank: int = None) -> Tuple[Any, Any, Any]:
        pass

    @property
    def isSparse(self) -> bool:
        return self.format == 'sparse'

    @property
    def isDense(self) -> bool:
        return self.format == 'dense'

    @property
    def backendName(self) -> str:
        return self._backendName