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
    @abstractmethod
    def T(self) -> 'operatorBase':
        pass

    def __add__(self, other: Any) -> 'operatorBase':
        from .sum import sumOperator
        return sumOperator([self, other], backendName=self.backendName)

    def __sub__(self, other: Any) -> 'operatorBase':
        from .sum import sumOperator
        return sumOperator([self, (other * -1.0)], backendName=self.backendName)

    def __mul__(self, scalar: float) -> 'operatorBase':
        # TODO: Implement scaledOperator for efficient scalar multiplication
        return self

    @property
    def backendName(self) -> str | None:
        return self._backendName
