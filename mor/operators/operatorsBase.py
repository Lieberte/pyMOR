from abc import ABC, abstractmethod
from numbers import Number
from typing import Tuple, Any

from mor.backends import backendRegistry

class operatorBase(ABC):
    def __init__(self, backendName: str | None = None):
        self.localBackend = backendRegistry.get(backendName)
        self._backendName = self.localBackend.name

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

    @abstractmethod
    def toBackendData(self) -> Any:
        pass

    def toNumpy(self) -> Any:
        return self.localBackend.array.toNumpy(self.toBackendData())

    @property
    def operatorKind(self) -> str:
        return type(self).__name__

    @property
    def structureTags(self) -> frozenset[str]:
        tags = {'sparse'} if self.isSparse else {'dense'}
        return frozenset(tags)

    @property
    def isDiagonal(self) -> bool:
        return 'diagonal' in self.structureTags

    @property
    def isComposed(self) -> bool:
        return 'composed' in self.structureTags

    @property
    def isLowRank(self) -> bool:
        return 'low_rank' in self.structureTags

    @property
    def materializationCost(self) -> int | None:
        shape = self.shape
        if len(shape) != 2:
            return None
        return int(shape[0] * shape[1])

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            'kind': self.operatorKind,
            'shape': self.shape,
            'dtype': self.dtype,
            'backendName': self.backendName,
            'isSparse': self.isSparse,
            'structureTags': sorted(self.structureTags),
            'isDiagonal': self.isDiagonal,
            'isComposed': self.isComposed,
            'isLowRank': self.isLowRank,
            'materializationCost': self.materializationCost,
        }

    def __add__(self, other: Any) -> 'operatorBase':
        from .sum import sumOperator
        return sumOperator([self, other], backendName=self.backendName)

    def __sub__(self, other: Any) -> 'operatorBase':
        from .sum import sumOperator
        return sumOperator([self, (other * -1.0)], backendName=self.backendName)

    def __mul__(self, scalar: float) -> 'operatorBase':
        if not isinstance(scalar, Number):
            return NotImplemented
        from .scaled import scaledOperator
        return scaledOperator(self, scalar, backendName=self.backendName)

    def __rmul__(self, scalar: float) -> 'operatorBase':
        return self * scalar

    def __neg__(self) -> 'operatorBase':
        return self * -1.0

    @property
    def backendName(self) -> str | None:
        return self._backendName
