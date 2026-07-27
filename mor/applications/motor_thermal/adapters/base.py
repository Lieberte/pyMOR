from typing import Any, Protocol, runtime_checkable

from mor.operators.operatorsBase import operatorBase

from ..problem import motorThermalProblem


@runtime_checkable
class motorThermalAdapter(Protocol):
    def loadProblem(self, source: Any) -> motorThermalProblem: ...

    def loadSnapshots(self, source: Any) -> operatorBase: ...
