from dataclasses import dataclass, field
from typing import Any

from mor.operators.operatorsBase import operatorBase


@dataclass
class motorThermalProblem:
    capacity: operatorBase
    conductivity: operatorBase
    inputMap: operatorBase
    outputMap: operatorBase
    feedthrough: operatorBase | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        operators = {
            "capacity": self.capacity,
            "conductivity": self.conductivity,
            "inputMap": self.inputMap,
            "outputMap": self.outputMap,
            "feedthrough": self.feedthrough,
        }
        for name, operator in operators.items():
            if operator is not None and not isinstance(operator, operatorBase):
                raise TypeError(f"{name} must be an operatorBase")

        n = self.capacity.shape[0]
        if self.capacity.shape != (n, n):
            raise ValueError("capacity must be square")
        if self.conductivity.shape != (n, n):
            raise ValueError("conductivity must have the same shape as capacity")
        if len(self.inputMap.shape) not in (1, 2) or self.inputMap.shape[0] != n:
            raise ValueError("inputMap must have one row per thermal state")
        if len(self.outputMap.shape) != 2 or self.outputMap.shape[1] != n:
            raise ValueError("outputMap must have one column per thermal state")
        if self.feedthrough is not None and self.feedthrough.shape != (self.outputDim, self.inputDim):
            raise ValueError(f"feedthrough must have shape {(self.outputDim, self.inputDim)}")

        for name, operator in operators.items():
            if operator is not None and operator.backendName != self.backendName:
                raise ValueError(f"{name} backend {operator.backendName} does not match {self.backendName}")

    @property
    def backendName(self) -> str:
        return self.capacity.backendName

    @property
    def stateDim(self) -> int:
        return self.capacity.shape[0]

    @property
    def inputDim(self) -> int:
        return self.inputMap.shape[1] if len(self.inputMap.shape) == 2 else 1

    @property
    def outputDim(self) -> int:
        return self.outputMap.shape[0]
