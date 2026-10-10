from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .matrix import matrixOperator
from .operatorsBase import operatorBase


@dataclass(frozen=True)
class thermalOperatorSet:
    conductivity: matrixOperator
    capacity: matrixOperator
    load: matrixOperator
    elementVolumes: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def nodeCount(self) -> int:
        return int(self.conductivity.shape[0])

    @property
    def elementCount(self) -> int:
        return int(self.elementVolumes.size)

    @property
    def K(self) -> matrixOperator:
        return self.conductivity

    @property
    def M(self) -> matrixOperator:
        return self.capacity

    @property
    def Q(self) -> matrixOperator:
        return self.load

    def steadyResidual(self, temperature: Any) -> Any:
        return self.conductivity.apply(temperature) - self.load.toBackendData()

    def transientResidual(self, temperature: Any, temperatureRate: Any) -> Any:
        return (
            self.capacity.apply(temperatureRate)
            + self.conductivity.apply(temperature)
            - self.load.toBackendData()
        )
