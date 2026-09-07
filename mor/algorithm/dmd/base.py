from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import numpy as np

from mor.backends import backendRegistry
from mor.operators.matrix import matrixOperator
from mor.operators.operatorsBase import operatorBase


@dataclass
class dmdResult:
    modes: matrixOperator
    eigenvalues: Any
    amplitudes: Any
    reducedOperator: matrixOperator
    singularValues: Any
    rank: int
    dt: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    trainingSteps: int | None = None
    inputWasReal: bool = False

    @property
    def reduced_operator(self) -> matrixOperator:
        return self.reducedOperator

    @property
    def singular_values(self) -> Any:
        return self.singularValues

    @property
    def continuousEigenvalues(self) -> Any:
        if self.dt is None:
            raise ValueError("dt is required to compute continuous-time eigenvalues.")
        eigenvalues = self.modes.localBackend.array.toNumpy(self.eigenvalues)
        if np.any(eigenvalues == 0):
            raise ValueError("Zero discrete-time eigenvalues have no finite continuous-time rate.")
        rates = np.log(eigenvalues.astype(complex)) / self.dt
        return self._fromNumpy(rates)

    @property
    def continuous_eigenvalues(self) -> Any:
        return self.continuousEigenvalues

    def reconstruct(self, realOutput: bool | None = None) -> Any:
        if self.trainingSteps is None:
            raise ValueError("Explicit paired snapshots require prediction steps for reconstruction.")
        return self.predict(steps=self.trainingSteps, realOutput=realOutput)

    def predict(
        self,
        steps: int | Any | None = None,
        times: Any | None = None,
        realOutput: bool | None = None,
    ) -> Any:
        if (steps is None) == (times is None):
            raise ValueError("Provide exactly one of steps or times.")
        backend = self.modes.localBackend
        eigenvalues = backend.array.toNumpy(self.eigenvalues)
        amplitudes = backend.array.toNumpy(self.amplitudes)
        modes = self.modes.toNumpy()
        if times is not None:
            if self.dt is None:
                raise ValueError("dt is required for prediction at physical times.")
            coordinates = np.asarray(times, dtype=float)
            if coordinates.ndim != 1 or not np.all(np.isfinite(coordinates)):
                raise ValueError("times must be a finite one-dimensional array.")
            rates = backend.array.toNumpy(self.continuousEigenvalues)
            dynamics = np.exp(rates[:, None] * coordinates[None, :])
        else:
            coordinates = self._validateSteps(steps)
            dynamics = eigenvalues[:, None] ** coordinates[None, :]
        prediction = modes @ (amplitudes[:, None] * dynamics)
        prediction = self._realIfAppropriate(prediction, realOutput)
        return self._fromNumpy(prediction)

    def _validateSteps(self, steps: int | Any) -> np.ndarray:
        if isinstance(steps, Integral) and not isinstance(steps, bool):
            if steps < 1:
                raise ValueError("steps must be a positive integer or a non-negative integer array.")
            return np.arange(int(steps))
        coordinates = np.asarray(steps)
        if (
            coordinates.ndim != 1
            or coordinates.dtype.kind not in "iu"
            or np.any(coordinates < 0)
        ):
            raise ValueError("steps must be a positive integer or a non-negative integer array.")
        return coordinates

    def _realIfAppropriate(self, values: np.ndarray, realOutput: bool | None) -> np.ndarray:
        shouldBeReal = self.inputWasReal if realOutput is None else realOutput
        if not shouldBeReal:
            return values
        imaginaryNorm = np.linalg.norm(values.imag)
        scale = max(np.linalg.norm(values.real), 1.0)
        if imaginaryNorm > 1e-10 * scale:
            raise ValueError("Prediction has a significant imaginary component.")
        return values.real

    def _fromNumpy(self, values: np.ndarray) -> Any:
        if self.modes.backendName == 'scipy':
            return values
        reference = self.modes.toBackendData()
        import torch

        return torch.as_tensor(values, device=reference.device)


class dmd(ABC):
    def __init__(self, backendName: str | None = None, **kwargs):
        self.localBackend = backendRegistry.get(backendName)
        self.options = kwargs

    @abstractmethod
    def fit(
        self,
        snapshots: operatorBase,
        shiftedSnapshots: operatorBase | None = None,
        rank: int | None = None,
        tol: float | None = None,
        dt: float | None = None,
        **kwargs,
    ) -> dmdResult:
        pass

    def _pairedData(
        self,
        snapshots: operatorBase,
        shiftedSnapshots: operatorBase | None,
        dt: float | None,
    ) -> tuple[Any, Any, int | None, bool]:
        if not isinstance(snapshots, operatorBase):
            raise TypeError("snapshots must be an operatorBase instance.")
        if snapshots.backendName != self.localBackend.name:
            raise ValueError("Snapshot and algorithm backends must match.")
        if dt is not None:
            if not isinstance(dt, Real) or isinstance(dt, bool) or not np.isfinite(dt) or dt <= 0:
                raise ValueError("dt must be a positive finite real number.")
            dt = float(dt)
        data = snapshots.toBackendData()
        shape = getattr(data, 'shape', ())
        if len(shape) != 2:
            raise ValueError("snapshots must be a two-dimensional matrix.")
        trainingSteps = None
        if shiftedSnapshots is None:
            if shape[1] < 2:
                raise ValueError("At least two snapshots are required.")
            xData, yData = data[:, :-1], data[:, 1:]
            trainingSteps = shape[1]
        else:
            if not isinstance(shiftedSnapshots, operatorBase):
                raise TypeError("shiftedSnapshots must be an operatorBase instance.")
            if shiftedSnapshots.backendName != snapshots.backendName:
                raise ValueError("Snapshot pair backends must match.")
            yData = shiftedSnapshots.toBackendData()
            if len(getattr(yData, 'shape', ())) != 2 or yData.shape != shape:
                raise ValueError("Explicit X and Y snapshots must have identical two-dimensional shapes.")
            if shape[1] < 1:
                raise ValueError("At least one snapshot pair is required.")
            xData = data
        finiteX = self.localBackend.array.isfinite(xData)
        finiteY = self.localBackend.array.isfinite(yData)
        if not self.localBackend.array.all(finiteX) or not self.localBackend.array.all(finiteY):
            raise ValueError("Snapshots must contain only finite values.")
        inputWasReal = not self.localBackend.array.iscomplexobj(data)
        return xData, yData, trainingSteps, inputWasReal
