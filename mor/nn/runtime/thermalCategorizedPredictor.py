from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mor.nn.data import (
    collateThermalOperatorCases,
    thermalCheckpointDataContract,
    thermalComponentCase,
)
from mor.nn.models import (
    thermalCaseCategoryLabels,
    thermalCategorizedOperatorTorch,
)


@dataclass(frozen=True)
class thermalComponentPrediction:
    caseId: str
    componentType: str
    queryCoordinatesMeters: np.ndarray
    temperatureK: np.ndarray | None
    temperatureDifferenceK: np.ndarray
    temperatureReferenceK: float | None
    temperatureScaleK: float
    absoluteTemperatureAvailable: bool
    routing: Mapping[str, Any]


class thermalCategorizedPredictor:
    """Inference wrapper that applies the preprocessing stored in a checkpoint."""

    def __init__(
        self,
        checkpoint: Mapping[str, Any],
        *,
        device: str = 'auto',
    ):
        self.checkpoint = dict(checkpoint)
        self.dataContract = thermalCheckpointDataContract.fromCheckpoint(checkpoint)
        self.model = thermalCategorizedOperatorTorch.fromCheckpoint(checkpoint)
        self.device = self.model.toDevice(device, allowAutoFallback=False)
        self.model.eval()

    @classmethod
    def fromCheckpoint(
        cls,
        checkpoint: Mapping[str, Any] | str | PathLike[str],
        *,
        device: str = 'auto',
    ) -> 'thermalCategorizedPredictor':
        payload = (
            torch.load(Path(checkpoint), map_location='cpu', weights_only=False)
            if isinstance(checkpoint, (str, PathLike))
            else checkpoint
        )
        return cls(payload, device=device)

    def predictCases(
        self,
        cases: Sequence[thermalComponentCase],
        *,
        requireAbsoluteTemperature: bool = True,
    ) -> tuple[thermalComponentPrediction, ...]:
        """Predict a complete parent assembly and preserve input case order.

        Assembly-context checkpoints must receive all available components from
        the same parent case together. The temperature values on each input case
        are ignored when an absolute boundary anchor is available; their sample
        coordinates define the query locations.
        """
        supported = frozenset(
            str(value)
            for value in self.checkpoint.get('metadata', {}).get(
                'supportedComponentTypes', ()
            )
        )
        prepared = self.dataContract.prepareCases(cases)
        predictionCases = tuple(
            case
            for case in prepared
            if not supported or case.componentType in supported
        )
        if not predictionCases:
            raise ValueError(
                'input contains no component supported by the checkpoint: '
                f'{sorted(supported)!r}'
            )
        transformed = self.dataContract.transformCases(predictionCases)
        predictions = []
        with torch.no_grad():
            for case, dimensionless in zip(predictionCases, transformed):
                scales = dimensionless.scales
                if requireAbsoluteTemperature and not scales.absoluteTemperatureAnchorAvailable:
                    raise ValueError(
                        f'case {case.caseId!r} has no assembly-level absolute '
                        'temperature anchor; only temperature differences can be predicted'
                    )
                batch = collateThermalOperatorCases([dimensionless.operatorCase])
                labels = thermalCaseCategoryLabels(case)
                batch = {
                    name: value.to(self.device)
                    if isinstance(value, torch.Tensor)
                    else value
                    for name, value in batch.items()
                }
                normalized, routing = self.model.forwardWithRouting(batch, labels)
                normalizedValues = (
                    normalized[0, :, 0].detach().cpu().numpy().astype(np.float64)
                )
                differenceK = normalizedValues * scales.temperatureScaleK
                absoluteAvailable = scales.absoluteTemperatureAnchorAvailable
                temperatureK = (
                    differenceK + scales.temperatureReferenceK
                    if absoluteAvailable
                    else None
                )
                predictions.append(
                    thermalComponentPrediction(
                        caseId=case.caseId,
                        componentType=case.componentType,
                        queryCoordinatesMeters=np.asarray(
                            case.temperature.coordinatesMeters,
                            dtype=np.float64,
                        ).copy(),
                        temperatureK=temperatureK,
                        temperatureDifferenceK=differenceK,
                        temperatureReferenceK=(
                            scales.temperatureReferenceK
                            if absoluteAvailable
                            else None
                        ),
                        temperatureScaleK=scales.temperatureScaleK,
                        absoluteTemperatureAvailable=absoluteAvailable,
                        routing=dict(routing[0]),
                    )
                )
        return tuple(predictions)
