from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from mor.nn.models.torchModelBase import torchModelBase
from mor.nn.registry import nnRegistry


def _mlp(inputDim: int, hiddenDims: list[int], outputDim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = inputDim
    for hidden in hiddenDims:
        layers.extend((nn.Linear(current, hidden), nn.SiLU()))
        current = hidden
    layers.append(nn.Linear(current, outputDim))
    return nn.Sequential(*layers)


class thermalSetDeepONetTorch(torchModelBase):
    def __init__(
        self,
        *,
        volumeFeatureDim: int,
        boundaryFeatureDim: int,
        globalFeatureDim: int,
        queryDim: int = 3,
        pointHiddenDim: int = 64,
        globalHiddenDim: int = 32,
        latentDim: int = 96,
        poolingMode: str = 'mean_max',
        queryFourierFrequencies: int = 0,
        **kwargs,
    ):
        super().__init__(modelName='thermalSetDeepONetTorch', **kwargs)
        poolingMode = str(poolingMode).lower()
        if poolingMode not in ('mean', 'mean_max'):
            raise ValueError("poolingMode must be 'mean' or 'mean_max'")
        self.poolingMode = poolingMode
        self.queryDim = int(queryDim)
        self.queryFourierFrequencies = int(queryFourierFrequencies)
        dimensions = (
            int(volumeFeatureDim),
            int(boundaryFeatureDim),
            int(globalFeatureDim),
            self.queryDim,
            int(pointHiddenDim),
            int(globalHiddenDim),
            int(latentDim),
        )
        if min(dimensions) <= 0:
            raise ValueError('thermal operator dimensions must be positive')
        if self.queryFourierFrequencies < 0:
            raise ValueError('queryFourierFrequencies must be non-negative')
        self.volumeEncoder = _mlp(
            int(volumeFeatureDim), [pointHiddenDim, pointHiddenDim], pointHiddenDim
        )
        self.boundaryEncoder = _mlp(
            int(boundaryFeatureDim), [pointHiddenDim, pointHiddenDim], pointHiddenDim
        )
        self.globalEncoder = _mlp(
            int(globalFeatureDim), [globalHiddenDim], globalHiddenDim
        )
        pooledDim = pointHiddenDim * (2 if poolingMode == 'mean_max' else 1) + 1
        branchInputDim = 2 * pooledDim + globalHiddenDim
        self.branch = _mlp(branchInputDim, [latentDim, latentDim], latentDim + 1)
        queryInputDim = self.queryDim * (1 + 2 * self.queryFourierFrequencies)
        self.trunk = _mlp(queryInputDim, [latentDim, latentDim], latentDim)
        self.latentScale = math.sqrt(float(latentDim))

    def _pool(
        self,
        encoded: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        validWeights = weights * mask.to(dtype=weights.dtype)
        totalWeight = validWeights.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )
        mean = (encoded * validWeights.unsqueeze(-1)).sum(dim=1) / totalWeight
        parts = [mean]
        if self.poolingMode == 'mean_max':
            parts.append(encoded.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(dim=1))
        parts.append(torch.log1p(totalWeight))
        return torch.cat(parts, dim=-1)

    def _queryFeatures(self, coordinates: torch.Tensor) -> torch.Tensor:
        if self.queryFourierFrequencies == 0:
            return coordinates
        parts = [coordinates]
        for frequencyIndex in range(self.queryFourierFrequencies):
            frequency = (2.0 ** frequencyIndex) * math.pi
            parts.extend((torch.sin(frequency * coordinates), torch.cos(frequency * coordinates)))
        return torch.cat(parts, dim=-1)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        volume = self.volumeEncoder(batch['volumeFeatures'])
        boundary = self.boundaryEncoder(batch['boundaryFeatures'])
        volumeState = self._pool(volume, batch['volumeWeights'], batch['volumeMask'])
        boundaryState = self._pool(boundary, batch['boundaryWeights'], batch['boundaryMask'])
        globalState = self.globalEncoder(batch['globalFeatures'])
        branch = self.branch(torch.cat((volumeState, boundaryState, globalState), dim=-1))
        coefficients, bias = branch[:, :-1], branch[:, -1:]
        trunk = self.trunk(self._queryFeatures(batch['queryCoordinates']))
        prediction = torch.einsum('bl,bql->bq', coefficients, trunk) / self.latentScale
        return prediction.unsqueeze(-1) + bias.unsqueeze(1)


nnRegistry.register('models', 'thermalSetDeepONetTorch', thermalSetDeepONetTorch)
