from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn

from mor.nn.models.torchModelBase import torchModelBase
from mor.nn.registry import nnRegistry


class thermalQuerySetOperatorTorch(torchModelBase):
    def __init__(
        self,
        *,
        volumeFeatureDim: int,
        boundaryFeatureDim: int,
        globalFeatureDim: int,
        queryDim: int = 3,
        hiddenDim: int = 32,
        kernelScales: Sequence[float] = (0.3,),
        queryFourierFrequencies: int = 1,
        **kwargs,
    ):
        super().__init__(modelName='thermalQuerySetOperatorTorch', **kwargs)
        if min(
            int(volumeFeatureDim),
            int(boundaryFeatureDim),
            int(globalFeatureDim),
            int(queryDim),
            int(hiddenDim),
        ) <= 0:
            raise ValueError('thermal query-set operator dimensions must be positive')
        scales = tuple(float(value) for value in kernelScales)
        if not scales or any(not math.isfinite(value) or value <= 0.0 for value in scales):
            raise ValueError('kernelScales must contain positive finite values')
        if queryFourierFrequencies < 0:
            raise ValueError('queryFourierFrequencies must be non-negative')
        self.queryDim = int(queryDim)
        self.hiddenDim = int(hiddenDim)
        self.kernelScales = scales
        self.queryFourierFrequencies = int(queryFourierFrequencies)
        self.volumeEncoder = nn.Sequential(
            nn.Linear(volumeFeatureDim, hiddenDim), nn.SiLU(), nn.Linear(hiddenDim, hiddenDim)
        )
        self.boundaryEncoder = nn.Sequential(
            nn.Linear(boundaryFeatureDim, hiddenDim), nn.SiLU(), nn.Linear(hiddenDim, hiddenDim)
        )
        self.globalEncoder = nn.Sequential(
            nn.Linear(globalFeatureDim + 2 * hiddenDim + 2, 2 * hiddenDim),
            nn.SiLU(),
            nn.Linear(2 * hiddenDim, 2 * hiddenDim),
        )
        queryFeatureDim = self.queryDim * (1 + 2 * self.queryFourierFrequencies)
        decoderInputDim = (
            queryFeatureDim
            + 2 * hiddenDim
            + 2 * len(self.kernelScales) * hiddenDim
        )
        self.decoder = nn.Sequential(
            nn.Linear(decoderInputDim, 4 * hiddenDim),
            nn.SiLU(),
            nn.Linear(4 * hiddenDim, 2 * hiddenDim),
            nn.SiLU(),
            nn.Linear(2 * hiddenDim, 1),
        )

    def _queryFeatures(self, coordinates: torch.Tensor) -> torch.Tensor:
        if self.queryFourierFrequencies == 0:
            return coordinates
        parts = [coordinates]
        for frequencyIndex in range(self.queryFourierFrequencies):
            frequency = (2.0 ** frequencyIndex) * math.pi
            parts.extend((torch.sin(frequency * coordinates), torch.cos(frequency * coordinates)))
        return torch.cat(parts, dim=-1)

    def _weightedMean(
        self,
        encoded: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        validWeights = weights * mask.to(dtype=weights.dtype)
        total = validWeights.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )
        mean = (encoded * validWeights.unsqueeze(-1)).sum(dim=1) / total
        return mean, total

    def _kernelAggregate(
        self,
        queryCoordinates: torch.Tensor,
        pointCoordinates: torch.Tensor,
        encoded: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        distanceSquared = (
            queryCoordinates.unsqueeze(2) - pointCoordinates.unsqueeze(1)
        ).square().sum(dim=-1)
        validWeights = weights.unsqueeze(1) * mask.unsqueeze(1).to(dtype=weights.dtype)
        contexts = []
        epsilon = torch.finfo(weights.dtype).eps
        for scale in self.kernelScales:
            kernelWeights = torch.exp(-distanceSquared / (2.0 * scale**2)) * validWeights
            denominator = kernelWeights.sum(dim=-1, keepdim=True).clamp_min(epsilon)
            contexts.append(torch.einsum('bqn,bnh->bqh', kernelWeights, encoded) / denominator)
        return torch.cat(contexts, dim=-1)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        volumeEncoded = self.volumeEncoder(batch['volumeFeatures'])
        boundaryEncoded = self.boundaryEncoder(batch['boundaryFeatures'])
        volumeMean, volumeTotal = self._weightedMean(
            volumeEncoded, batch['volumeWeights'], batch['volumeMask']
        )
        boundaryMean, boundaryTotal = self._weightedMean(
            boundaryEncoded, batch['boundaryWeights'], batch['boundaryMask']
        )
        globalState = self.globalEncoder(
            torch.cat(
                (
                    batch['globalFeatures'],
                    volumeMean,
                    boundaryMean,
                    torch.log1p(volumeTotal),
                    torch.log1p(boundaryTotal),
                ),
                dim=-1,
            )
        )
        queryCoordinates = batch['queryCoordinates']
        volumeContext = self._kernelAggregate(
            queryCoordinates,
            batch['volumeFeatures'][..., : self.queryDim],
            volumeEncoded,
            batch['volumeWeights'],
            batch['volumeMask'],
        )
        boundaryContext = self._kernelAggregate(
            queryCoordinates,
            batch['boundaryFeatures'][..., : self.queryDim],
            boundaryEncoded,
            batch['boundaryWeights'],
            batch['boundaryMask'],
        )
        expandedGlobal = globalState.unsqueeze(1).expand(-1, queryCoordinates.shape[1], -1)
        decoderInput = torch.cat(
            (
                self._queryFeatures(queryCoordinates),
                expandedGlobal,
                volumeContext,
                boundaryContext,
            ),
            dim=-1,
        )
        return self.decoder(decoderInput)


nnRegistry.register('models', 'thermalQuerySetOperatorTorch', thermalQuerySetOperatorTorch)
