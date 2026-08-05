from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from mor.nn.models.torchModelBase import torchModelBase
from mor.nn.registry import nnRegistry


def _feedForward(hiddenDim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hiddenDim, 2 * hiddenDim),
        nn.GELU(),
        nn.Linear(2 * hiddenDim, hiddenDim),
    )


class thermalSetAttentionOperatorTorch(torchModelBase):
    def __init__(
        self,
        *,
        volumeFeatureDim: int,
        boundaryFeatureDim: int,
        globalFeatureDim: int,
        queryDim: int = 3,
        hiddenDim: int = 64,
        latentCount: int = 8,
        attentionHeads: int = 4,
        queryFourierFrequencies: int = 1,
        **kwargs,
    ):
        super().__init__(modelName='thermalSetAttentionOperatorTorch', **kwargs)
        values = (
            volumeFeatureDim,
            boundaryFeatureDim,
            globalFeatureDim,
            queryDim,
            hiddenDim,
            latentCount,
            attentionHeads,
        )
        if min(int(value) for value in values) <= 0:
            raise ValueError('thermal attention operator dimensions must be positive')
        if hiddenDim % attentionHeads != 0:
            raise ValueError('hiddenDim must be divisible by attentionHeads')
        if queryFourierFrequencies < 0:
            raise ValueError('queryFourierFrequencies must be non-negative')
        self.queryDim = int(queryDim)
        self.hiddenDim = int(hiddenDim)
        self.queryFourierFrequencies = int(queryFourierFrequencies)
        self.volumeEncoder = nn.Sequential(
            nn.Linear(volumeFeatureDim, hiddenDim), nn.GELU(), nn.Linear(hiddenDim, hiddenDim)
        )
        self.boundaryEncoder = nn.Sequential(
            nn.Linear(boundaryFeatureDim, hiddenDim), nn.GELU(), nn.Linear(hiddenDim, hiddenDim)
        )
        self.measureEncoder = nn.Sequential(
            nn.Linear(1, hiddenDim), nn.GELU(), nn.Linear(hiddenDim, hiddenDim)
        )
        self.typeEmbedding = nn.Parameter(torch.randn(2, hiddenDim) * 0.02)
        self.globalEncoder = nn.Sequential(
            nn.Linear(globalFeatureDim + 2, hiddenDim),
            nn.GELU(),
            nn.Linear(hiddenDim, hiddenDim),
        )
        self.latentTokens = nn.Parameter(torch.randn(latentCount, hiddenDim) * 0.02)
        self.latentAttention = nn.MultiheadAttention(
            hiddenDim, attentionHeads, batch_first=True
        )
        self.latentNorm1 = nn.LayerNorm(hiddenDim)
        self.latentFeedForward = _feedForward(hiddenDim)
        self.latentNorm2 = nn.LayerNorm(hiddenDim)
        queryInputDim = queryDim * (1 + 2 * queryFourierFrequencies)
        self.queryEncoder = nn.Sequential(
            nn.Linear(queryInputDim, hiddenDim), nn.GELU(), nn.Linear(hiddenDim, hiddenDim)
        )
        self.queryAttention = nn.MultiheadAttention(
            hiddenDim, attentionHeads, batch_first=True
        )
        self.queryNorm1 = nn.LayerNorm(hiddenDim)
        self.queryFeedForward = _feedForward(hiddenDim)
        self.queryNorm2 = nn.LayerNorm(hiddenDim)
        self.output = nn.Sequential(
            nn.Linear(hiddenDim, hiddenDim), nn.GELU(), nn.Linear(hiddenDim, 1)
        )

    def _queryFeatures(self, coordinates: torch.Tensor) -> torch.Tensor:
        if self.queryFourierFrequencies == 0:
            return coordinates
        parts = [coordinates]
        for frequencyIndex in range(self.queryFourierFrequencies):
            frequency = (2.0 ** frequencyIndex) * math.pi
            parts.extend((torch.sin(frequency * coordinates), torch.cos(frequency * coordinates)))
        return torch.cat(parts, dim=-1)

    def _measureEmbedding(self, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = weights * mask.to(dtype=weights.dtype)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1)
        meanWeight = valid.sum(dim=1, keepdim=True) / count
        relative = valid / meanWeight.clamp_min(torch.finfo(weights.dtype).eps)
        return self.measureEncoder(torch.log1p(relative).unsqueeze(-1))

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        volume = self.volumeEncoder(batch['volumeFeatures'])
        volume = volume + self._measureEmbedding(
            batch['volumeWeights'], batch['volumeMask']
        ) + self.typeEmbedding[0]
        boundary = self.boundaryEncoder(batch['boundaryFeatures'])
        boundary = boundary + self._measureEmbedding(
            batch['boundaryWeights'], batch['boundaryMask']
        ) + self.typeEmbedding[1]
        context = torch.cat((volume, boundary), dim=1)
        contextMask = torch.cat((batch['volumeMask'], batch['boundaryMask']), dim=1)
        measureTotals = torch.stack(
            (
                torch.log1p(
                    (batch['volumeWeights'] * batch['volumeMask']).sum(dim=1)
                ),
                torch.log1p(
                    (batch['boundaryWeights'] * batch['boundaryMask']).sum(dim=1)
                ),
            ),
            dim=1,
        )
        globalState = self.globalEncoder(
            torch.cat((batch['globalFeatures'], measureTotals), dim=-1)
        )
        latents = self.latentTokens.unsqueeze(0).expand(context.shape[0], -1, -1)
        latents = latents + globalState.unsqueeze(1)
        attended, _ = self.latentAttention(
            latents,
            context,
            context,
            key_padding_mask=~contextMask,
            need_weights=False,
        )
        latents = self.latentNorm1(latents + attended)
        latents = self.latentNorm2(latents + self.latentFeedForward(latents))
        query = self.queryEncoder(self._queryFeatures(batch['queryCoordinates']))
        query = query + globalState.unsqueeze(1)
        queried, _ = self.queryAttention(query, latents, latents, need_weights=False)
        query = self.queryNorm1(query + queried)
        query = self.queryNorm2(query + self.queryFeedForward(query))
        return self.output(query)


nnRegistry.register(
    'models', 'thermalSetAttentionOperatorTorch', thermalSetAttentionOperatorTorch
)
