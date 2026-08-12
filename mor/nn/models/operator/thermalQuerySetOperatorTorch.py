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
        includeRelativeKernelFeatures: bool = False,
        separateBoundaryTypeContexts: bool = False,
        boundaryTypeFeatureStart: int = 9,
        boundaryTypeCount: int = 0,
        decomposeMeanAndShape: bool = False,
        anchorFeatureIndex: int = -1,
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
        boundaryTypeFeatureStart = int(boundaryTypeFeatureStart)
        boundaryTypeCount = int(boundaryTypeCount)
        if boundaryTypeFeatureStart < 0 or boundaryTypeCount < 0:
            raise ValueError('boundary type feature layout must be non-negative')
        if separateBoundaryTypeContexts and boundaryTypeCount <= 0:
            raise ValueError(
                'separateBoundaryTypeContexts requires a positive boundaryTypeCount'
            )
        if (
            separateBoundaryTypeContexts
            and boundaryTypeFeatureStart + boundaryTypeCount > boundaryFeatureDim
        ):
            raise ValueError('boundary type one-hot features exceed boundaryFeatureDim')
        self.queryDim = int(queryDim)
        self.hiddenDim = int(hiddenDim)
        self.kernelScales = scales
        self.queryFourierFrequencies = int(queryFourierFrequencies)
        self.includeRelativeKernelFeatures = bool(includeRelativeKernelFeatures)
        self.separateBoundaryTypeContexts = bool(separateBoundaryTypeContexts)
        self.boundaryTypeFeatureStart = boundaryTypeFeatureStart
        self.boundaryTypeCount = boundaryTypeCount
        self.decomposeMeanAndShape = bool(decomposeMeanAndShape)
        self.anchorFeatureIndex = int(anchorFeatureIndex)
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
        kernelContextDim = hiddenDim + (
            self.queryDim + 1 if self.includeRelativeKernelFeatures else 0
        )
        boundaryContextGroups = (
            1 + self.boundaryTypeCount
            if self.separateBoundaryTypeContexts
            else 1
        )
        decoderInputDim = (
            queryFeatureDim
            + 2 * hiddenDim
            + len(self.kernelScales) * kernelContextDim
            + boundaryContextGroups * len(self.kernelScales) * kernelContextDim
        )
        self.decoder = nn.Sequential(
            nn.Linear(decoderInputDim, 4 * hiddenDim),
            nn.SiLU(),
            nn.Linear(4 * hiddenDim, 2 * hiddenDim),
            nn.SiLU(),
            nn.Linear(2 * hiddenDim, 1),
        )
        self.meanHead = (
            nn.Sequential(
                nn.Linear(2 * hiddenDim, hiddenDim),
                nn.SiLU(),
                nn.Linear(hiddenDim, 1),
            )
            if self.decomposeMeanAndShape
            else None
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
        relativeCoordinates = (
            pointCoordinates.unsqueeze(1) - queryCoordinates.unsqueeze(2)
        )
        validWeights = weights.unsqueeze(1) * mask.unsqueeze(1).to(dtype=weights.dtype)
        contexts = []
        epsilon = torch.finfo(weights.dtype).eps
        for scale in self.kernelScales:
            kernelWeights = torch.exp(-distanceSquared / (2.0 * scale**2)) * validWeights
            denominator = kernelWeights.sum(dim=-1, keepdim=True).clamp_min(epsilon)
            contexts.append(torch.einsum('bqn,bnh->bqh', kernelWeights, encoded) / denominator)
            if self.includeRelativeKernelFeatures:
                contexts.append(
                    torch.einsum(
                        'bqn,bqnd->bqd', kernelWeights, relativeCoordinates
                    )
                    / denominator
                )
                contexts.append(
                    (
                        kernelWeights * torch.sqrt(distanceSquared.clamp_min(0.0))
                    ).sum(dim=-1, keepdim=True)
                    / denominator
                )
        return torch.cat(contexts, dim=-1)

    def _boundaryContext(
        self,
        batch: dict[str, Any],
        boundaryEncoded: torch.Tensor,
    ) -> torch.Tensor:
        contexts = [
            self._kernelAggregate(
                batch['queryCoordinates'],
                batch['boundaryFeatures'][..., : self.queryDim],
                boundaryEncoded,
                batch['boundaryWeights'],
                batch['boundaryMask'],
            )
        ]
        if self.separateBoundaryTypeContexts:
            oneHot = batch['boundaryFeatures'][
                ...,
                self.boundaryTypeFeatureStart : (
                    self.boundaryTypeFeatureStart + self.boundaryTypeCount
                ),
            ]
            for boundaryType in range(self.boundaryTypeCount):
                typeMask = batch['boundaryMask'] & (oneHot[..., boundaryType] > 0.5)
                contexts.append(
                    self._kernelAggregate(
                        batch['queryCoordinates'],
                        batch['boundaryFeatures'][..., : self.queryDim],
                        boundaryEncoded,
                        batch['boundaryWeights'],
                        typeMask,
                    )
                )
        return torch.cat(contexts, dim=-1)

    def _globalState(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return globalState, volumeEncoded, boundaryEncoded

    def forwardMean(self, batch: dict[str, Any]) -> torch.Tensor:
        if self.meanHead is None:
            raise RuntimeError('forwardMean requires decomposeMeanAndShape=True')
        globalState, _, _ = self._globalState(batch)
        return self.meanHead(globalState)

    def _forwardParts(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        globalState, volumeEncoded, boundaryEncoded = self._globalState(batch)
        queryCoordinates = batch['queryCoordinates']
        volumeContext = self._kernelAggregate(
            queryCoordinates,
            batch['volumeFeatures'][..., : self.queryDim],
            volumeEncoded,
            batch['volumeWeights'],
            batch['volumeMask'],
        )
        boundaryContext = self._boundaryContext(batch, boundaryEncoded)
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
        shape = self.decoder(decoderInput)
        if self.meanHead is None:
            return {
                'prediction': shape,
                'shape': shape,
                'meanOffset': torch.zeros_like(shape[:, :1]),
                'anchorAvailable': torch.ones(
                    shape.shape[0],
                    dtype=torch.bool,
                    device=shape.device,
                ),
            }
        meanOffset = self.meanHead(globalState).unsqueeze(1)
        anchorAvailable = (
            batch['globalFeatures'][:, self.anchorFeatureIndex] > 0.5
        )
        prediction = shape + torch.where(
            anchorAvailable[:, None, None],
            meanOffset,
            torch.zeros_like(meanOffset),
        )
        return {
            'prediction': prediction,
            'shape': shape,
            'meanOffset': meanOffset,
            'anchorAvailable': anchorAvailable,
        }

    def forwardWithAux(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return self._forwardParts(batch)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return self._forwardParts(batch)['prediction']


nnRegistry.register('models', 'thermalQuerySetOperatorTorch', thermalQuerySetOperatorTorch)
