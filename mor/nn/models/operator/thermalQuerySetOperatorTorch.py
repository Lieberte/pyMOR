from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn

from mor.nn.models.torchModelBase import torchModelBase
from mor.nn.registry import nnRegistry


class _SmoothResidualBlock(nn.Module):
    """A twice-differentiable residual block for strong-form PINN training."""

    def __init__(self, width: int, residualScale: float):
        super().__init__()
        self.residualScale = float(residualScale)
        self.residual = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        # Start from the exact shallow-network function. The residual branch
        # then grows only when it is useful to the optimization.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.residualScale * self.residual(values)


class _SmoothUNetDecoder(nn.Module):
    """U-shaped MLP decoder for unstructured query features."""

    def __init__(self, inputDim: int, hiddenDim: int):
        super().__init__()
        width0 = 4 * hiddenDim
        width1 = 2 * hiddenDim
        width2 = hiddenDim
        self.inputProjection = nn.Linear(inputDim, width0)
        self.down1 = nn.Linear(width0, width1)
        self.bottleneck = nn.Linear(width1, width2)
        self.up1 = nn.Linear(width2, width1)
        self.merge1 = nn.Linear(2 * width1, width1)
        self.up0 = nn.Linear(width1, width0)
        self.merge0 = nn.Linear(2 * width0, width0)
        self.output = nn.Linear(width0, 1)
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        skip0 = self.activation(self.inputProjection(values))
        skip1 = self.activation(self.down1(skip0))
        hidden = self.activation(self.bottleneck(skip1))
        hidden = self.activation(self.up1(hidden))
        hidden = self.activation(self.merge1(torch.cat((hidden, skip1), dim=-1)))
        hidden = self.activation(self.up0(hidden))
        hidden = self.activation(self.merge0(torch.cat((hidden, skip0), dim=-1)))
        return self.output(hidden)


class _SmoothDenseDecoder(nn.Module):
    """Dense skip-connected MLP decoder with smooth PINN derivatives."""

    def __init__(self, inputDim: int, hiddenDim: int):
        super().__init__()
        stemWidth = 4 * hiddenDim
        growthWidth = 2 * hiddenDim
        self.stem = nn.Linear(inputDim, stemWidth)
        self.dense1 = nn.Linear(stemWidth, growthWidth)
        self.dense2 = nn.Linear(stemWidth + growthWidth, growthWidth)
        self.dense3 = nn.Linear(stemWidth + 2 * growthWidth, growthWidth)
        self.output = nn.Linear(stemWidth + 3 * growthWidth, 1)
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        stem = self.activation(self.stem(values))
        dense1 = self.activation(self.dense1(stem))
        dense2 = self.activation(self.dense2(torch.cat((stem, dense1), dim=-1)))
        dense3 = self.activation(
            self.dense3(torch.cat((stem, dense1, dense2), dim=-1))
        )
        return self.output(torch.cat((stem, dense1, dense2, dense3), dim=-1))


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
        residualBlockCount: int = 0,
        residualScale: float = 1.0,
        decoderArchitecture: str | None = None,
        contextArchitecture: str = 'rbf',
        attentionHeadCount: int = 4,
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
        residualBlockCount = int(residualBlockCount)
        residualScale = float(residualScale)
        if residualBlockCount < 0:
            raise ValueError('residualBlockCount must be non-negative')
        if not math.isfinite(residualScale) or residualScale <= 0.0:
            raise ValueError('residualScale must be positive and finite')
        if decoderArchitecture is None or str(decoderArchitecture).strip() == 'auto':
            decoderArchitecture = 'resnet' if residualBlockCount else 'mlp'
        decoderArchitecture = str(decoderArchitecture).strip().lower()
        if decoderArchitecture not in ('mlp', 'resnet', 'unet', 'dense'):
            raise ValueError(
                'decoderArchitecture must be mlp, resnet, unet, or dense'
            )
        if decoderArchitecture == 'resnet' and residualBlockCount <= 0:
            raise ValueError('resnet decoder requires residualBlockCount > 0')
        if decoderArchitecture != 'resnet' and residualBlockCount != 0:
            raise ValueError(
                'residualBlockCount is only supported by the resnet decoder'
            )
        contextArchitecture = str(contextArchitecture).strip().lower()
        if contextArchitecture not in (
            'rbf',
            'global',
            'deeponet',
            'attention',
            'hybrid',
        ):
            raise ValueError(
                'contextArchitecture must be rbf, global, deeponet, '
                'attention, or hybrid'
            )
        attentionHeadCount = int(attentionHeadCount)
        if attentionHeadCount <= 0 or int(hiddenDim) % attentionHeadCount:
            raise ValueError(
                'attentionHeadCount must be positive and divide hiddenDim'
            )
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
        self.residualBlockCount = residualBlockCount
        self.residualScale = residualScale
        self.decoderArchitecture = decoderArchitecture
        self.contextArchitecture = contextArchitecture
        self.attentionHeadCount = attentionHeadCount
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
        rbfContextDim = (
            len(self.kernelScales) * kernelContextDim
            + boundaryContextGroups * len(self.kernelScales) * kernelContextDim
        )
        attentionContextDim = 2 * hiddenDim
        decoderInputDim = queryFeatureDim + 2 * hiddenDim
        if contextArchitecture in ('rbf', 'hybrid'):
            decoderInputDim += rbfContextDim
        if contextArchitecture in ('attention', 'hybrid'):
            decoderInputDim += attentionContextDim
        self.queryAttentionProjection = None
        self.volumeAttention = None
        self.boundaryAttention = None
        if contextArchitecture in ('attention', 'hybrid'):
            self.queryAttentionProjection = nn.Sequential(
                nn.Linear(queryFeatureDim + 2 * hiddenDim, hiddenDim),
                nn.SiLU(),
                nn.Linear(hiddenDim, hiddenDim),
            )
            self.volumeAttention = nn.MultiheadAttention(
                hiddenDim,
                attentionHeadCount,
                batch_first=True,
            )
            self.boundaryAttention = nn.MultiheadAttention(
                hiddenDim,
                attentionHeadCount,
                batch_first=True,
            )
        self.deepONetBranch = None
        self.deepONetTrunk = None
        self.deepONetBias = None
        if contextArchitecture == 'deeponet':
            self.deepONetBranch = nn.Sequential(
                nn.Linear(2 * hiddenDim, 2 * hiddenDim),
                nn.SiLU(),
                nn.Linear(2 * hiddenDim, 2 * hiddenDim),
            )
            self.deepONetTrunk = nn.Sequential(
                nn.Linear(queryFeatureDim, 2 * hiddenDim),
                nn.SiLU(),
                nn.Linear(2 * hiddenDim, 2 * hiddenDim),
            )
            self.deepONetBias = nn.Parameter(torch.zeros(1))
        if decoderArchitecture == 'unet':
            self.decoder = _SmoothUNetDecoder(decoderInputDim, hiddenDim)
        elif decoderArchitecture == 'dense':
            self.decoder = _SmoothDenseDecoder(decoderInputDim, hiddenDim)
        else:
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
        # Keep the legacy decoder module and all of its state-dict keys
        # unchanged. An empty block list therefore loads old checkpoints
        # strictly and produces the original architecture.
        self.decoderResidualBlocks = nn.Sequential(
            *(
                _SmoothResidualBlock(2 * hiddenDim, residualScale)
                for _ in range(
                    residualBlockCount if decoderArchitecture == 'resnet' else 0
                )
            )
        )

    def _decode(self, decoderInput: torch.Tensor) -> torch.Tensor:
        if self.decoderArchitecture in ('unet', 'dense'):
            return self.decoder(decoderInput)
        if self.decoderArchitecture == 'mlp':
            return self.decoder(decoderInput)
        hidden = self.decoder[0](decoderInput)
        hidden = self.decoder[1](hidden)
        hidden = self.decoder[2](hidden)
        hidden = self.decoder[3](hidden)
        hidden = self.decoderResidualBlocks(hidden)
        return self.decoder[4](hidden)

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
                        kernelWeights * torch.sqrt(distanceSquared.clamp_min(epsilon))
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

    def _attentionContext(
        self,
        batch: dict[str, Any],
        queryFeatures: torch.Tensor,
        expandedGlobal: torch.Tensor,
        volumeEncoded: torch.Tensor,
        boundaryEncoded: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.queryAttentionProjection is None
            or self.volumeAttention is None
            or self.boundaryAttention is None
        ):
            raise RuntimeError('attention context modules are not configured')
        query = self.queryAttentionProjection(
            torch.cat((queryFeatures, expandedGlobal), dim=-1)
        )
        volumeContext = self.volumeAttention(
            query,
            volumeEncoded,
            volumeEncoded,
            key_padding_mask=~batch['volumeMask'],
            need_weights=False,
        )[0]
        boundaryContext = self.boundaryAttention(
            query,
            boundaryEncoded,
            boundaryEncoded,
            key_padding_mask=~batch['boundaryMask'],
            need_weights=False,
        )[0]
        return torch.cat((volumeContext, boundaryContext), dim=-1)

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
        queryFeatures = self._queryFeatures(queryCoordinates)
        expandedGlobal = globalState.unsqueeze(1).expand(-1, queryCoordinates.shape[1], -1)
        if self.contextArchitecture == 'deeponet':
            if (
                self.deepONetBranch is None
                or self.deepONetTrunk is None
                or self.deepONetBias is None
            ):
                raise RuntimeError('DeepONet modules are not configured')
            branch = self.deepONetBranch(globalState).unsqueeze(1)
            trunk = self.deepONetTrunk(queryFeatures)
            shape = (
                torch.sum(branch * trunk, dim=-1, keepdim=True)
                / math.sqrt(float(trunk.shape[-1]))
                + self.deepONetBias
            )
        else:
            decoderParts = [queryFeatures, expandedGlobal]
            if self.contextArchitecture in ('rbf', 'hybrid'):
                decoderParts.extend(
                    (
                        self._kernelAggregate(
                            queryCoordinates,
                            batch['volumeFeatures'][..., : self.queryDim],
                            volumeEncoded,
                            batch['volumeWeights'],
                            batch['volumeMask'],
                        ),
                        self._boundaryContext(batch, boundaryEncoded),
                    )
                )
            if self.contextArchitecture in ('attention', 'hybrid'):
                decoderParts.append(
                    self._attentionContext(
                        batch,
                        queryFeatures,
                        expandedGlobal,
                        volumeEncoded,
                        boundaryEncoded,
                    )
                )
            shape = self._decode(torch.cat(decoderParts, dim=-1))
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
