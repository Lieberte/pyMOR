from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from mor.nn.models.torchModelBase import torchModelBase
from mor.nn.registry import nnRegistry


def _knnIndices(features: torch.Tensor, neighborCount: int) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError('DGCNN features must have shape (batch, channels, points)')
    pointCount = features.shape[2]
    if neighborCount >= pointCount:
        raise ValueError('DGCNN neighborCount must be smaller than point count')
    squaredNorm = torch.sum(features * features, dim=1, keepdim=True)
    negativeSquaredDistance = (
        -squaredNorm.transpose(1, 2)
        + 2.0 * torch.matmul(features.transpose(1, 2), features)
        - squaredNorm
    )
    return negativeSquaredDistance.topk(k=neighborCount, dim=-1).indices


def _edgeFeatures(features: torch.Tensor, neighborCount: int) -> torch.Tensor:
    batchSize, channelCount, pointCount = features.shape
    indices = _knnIndices(features, neighborCount)
    batchOffset = (
        torch.arange(batchSize, device=features.device).view(-1, 1, 1) * pointCount
    )
    flattenedIndices = (indices + batchOffset).reshape(-1)
    pointsFirst = features.transpose(1, 2).contiguous()
    neighbors = pointsFirst.reshape(batchSize * pointCount, channelCount)[
        flattenedIndices
    ].reshape(batchSize, pointCount, neighborCount, channelCount)
    centers = pointsFirst.unsqueeze(2).expand(-1, -1, neighborCount, -1)
    return torch.cat((neighbors - centers, centers), dim=-1).permute(0, 3, 1, 2)


def _edgeBlock(inputChannels: int, outputChannels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(2 * inputChannels, outputChannels, kernel_size=1, bias=False),
        nn.BatchNorm2d(outputChannels),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


class geometryDgcnnEncoderTorch(torchModelBase):
    def __init__(
        self,
        *,
        inputFeatureDim: int = 6,
        edgeChannels: Sequence[int] = (32, 32, 64, 128),
        globalFeatureDim: int = 256,
        embeddingDim: int = 128,
        neighborCount: int = 16,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__(modelName='geometryDgcnnEncoderTorch', **kwargs)
        edgeChannels = tuple(int(value) for value in edgeChannels)
        dimensions = (
            int(inputFeatureDim),
            int(globalFeatureDim),
            int(embeddingDim),
            int(neighborCount),
            *edgeChannels,
        )
        if not edgeChannels or min(dimensions) <= 0:
            raise ValueError('DGCNN dimensions and neighbor count must be positive')
        if not 0.0 <= dropout < 1.0:
            raise ValueError('DGCNN dropout must be in [0, 1)')
        self.inputFeatureDim = int(inputFeatureDim)
        self.edgeChannels = edgeChannels
        self.globalFeatureDim = int(globalFeatureDim)
        self.embeddingDim = int(embeddingDim)
        self.neighborCount = int(neighborCount)
        self.dropout = float(dropout)
        blocks = []
        inputChannels = self.inputFeatureDim
        for outputChannels in self.edgeChannels:
            blocks.append(_edgeBlock(inputChannels, outputChannels))
            inputChannels = outputChannels
        self.edgeBlocks = nn.ModuleList(blocks)
        concatenatedChannels = sum(self.edgeChannels)
        self.globalProjection = nn.Sequential(
            nn.Conv1d(
                concatenatedChannels,
                self.globalFeatureDim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(self.globalFeatureDim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.embeddingProjection = nn.Sequential(
            nn.Linear(2 * self.globalFeatureDim, self.globalFeatureDim),
            nn.LayerNorm(self.globalFeatureDim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.globalFeatureDim, self.embeddingDim),
        )

    def forward(self, pointFeatures: torch.Tensor) -> torch.Tensor:
        if pointFeatures.ndim != 3 or pointFeatures.shape[2] != self.inputFeatureDim:
            raise ValueError(
                'DGCNN input must have shape (batch, pointCount, inputFeatureDim)'
            )
        if pointFeatures.shape[1] <= self.neighborCount:
            raise ValueError('DGCNN input point count must exceed neighborCount')
        state = pointFeatures.transpose(1, 2).contiguous()
        multiscale = []
        for block in self.edgeBlocks:
            state = block(_edgeFeatures(state, self.neighborCount)).max(dim=-1).values
            multiscale.append(state)
        globalFeatures = self.globalProjection(torch.cat(multiscale, dim=1))
        pooled = torch.cat(
            (globalFeatures.max(dim=2).values, globalFeatures.mean(dim=2)), dim=1
        )
        return F.normalize(self.embeddingProjection(pooled), p=2, dim=1)

    def configDict(self) -> dict:
        return {
            'inputFeatureDim': self.inputFeatureDim,
            'edgeChannels': list(self.edgeChannels),
            'globalFeatureDim': self.globalFeatureDim,
            'embeddingDim': self.embeddingDim,
            'neighborCount': self.neighborCount,
            'dropout': self.dropout,
        }


class geometryDgcnnClassifierTorch(torchModelBase):
    def __init__(
        self,
        *,
        classCount: int,
        encoder: geometryDgcnnEncoderTorch | None = None,
        **encoderArguments,
    ):
        super().__init__(modelName='geometryDgcnnClassifierTorch')
        if int(classCount) <= 1:
            raise ValueError('DGCNN classifier requires at least two classes')
        self.classCount = int(classCount)
        self.encoder = encoder or geometryDgcnnEncoderTorch(**encoderArguments)
        self.classifier = nn.Linear(self.encoder.embeddingDim, self.classCount)

    def encode(self, pointFeatures: torch.Tensor) -> torch.Tensor:
        return self.encoder(pointFeatures)

    def forward(self, pointFeatures: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(pointFeatures))


nnRegistry.register('models', 'geometryDgcnnEncoderTorch', geometryDgcnnEncoderTorch)
nnRegistry.register('models', 'geometryDgcnnClassifierTorch', geometryDgcnnClassifierTorch)
