from __future__ import annotations

import torch
from torch import nn

from mor.nn.models.torchModelBase import torchModelBase
from mor.nn.registry import nnRegistry


def _mlp(inputDim: int, hiddenDim: int, outputDim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inputDim, hiddenDim),
        nn.SiLU(),
        nn.Linear(hiddenDim, outputDim),
    )


class thermalMeshGnnTorch(torchModelBase):
    """Message-passing baseline for one thermal finite-element mesh graph."""

    def __init__(
        self,
        *,
        nodeFeatureDim: int,
        edgeFeatureDim: int = 4,
        hiddenDim: int = 48,
        messagePassingSteps: int = 4,
        useGlobalContext: bool = False,
        **kwargs,
    ):
        super().__init__(modelName='thermalMeshGnnTorch', **kwargs)
        dimensions = (
            int(nodeFeatureDim),
            int(edgeFeatureDim),
            int(hiddenDim),
            int(messagePassingSteps),
        )
        if min(dimensions) <= 0:
            raise ValueError('thermal mesh GNN dimensions and steps must be positive')
        self.nodeEncoder = _mlp(nodeFeatureDim, hiddenDim, hiddenDim)
        self.edgeEncoder = _mlp(edgeFeatureDim, hiddenDim, hiddenDim)
        self.messageLayers = nn.ModuleList(
            _mlp(3 * hiddenDim, hiddenDim, hiddenDim)
            for _ in range(messagePassingSteps)
        )
        self.useGlobalContext = bool(useGlobalContext)
        updateInputDim = (3 if self.useGlobalContext else 2) * hiddenDim
        self.updateLayers = nn.ModuleList(
            _mlp(updateInputDim, hiddenDim, hiddenDim)
            for _ in range(messagePassingSteps)
        )
        self.normalizations = nn.ModuleList(
            nn.LayerNorm(hiddenDim) for _ in range(messagePassingSteps)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hiddenDim, hiddenDim),
            nn.SiLU(),
            nn.Linear(hiddenDim, 1),
        )

    def forward(
        self,
        nodeFeatures: torch.Tensor,
        edgeIndex: torch.Tensor,
        edgeFeatures: torch.Tensor,
    ) -> torch.Tensor:
        if edgeIndex.ndim != 2 or edgeIndex.shape[0] != 2:
            raise ValueError('edgeIndex must have shape (2, edgeCount)')
        source, target = edgeIndex
        nodeState = self.nodeEncoder(nodeFeatures)
        edgeState = self.edgeEncoder(edgeFeatures)
        for messageLayer, updateLayer, normalization in zip(
            self.messageLayers,
            self.updateLayers,
            self.normalizations,
        ):
            messages = messageLayer(
                torch.cat(
                    (nodeState[source], nodeState[target], edgeState),
                    dim=-1,
                )
            )
            aggregate = torch.zeros_like(nodeState)
            aggregate.index_add_(0, target, messages)
            degree = torch.zeros(
                nodeState.shape[0],
                dtype=nodeState.dtype,
                device=nodeState.device,
            )
            degree.index_add_(0, target, torch.ones_like(target, dtype=nodeState.dtype))
            aggregate = aggregate / degree.clamp_min(1.0).unsqueeze(-1)
            updateInputs = [nodeState, aggregate]
            if self.useGlobalContext:
                globalState = nodeState.mean(dim=0, keepdim=True).expand_as(nodeState)
                updateInputs.append(globalState)
            update = updateLayer(torch.cat(updateInputs, dim=-1))
            nodeState = normalization(nodeState + update)
        return self.decoder(nodeState)


nnRegistry.register('models', 'thermalMeshGnnTorch', thermalMeshGnnTorch)
