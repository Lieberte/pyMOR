import torch
from torch import nn
from typing import Any
from mor.nn.models.torchModelBase import torchModelBase
from mor.nn.registry import nnRegistry

class variationalAutoEncoderTorch(torchModelBase):
    def __init__(self, inputDim: int, latentDim: int, hiddenDims: list[int] | None = None, **kwargs):
        super().__init__(modelName='variationalAutoEncoderTorch', **kwargs)
        hiddenDims = hiddenDims or []
        encoderLayers = []
        inDim = inputDim
        for hDim in hiddenDims:
            encoderLayers.append(nn.Linear(inDim, hDim))
            encoderLayers.append(nn.ReLU())
            inDim = hDim
        self.encoderBase = nn.Sequential(*encoderLayers)
        self.fcMu = nn.Linear(inDim, latentDim)
        self.fcLogVar = nn.Linear(inDim, latentDim)
        decoderLayers = []
        inDim = latentDim
        for hDim in reversed(hiddenDims):
            decoderLayers.append(nn.Linear(inDim, hDim))
            decoderLayers.append(nn.ReLU())
            inDim = hDim
        decoderLayers.append(nn.Linear(inDim, inputDim))
        self.decoder = nn.Sequential(*decoderLayers)

    def encode(self, inputs: Any) -> tuple[Any, Any]:
        h = self.encoderBase(inputs)
        return self.fcMu(h), self.fcLogVar(h)

    def reparameterize(self, mu: Any, logVar: Any) -> Any:
        std = torch.exp(0.5 * logVar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latents: Any) -> Any:
        return self.decoder(latents)

    def forward(self, inputs: Any) -> tuple[Any, Any, Any]:
        mu, logVar = self.encode(inputs)
        z = self.reparameterize(mu, logVar)
        return self.decode(z), mu, logVar

nnRegistry.register('models.representationLearning', 'variationalAutoEncoderTorch', variationalAutoEncoderTorch)
