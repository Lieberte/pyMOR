from torch import nn
from typing import Any
from mor.nn.models.torchModelBase import torchModelBase
from mor.nn.registry import nnRegistry

class autoEncoderTorch(torchModelBase):
    def __init__(self, inputDim: int, latentDim: int, hiddenDims: list[int] | None = None, **kwargs):
        super().__init__(modelName='autoEncoderTorch', **kwargs)
        hiddenDims = hiddenDims or []
        encoderLayers = []
        inDim = inputDim
        for hiddenDim in hiddenDims:
            encoderLayers.append(nn.Linear(inDim, hiddenDim))
            encoderLayers.append(nn.ReLU())
            inDim = hiddenDim
        encoderLayers.append(nn.Linear(inDim, latentDim))
        self.encoder = nn.Sequential(*encoderLayers)
        decoderLayers = []
        inDim = latentDim
        for hiddenDim in reversed(hiddenDims):
            decoderLayers.append(nn.Linear(inDim, hiddenDim))
            decoderLayers.append(nn.ReLU())
            inDim = hiddenDim
        decoderLayers.append(nn.Linear(inDim, inputDim))
        self.decoder = nn.Sequential(*decoderLayers)

    def encode(self, inputs: Any) -> Any:
        return self.encoder(inputs)

    def decode(self, latents: Any) -> Any:
        return self.decoder(latents)

    def forward(self, inputs: Any) -> Any:
        return self.decode(self.encode(inputs))

nnRegistry.register('models', 'autoEncoderTorch', autoEncoderTorch)
