import torch
from torch import nn
from typing import Any
from mor.nn.models.torchModelBase import torchModelBase
from mor.nn.registry import nnRegistry

class convolutionalAutoEncoderTorch(torchModelBase):
    def __init__(self, inputChannels: int, latentDim: int, hiddenChannels: list[int] | None = None, **kwargs):
        super().__init__(modelName='convolutionalAutoEncoderTorch', **kwargs)
        hiddenChannels = hiddenChannels or [16, 32, 64]
        encoderLayers = []
        inChannels = inputChannels
        for hChannels in hiddenChannels:
            encoderLayers.append(nn.Conv2d(inChannels, hChannels, kernel_size=3, stride=2, padding=1))
            encoderLayers.append(nn.ReLU())
            inChannels = hChannels
        self.encoder = nn.Sequential(*encoderLayers)
        self.flatten = nn.Flatten()
        self.fcEncode = nn.LazyLinear(latentDim)
        self.fcDecode = nn.LazyLinear(inChannels * 4 * 4) # Placeholder for spatial size
        self.unflatten = nn.Unflatten(1, (inChannels, 4, 4))
        decoderLayers = []
        for hChannels in reversed(hiddenChannels[:-1]):
            decoderLayers.append(nn.ConvTranspose2d(inChannels, hChannels, kernel_size=3, stride=2, padding=1, output_padding=1))
            decoderLayers.append(nn.ReLU())
            inChannels = hChannels
        decoderLayers.append(nn.ConvTranspose2d(inChannels, inputChannels, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.decoder = nn.Sequential(*decoderLayers)

    def encode(self, inputs: Any) -> Any:
        x = self.encoder(inputs)
        x = self.flatten(x)
        return self.fcEncode(x)

    def decode(self, latents: Any) -> Any:
        x = self.fcDecode(latents)
        x = self.unflatten(x)
        return self.decoder(x)

    def forward(self, inputs: Any) -> Any:
        return self.decode(self.encode(inputs))

nnRegistry.register('models.representationLearning', 'convolutionalAutoEncoderTorch', convolutionalAutoEncoderTorch)
