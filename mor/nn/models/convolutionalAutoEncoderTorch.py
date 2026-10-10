import torch
from torch import nn
from typing import Any
from mor.nn.models.torchModelBase import torchModelBase
from mor.nn.registry import nnRegistry

class convolutionalAutoEncoderTorch(torchModelBase):
    def __init__(self, inputChannels: int, latentDim: int, hiddenChannels: list[int] | None = None, inputShape: tuple[int, int] | None = None, **kwargs):
        super().__init__(modelName='convolutionalAutoEncoderTorch', **kwargs)
        hiddenChannels = hiddenChannels or [16, 32, 64]
        encoderLayers = []
        inChannels = inputChannels
        for hChannels in hiddenChannels:
            encoderLayers.append(nn.Conv2d(inChannels, hChannels, kernel_size=3, stride=2, padding=1))
            encoderLayers.append(nn.ReLU())
            inChannels = hChannels
        self.encoder = nn.Sequential(*encoderLayers)
        self.inputChannels = inputChannels
        self.hiddenChannels = hiddenChannels
        self.register_buffer('_inputSpatialShape', torch.zeros(2, dtype=torch.int64))
        self.flatten = nn.Flatten()
        self.fcEncode = nn.LazyLinear(latentDim)
        self.fcDecode = nn.LazyLinear(0)
        decoderLayers = []
        for hChannels in reversed(hiddenChannels[:-1]):
            decoderLayers.append(nn.ConvTranspose2d(inChannels, hChannels, kernel_size=3, stride=2, padding=1, output_padding=1))
            decoderLayers.append(nn.ReLU())
            inChannels = hChannels
        decoderLayers.append(nn.ConvTranspose2d(inChannels, inputChannels, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.decoder = nn.Sequential(*decoderLayers)
        if inputShape is not None:
            self._configureShape(inputShape)

    def _configureShape(self, shape):
        if len(shape) != 2 or any(int(size) != size or size <= 0 for size in shape):
            raise ValueError('inputShape must contain two positive spatial dimensions')
        shape = tuple(int(size) for size in shape)
        previous = tuple(self._inputSpatialShape.tolist())
        if previous != (0, 0) and previous != shape:
            raise ValueError(f'input spatial shape must remain {previous}')
        self._inputSpatialShape.copy_(torch.tensor(shape, device=self._inputSpatialShape.device))
        for _ in self.hiddenChannels:
            shape = tuple((size + 1) // 2 for size in shape)
        self.fcDecode.out_features = self.hiddenChannels[-1] * shape[0] * shape[1]

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if '_inputSpatialShape' not in state_dict:
            from torch.nn.parameter import UninitializedParameter
            decoderWeight = state_dict.get('fcDecode.weight')
            encoderWeight = state_dict.get('fcEncode.weight')
            if isinstance(decoderWeight, UninitializedParameter) or isinstance(encoderWeight, UninitializedParameter):
                raise ValueError('legacy checkpoint has uninitialized linear layers; initialize and save the original model first')
            shape = tuple(self._inputSpatialShape.tolist())
            if shape == (0, 0):
                legacyFeatures = self.hiddenChannels[-1] * 4 * 4
                if decoderWeight is None or encoderWeight is None or decoderWeight.shape[0] != legacyFeatures or encoderWeight.shape[1] != legacyFeatures:
                    raise ValueError('legacy checkpoint requires explicit inputShape matching its linear layers')
                side = 4 * 2 ** len(self.hiddenChannels)
                shape = (side, side)
            encodedShape = shape
            for _ in self.hiddenChannels:
                encodedShape = tuple((size + 1) // 2 for size in encodedShape)
            features = self.hiddenChannels[-1] * encodedShape[0] * encodedShape[1]
            if decoderWeight is None or encoderWeight is None or decoderWeight.shape[0] != features or encoderWeight.shape[1] != features:
                raise ValueError('legacy checkpoint linear layers do not match inputShape; weights cannot be migrated losslessly')
            self._configureShape(shape)
            original = state_dict
            state_dict = state_dict.copy()
            if hasattr(original, '_metadata'):
                state_dict._metadata = original._metadata
            state_dict['_inputSpatialShape'] = self._inputSpatialShape.clone()
        if '_inputSpatialShape' in state_dict:
            shape = tuple(state_dict['_inputSpatialShape'].tolist())
            if shape != (0, 0):
                self._configureShape(shape)
        if assign:
            return super().load_state_dict(state_dict, strict=strict, assign=True)
        return super().load_state_dict(state_dict, strict=strict)

    def encode(self, inputs: Any) -> Any:
        if inputs.ndim != 4 or inputs.shape[1] != self.inputChannels:
            raise ValueError('inputs must have shape (sample, inputChannels, height, width)')
        self._configureShape(inputs.shape[-2:])
        x = self.encoder(inputs)
        x = self.flatten(x)
        return self.fcEncode(x)

    def decode(self, latents: Any) -> Any:
        shape = tuple(self._inputSpatialShape.tolist())
        if shape == (0, 0):
            raise ValueError('encode an input or supply inputShape before decoding')
        shapes = [shape]
        for _ in self.hiddenChannels:
            shapes.append(tuple((size + 1) // 2 for size in shapes[-1]))
        x = self.fcDecode(latents)
        x = x.reshape(-1, self.hiddenChannels[-1], *shapes[-1])
        targetShapes = iter(reversed(shapes[:-1]))
        for layer in self.decoder:
            if isinstance(layer, nn.ConvTranspose2d):
                target = next(targetShapes)
                x = layer(x, output_size=(x.shape[0], layer.out_channels, *target))
            else:
                x = layer(x)
        return x

    def forward(self, inputs: Any) -> Any:
        return self.decode(self.encode(inputs))

nnRegistry.register('models', 'convolutionalAutoEncoderTorch', convolutionalAutoEncoderTorch)
