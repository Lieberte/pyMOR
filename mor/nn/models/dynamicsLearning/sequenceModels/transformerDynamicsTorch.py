import torch
from torch import nn
from typing import Any
from mor.nn.models.torchModelBase import torchModelBase
from mor.nn.registry import nnRegistry

class transformerDynamicsTorch(torchModelBase):
    def __init__(self, inputDim: int, dModel: int, nHead: int, numLayers: int, outputDim: int, **kwargs):
        super().__init__(modelName='transformerDynamicsTorch', **kwargs)
        self.embedding = nn.Linear(inputDim, dModel)
        encoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHead, batch_first=True)
        self.transformerEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.fc = nn.Linear(dModel, outputDim)

    def forward(self, inputs: Any, mask: Any | None = None) -> Any:
        x = self.embedding(inputs)
        out = self.transformerEncoder(x, mask)
        return self.fc(out)

nnRegistry.register('models.dynamicsLearning', 'sequenceModels.transformerDynamicsTorch', transformerDynamicsTorch)
