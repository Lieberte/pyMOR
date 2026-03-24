import torch
from torch import nn
from typing import Any
from mor.nn.models.torchModelBase import torchModelBase
from mor.nn.registry import nnRegistry

class gruDynamicsTorch(torchModelBase):
    def __init__(self, **kwargs):
        super().__init__(modelName='gruDynamicsTorch', **kwargs)
        inputDim = kwargs.get('inputDim', 0)
        hiddenDims = kwargs.get('hiddenDims', [64])
        outputDim = kwargs.get('outputDim', inputDim)
        layers = []
        inDim = inputDim
        for hDim in hiddenDims:
            layers.append(nn.GRU(inDim, hDim, batch_first=True))
            inDim = hDim
        self.grus = nn.ModuleList(layers)
        self.fc = nn.Linear(inDim, outputDim)

    def forward(self, inputs: Any, hiddens: list[Any] | None = None) -> tuple[Any, list[Any]]:
        out = inputs
        newHiddens = []
        for i, gru in enumerate(self.grus):
            h = hiddens[i] if hiddens is not None else None
            out, h = gru(out, h)
            newHiddens.append(h)
        return self.fc(out), newHiddens

nnRegistry.register('models.dynamicsLearning', 'sequenceModels.gruDynamicsTorch', gruDynamicsTorch)
