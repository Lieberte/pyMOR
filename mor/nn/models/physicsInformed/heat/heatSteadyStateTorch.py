import torch
import torch.nn as nn
from mor.nn.models.torchModelBase import torchModelBase
from mor.nn.registry import nnRegistry

class heatSteadyStateTorch(torchModelBase):
    def __init__(self, **kwargs):
        super().__init__(modelName='heatSteadyStateTorch', **kwargs)
        inputDim = kwargs.get('inputDim', 3)  # x, y, z
        outputDim = kwargs.get('outputDim', 1)  # T
        hiddenDims = kwargs.get('hiddenDims', [64, 64, 64])
        
        layers = []
        inDim = inputDim
        for hDim in hiddenDims:
            layers.append(nn.Linear(inDim, hDim))
            layers.append(nn.Tanh())  # Use Tanh for smooth derivatives
            inDim = hDim
        layers.append(nn.Linear(inDim, outputDim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

nnRegistry.register('models.physicsInformed', 'heat.heatSteadyStateTorch', heatSteadyStateTorch)
