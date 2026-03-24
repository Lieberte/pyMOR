import torch
from torch import nn
from typing import Any
from mor.nn.models.torchModelBase import torchModelBase
from mor.nn.registry import nnRegistry

class hamiltonianDynamicsTorch(torchModelBase):
    def __init__(self, **kwargs):
        super().__init__(modelName='hamiltonianDynamicsTorch', **kwargs)
        inputDim = kwargs.get('inputDim', 0)
        hiddenDims = kwargs.get('hiddenDims', [64, 64])
        layers = []
        inDim = inputDim
        for hDim in hiddenDims:
            layers.append(nn.Linear(inDim, hDim))
            layers.append(nn.Tanh())
            inDim = hDim
        layers.append(nn.Linear(inDim, 1))
        self.hamiltonian = nn.Sequential(*layers)

    def forward(self, inputs: Any) -> Any:
        with torch.set_grad_enabled(True):
            inputs = inputs.requires_grad_(True)
            h = self.hamiltonian(inputs)
            dH = torch.autograd.grad(h.sum(), inputs, create_graph=True)[0]
        qDim = dH.shape[-1] // 2
        dq, dp = dH[..., :qDim], dH[..., qDim:]
        return torch.cat([dp, -dq], dim=-1)

nnRegistry.register('models.dynamicsLearning', 'physicsInformed.hamiltonianDynamicsTorch', hamiltonianDynamicsTorch)
