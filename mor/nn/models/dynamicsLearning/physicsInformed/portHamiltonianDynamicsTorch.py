import torch
from torch import nn
from typing import Any
from mor.nn.models.torchModelBase import torchModelBase
from mor.nn.registry import nnRegistry

class portHamiltonianDynamicsTorch(torchModelBase):
    def __init__(self, **kwargs):
        super().__init__(modelName='portHamiltonianDynamicsTorch', **kwargs)
        inputDim = kwargs.get('inputDim', 0)
        hiddenDims = kwargs.get('hiddenDims', [64, 64])
        self.qDim = inputDim // 2
        layers = []
        inDim = inputDim
        for hDim in hiddenDims:
            layers.append(nn.Linear(inDim, hDim))
            layers.append(nn.Tanh())
            inDim = hDim
        layers.append(nn.Linear(inDim, 1))
        self.hamiltonian = nn.Sequential(*layers)
        self.jMatrix = nn.Parameter(torch.randn(inputDim, inputDim))
        self.rMatrix = nn.Parameter(torch.randn(inputDim, inputDim))

    def forward(self, inputs: Any) -> Any:
        with torch.set_grad_enabled(True):
            inputs = inputs.requires_grad_(True)
            h = self.hamiltonian(inputs)
            dH = torch.autograd.grad(h.sum(), inputs, create_graph=True)[0]
        j = self.jMatrix - self.jMatrix.t()
        r = self.rMatrix @ self.rMatrix.t()
        return (j - r) @ dH.unsqueeze(-1)

nnRegistry.register('models.dynamicsLearning', 'physicsInformed.portHamiltonianDynamicsTorch', portHamiltonianDynamicsTorch)
