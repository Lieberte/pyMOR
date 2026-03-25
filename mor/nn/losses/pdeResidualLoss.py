import torch
from typing import Any
from mor.nn.losses.baseLoss import baseLoss
from mor.nn.registry import nnRegistry

class pdeResidualLoss(baseLoss):
    def __init__(self, **kwargs):
        super().__init__(lossName='pdeResidualLoss', **kwargs)

    def compute(self, predictions: Any, targets: Any, **kwargs) -> Any:
        # TODO: Implement generic PDE residual computation
        return torch.mean((predictions - targets) ** 2)

nnRegistry.register('losses', 'pdeResidualLoss', pdeResidualLoss)
