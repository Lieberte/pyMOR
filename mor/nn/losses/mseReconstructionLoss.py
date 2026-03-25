import torch.nn.functional as functional
from typing import Any
from mor.nn.losses.baseLoss import baseLoss
from mor.nn.registry import nnRegistry

class mseReconstructionLoss(baseLoss):
    def __init__(self, **kwargs):
        super().__init__(lossName='mseReconstructionLoss', **kwargs)

    def compute(self, predictions: Any, targets: Any) -> Any:
        return functional.mse_loss(predictions, targets)

nnRegistry.register('losses', 'mseReconstructionLoss', mseReconstructionLoss)
