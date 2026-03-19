import torch
import torch.nn.functional as F
from typing import Any
from mor.nn.losses.baseLoss import baseLoss
from mor.nn.registry import nnRegistry

class vaeReconstructionLoss(baseLoss):
    def __init__(self, klWeight: float = 1.0, **kwargs):
        super().__init__(lossName='vaeReconstructionLoss', **kwargs)
        self.klWeight = klWeight

    def compute(self, outputs: tuple[Any, Any, Any], targets: Any) -> Any:
        reconX, mu, logVar = outputs
        reconLoss = F.mse_loss(reconX, targets, reduction='sum')
        klLoss = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
        return (reconLoss + self.klWeight * klLoss) / targets.size(0)

nnRegistry.register('losses.representationLearning', 'vaeReconstructionLoss', vaeReconstructionLoss)
