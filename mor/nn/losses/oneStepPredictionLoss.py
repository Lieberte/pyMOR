import torch
import torch.nn.functional as F
from typing import Any
from mor.nn.losses.baseLoss import baseLoss
from mor.nn.registry import nnRegistry

class oneStepPredictionLoss(baseLoss):
    def __init__(self, **kwargs):
        super().__init__(lossName='oneStepPredictionLoss', **kwargs)

    def compute(self, predictions: Any, targets: Any) -> Any:
        if isinstance(predictions, (tuple, list)): predictions = predictions[0]
        return F.mse_loss(predictions, targets)

class multiStepRolloutLoss(baseLoss):
    def __init__(self, gamma: float = 1.0, **kwargs):
        super().__init__(lossName='multiStepRolloutLoss', **kwargs)
        self.gamma = gamma

    def compute(self, predictions: Any, targets: Any) -> Any:
        if isinstance(predictions, (tuple, list)): predictions = predictions[0]
        if self.gamma == 1.0: return F.mse_loss(predictions, targets)
        msePerStep = torch.mean((predictions - targets)**2, dim=-1)
        horizon = targets.size(1)
        weights = torch.pow(self.gamma, torch.arange(horizon, device=targets.device)).unsqueeze(0)
        return torch.mean(msePerStep * weights)

nnRegistry.register('losses', 'oneStepPredictionLoss', oneStepPredictionLoss)
nnRegistry.register('losses', 'multiStepRolloutLoss', multiStepRolloutLoss)
