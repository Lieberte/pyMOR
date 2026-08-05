import torch

from mor.nn.registry import nnRegistry

from .baseLoss import baseLoss


class thermalOperatorLoss(baseLoss):
    def __init__(self, hotspotWeight: float = 0.0, **kwargs):
        super().__init__(lossName='thermalOperatorLoss', **kwargs)
        self.hotspotWeight = float(hotspotWeight)
        if self.hotspotWeight < 0.0:
            raise ValueError('hotspotWeight must be non-negative')

    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        queryWeights: torch.Tensor,
        queryMask: torch.Tensor,
    ) -> torch.Tensor:
        weights = queryWeights * queryMask.to(dtype=queryWeights.dtype)
        if self.hotspotWeight:
            targetMinimum = targets.masked_fill(
                ~queryMask.unsqueeze(-1), torch.inf
            ).amin(dim=1, keepdim=True)
            targetMaximum = targets.masked_fill(
                ~queryMask.unsqueeze(-1), -torch.inf
            ).amax(dim=1, keepdim=True)
            normalized = (targets - targetMinimum) / (
                targetMaximum - targetMinimum
            ).clamp_min(1e-6)
            weights = weights * (1.0 + self.hotspotWeight * normalized.squeeze(-1))
        squaredError = (predictions - targets).square().squeeze(-1)
        return (squaredError * weights).sum() / weights.sum().clamp_min(1e-12)


nnRegistry.register('losses', 'thermalOperatorLoss', thermalOperatorLoss)
