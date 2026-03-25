import torch
from typing import Any
from mor.nn.trainers.autoEncoderTrainerTorch import autoEncoderTrainerTorch
from mor.nn.registry import nnRegistry

class dynamicsTrainerTorch(autoEncoderTrainerTorch):
    def __init__(self, lossFunction: Any, config: Any | None = None, **kwargs):
        super().__init__(lossFunction=lossFunction, config=config, **kwargs)
        self.trainerName = 'dynamicsTrainerTorch'

    def _runBatch(self, model: Any, inputs: Any, targets: Any, device: str, dtype: torch.dtype, optimizer: Any | None = None) -> float:
        inputs, targets = self._toRuntimeTensor(inputs, device, dtype), self._toRuntimeTensor(targets, device, dtype)
        if optimizer: optimizer.zero_grad()
        preds = model.forward(inputs)
        loss = self.lossFunction.compute(preds, targets)
        if optimizer:
            loss.backward()
            optimizer.step()
        return float(loss.item())

nnRegistry.register('trainers', 'dynamicsTrainerTorch', dynamicsTrainerTorch)
