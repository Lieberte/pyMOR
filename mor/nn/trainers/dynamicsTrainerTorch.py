import torch
from typing import Any
from mor.nn.trainers.autoEncoderTrainerTorch import autoEncoderTrainerTorch
from mor.nn.registry import nnRegistry

class dynamicsTrainerTorch(autoEncoderTrainerTorch):
    def __init__(self, lossFunction: Any, config: Any | None = None, **kwargs):
        super().__init__(lossFunction=lossFunction, config=config, **kwargs)
        self.trainerName = 'dynamicsTrainerTorch'

nnRegistry.register('trainers', 'dynamicsTrainerTorch', dynamicsTrainerTorch)
