from typing import Any
from mor.nn.trainers.autoEncoderTrainerTorch import autoEncoderTrainerTorch
from mor.nn.registry import nnRegistry


class operatorTrainerTorch(autoEncoderTrainerTorch):
    def __init__(self, lossFunction: Any, config: Any | None = None, **kwargs):
        super().__init__(lossFunction=lossFunction, config=config, **kwargs)
        self.trainerName = 'operatorTrainerTorch'


nnRegistry.register('trainers', 'operatorTrainerTorch', operatorTrainerTorch)
