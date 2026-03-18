from dataclasses import dataclass, field
from .base import baseConfig

@dataclass
class stateReconstructionConfig(baseConfig):
    inputDim: int = 0
    latentDim: int = 0
    hiddenDims: list[int] = field(default_factory=list)
    lossFunction: str = 'mse'