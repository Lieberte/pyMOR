from dataclasses import dataclass, field
from .base import baseConfig

@dataclass
class representationLearningConfig(baseConfig):
    modelName: str = 'autoEncoderTorch'
    trainerName: str = 'autoEncoderTrainerTorch'
    lossFunction: str = 'mseReconstructionLoss'
    validationName: str = 'reconstructionMetrics'
    dataModuleName: str = 'snapshotDataModule'
    inputDim: int = 0
    latentDim: int = 0
    hiddenDims: list[int] = field(default_factory=list)
