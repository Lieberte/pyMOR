from dataclasses import dataclass, field
from mor.nn.configs.base import baseConfig

@dataclass
class representationLearningConfig(baseConfig):
    modelName: str = 'autoEncoderTorch'
    trainerName: str = 'autoEncoderTrainerTorch'
    lossFunction: str = 'mseReconstructionLoss'
    validationName: str = 'reconstructionMetrics'
    dataModuleName: str = 'snapshotDataModule'
