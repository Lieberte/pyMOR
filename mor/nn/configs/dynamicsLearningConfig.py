from dataclasses import dataclass, field
from mor.nn.configs.base import baseConfig

@dataclass
class dynamicsLearningConfig(baseConfig):
    modelName: str = 'sequenceModels.rnnDynamicsTorch'
    trainerName: str = 'dynamicsTrainerTorch'
    lossFunction: str = 'oneStepPredictionLoss'
    validationName: str = 'dynamicsPredictionMetrics'
    dataModuleName: str = 'trajectoryDataModule'
    
    # 动力学任务特有全局参数
    horizon: int = 1
    teacherForcingRatio: float = 0.0
