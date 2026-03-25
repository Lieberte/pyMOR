from dataclasses import dataclass
from mor.nn.configs.base import baseConfig

@dataclass
class dynamicsLearningConfig(baseConfig):
    name: str = 'dynamicsLearningConfig'
    
    horizon: int = 1
    teacherForcingRatio: float = 0.0
