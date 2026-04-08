from dataclasses import dataclass
from mor.nn.configs.base import baseConfig

# TODO: Implement specific dynamics learning config, and figure out how to fit frame. At least LSTM/GRU/Transformer
@dataclass
class dynamicsLearningConfig(baseConfig):
    name: str = 'dynamicsLearningConfig'
    
    horizon: int = 1
    teacherForcingRatio: float = 0.0
