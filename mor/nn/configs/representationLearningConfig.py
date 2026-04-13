from dataclasses import dataclass
from mor.nn.configs.base import baseConfig

@dataclass
class representationLearningConfig(baseConfig):
    name: str = 'representationLearningConfig'
