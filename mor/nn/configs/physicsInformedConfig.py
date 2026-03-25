from dataclasses import dataclass
from mor.nn.configs.base import baseConfig

@dataclass
class physicsInformedConfig(baseConfig):
    name: str = 'physicsInformedConfig'
