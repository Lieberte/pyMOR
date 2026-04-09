from dataclasses import dataclass
from mor.nn.configs.physicsInformedConfig import physicsInformedConfig

@dataclass
class fixedGeometryPinnConfig(physicsInformedConfig):
    name: str = 'fixedGeometryPinnConfig'
