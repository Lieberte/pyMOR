from dataclasses import dataclass
from mor.nn.configs.physicsInformedConfig import physicsInformedConfig

@dataclass
class parametricGeometryPinnConfig(physicsInformedConfig):
    name: str = 'parametricGeometryPinnConfig'
