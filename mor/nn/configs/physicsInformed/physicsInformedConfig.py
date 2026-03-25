from .base.baseConfig import baseConfig
from dataclasses import dataclass, field

@dataclass
class physicsInformedConfig(baseConfig):
    taskName: str = "physicsInformed"
    # Placeholder for PINN specific task parameters
    
    def __post_init__(self):
        super().__post_init__()
