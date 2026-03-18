from dataclasses import dataclass, field

@dataclass
class baseConfig:
    name: str = 'baseConfig'
    options: dict = field(default_factory=dict)
