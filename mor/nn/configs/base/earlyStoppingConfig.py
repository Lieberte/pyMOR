from dataclasses import dataclass

@dataclass
class earlyStoppingConfig:
    enabled: bool = False
    patience: int = 10
    delta: float = 0.0
