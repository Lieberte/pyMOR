from dataclasses import dataclass, field

@dataclass
class schedulerConfig:
    schedulerName: str = 'none'
    stepSize: int = 10
    gamma: float = 0.1
    milestones: list[int] = field(default_factory=list)
    tMax: int = 100
    etaMin: float = 0.0
