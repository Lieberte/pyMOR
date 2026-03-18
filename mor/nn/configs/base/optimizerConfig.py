from dataclasses import dataclass

@dataclass
class optimizerConfig:
    learningRate: float = 1e-03
    weightDecay: float = 0.0
    gradientClip: float | None = None
