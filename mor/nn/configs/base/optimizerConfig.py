from dataclasses import dataclass, field

@dataclass
class adamOptimizerConfig:
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-08
    amsgrad: bool = False

@dataclass
class adamwOptimizerConfig:
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-08
    amsgrad: bool = False

@dataclass
class sgdOptimizerConfig:
    momentum: float = 0.9
    dampening: float = 0.0
    nesterov: bool = False

@dataclass
class rmspropOptimizerConfig:
    alpha: float = 0.99
    epsilon: float = 1e-08
    momentum: float = 0.0
    centered: bool = False

@dataclass
class adagradOptimizerConfig:
    lrDecay: float = 0.0
    epsilon: float = 1e-10
    initialAccumulatorValue: float = 0.0

@dataclass
class optimizerConfig:
    optimizerName: str = 'adam'
    learningRate: float = 1e-03
    weightDecay: float = 0.0
    gradientClip: float | None = None
    adam: adamOptimizerConfig = field(default_factory=adamOptimizerConfig)
    adamw: adamwOptimizerConfig = field(default_factory=adamwOptimizerConfig)
    sgd: sgdOptimizerConfig = field(default_factory=sgdOptimizerConfig)
    rmsprop: rmspropOptimizerConfig = field(default_factory=rmspropOptimizerConfig)
    adagrad: adagradOptimizerConfig = field(default_factory=adagradOptimizerConfig)
