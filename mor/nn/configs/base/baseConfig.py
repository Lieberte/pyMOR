from dataclasses import dataclass, field
from .earlyStoppingConfig import earlyStoppingConfig
from .loggingConfig import loggingConfig
from .optimizerConfig import optimizerConfig
from .schedulerConfig import schedulerConfig
from .dataLoaderConfig import dataLoaderConfig
from .runtimeConfig import runtimeConfig
from .checkpointConfig import checkpointConfig

@dataclass
class baseConfig:
    name: str = 'baseConfig'
    options: dict = field(default_factory=dict)
    epochs: int = 100
    earlyStopping: earlyStoppingConfig = field(default_factory=earlyStoppingConfig)
    logging: loggingConfig = field(default_factory=loggingConfig)
    optimizer: optimizerConfig = field(default_factory=optimizerConfig)
    scheduler: schedulerConfig = field(default_factory=schedulerConfig)
    dataLoader: dataLoaderConfig = field(default_factory=dataLoaderConfig)
    runtime: runtimeConfig = field(default_factory=runtimeConfig)
    checkpoint: checkpointConfig = field(default_factory=checkpointConfig)
