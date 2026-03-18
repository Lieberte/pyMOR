from dataclasses import dataclass

@dataclass
class dataLoaderConfig:
    batchSize: int = 32
    validationSplit: float = 0.2
    shuffle: bool = True
    numWorkers: int = 0
    pinMemory: bool = False
    dropLast: bool = False
    persistentWorkers: bool = False
