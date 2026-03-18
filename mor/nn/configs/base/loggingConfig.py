from dataclasses import dataclass

@dataclass
class loggingConfig:
    verbose: bool = True
    logInterval: int = 10
    logLevel: str = 'info'
    reportTrainLoss: bool = True
    reportValidationLoss: bool = True
