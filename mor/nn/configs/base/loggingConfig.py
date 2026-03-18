from dataclasses import dataclass

@dataclass
class loggingConfig:
    verbose: bool = True
    logInterval: int = 10
