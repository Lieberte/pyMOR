from dataclasses import dataclass

@dataclass
class runtimeConfig:
    backendName: str = 'torch'
    deviceName: str = 'cpu'
    randomSeed: int | None = None
