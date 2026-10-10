from dataclasses import dataclass

@dataclass
class runtimeSession:
    backendName: str = 'torch'
    device: str = 'auto'
    dtype: str = 'float32'
    seed: int | None = None
