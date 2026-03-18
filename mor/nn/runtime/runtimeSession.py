from dataclasses import dataclass

@dataclass
class runtimeSession:
    backendName: str = 'torch'
    device: str = 'cpu'
    dtype: str = 'float32'
    seed: int | None = None
