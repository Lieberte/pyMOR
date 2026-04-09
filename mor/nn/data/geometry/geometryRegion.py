from dataclasses import dataclass

@dataclass(frozen=True)
class geometryRegion:
    name: str
    kind: str
    dim: int
