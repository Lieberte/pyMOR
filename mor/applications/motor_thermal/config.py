from dataclasses import dataclass, field
from typing import Any


@dataclass
class motorThermalReductionConfig:
    rank: int | None = None
    tol: float | None = None
    method: str = "auto"
    backendName: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.rank is not None and self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.tol is not None and self.tol <= 0:
            raise ValueError("tol must be positive")
        if not self.method:
            raise ValueError("method must not be empty")
