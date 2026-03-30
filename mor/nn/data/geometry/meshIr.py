from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class meshIr:
    nodes: np.ndarray
    boundaryNodes: dict[str, np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]] = field(default_factory=dict)
    cells: list[tuple[str, np.ndarray]] | None = None
