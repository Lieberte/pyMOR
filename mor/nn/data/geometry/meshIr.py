from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .utils import normalizeBoundaryMap
from .utils import validateNodes
from .utils.facetUtils import validateBoundaryFacesDict

@dataclass
class meshIr:
    nodes: np.ndarray
    boundaryNodes: dict[str, np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]] = field(default_factory=dict)
    cells: list[tuple[str, np.ndarray]] | None = None
    boundaryFaces: dict[str, list[tuple[str, np.ndarray]]] = field(default_factory=dict)

    def __post_init__(self):
        self.nodes = validateNodes(self.nodes)
        self.boundaryNodes = normalizeBoundaryMap(self.boundaryNodes, dim=self.nodes.shape[1])
        self.boundaryFaces = validateBoundaryFacesDict(self.boundaryFaces, self.nodes.shape[0])
