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
    boundaryNodeIndices: dict[str, np.ndarray] = field(default_factory=dict)
    pointData: dict[str, np.ndarray] = field(default_factory=dict)
    cellData: dict[str, list[np.ndarray]] = field(default_factory=dict)
    cellSets: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def __post_init__(self):
        self.nodes = validateNodes(self.nodes)
        numNodes = self.nodes.shape[0]
        cleanedBoundaryIndices: dict[str, np.ndarray] = {}
        for name, indices in self.boundaryNodeIndices.items():
            idx = np.unique(np.asarray(indices, dtype=int).reshape(-1))
            if idx.size > 0 and (idx[0] < 0 or idx[-1] >= numNodes):
                raise ValueError(f'boundaryNodeIndices[{name!r}] index out of range')
            cleanedBoundaryIndices[name] = idx
        self.boundaryNodeIndices = cleanedBoundaryIndices
        if not self.boundaryNodes and self.boundaryNodeIndices:
            self.boundaryNodes = {name: self.nodes[idx] for name, idx in self.boundaryNodeIndices.items()}
        self.boundaryNodes = normalizeBoundaryMap(self.boundaryNodes, dim=self.nodes.shape[1])
        self.boundaryFaces = validateBoundaryFacesDict(self.boundaryFaces, numNodes)
        cleanedPointData: dict[str, np.ndarray] = {}
        for name, values in self.pointData.items():
            arr = np.asarray(values)
            if arr.ndim == 0 or arr.shape[0] != numNodes:
                raise ValueError(f'pointData[{name!r}] must have first dimension {numNodes}')
            cleanedPointData[name] = arr
        self.pointData = cleanedPointData
        numCellBlocks = len(self.cells or [])
        cleanedCellData: dict[str, list[np.ndarray]] = {}
        for name, blocks in self.cellData.items():
            if len(blocks) != numCellBlocks:
                raise ValueError(f'cellData[{name!r}] must have {numCellBlocks} blocks')
            cleanedBlocks = []
            for blockIndex, values in enumerate(blocks):
                arr = np.asarray(values)
                numCells = int(np.asarray(self.cells[blockIndex][1]).shape[0])
                if arr.ndim == 0 or arr.shape[0] != numCells:
                    raise ValueError(f'cellData[{name!r}][{blockIndex}] must have first dimension {numCells}')
                cleanedBlocks.append(arr)
            cleanedCellData[name] = cleanedBlocks
        self.cellData = cleanedCellData
        cleanedCellSets: dict[str, list[np.ndarray]] = {}
        for name, blocks in self.cellSets.items():
            if len(blocks) != numCellBlocks:
                raise ValueError(f'cellSets[{name!r}] must have {numCellBlocks} blocks')
            cleanedBlocks = []
            for blockIndex, indices in enumerate(blocks):
                idx = np.unique(np.asarray(indices, dtype=int).reshape(-1))
                numCells = int(np.asarray(self.cells[blockIndex][1]).shape[0])
                if idx.size > 0 and (idx[0] < 0 or idx[-1] >= numCells):
                    raise ValueError(f'cellSets[{name!r}][{blockIndex}] index out of range')
                cleanedBlocks.append(idx)
            cleanedCellSets[name] = cleanedBlocks
        self.cellSets = cleanedCellSets
