from dataclasses import dataclass
import numpy as np

@dataclass
class sampleBatch:
    x: np.ndarray
    regionNames: np.ndarray
    normals: np.ndarray | None = None
    weights: np.ndarray | None = None
