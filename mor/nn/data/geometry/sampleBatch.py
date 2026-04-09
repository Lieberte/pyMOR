from dataclasses import dataclass
import numpy as np

@dataclass
class sampleBatch:
    x: np.ndarray
    regionNames: np.ndarray
    normals: np.ndarray | None = None
    weights: np.ndarray | None = None

    @classmethod
    def concat(cls, batches: list['sampleBatch']) -> 'sampleBatch':
        if not batches:
            raise ValueError('batches must not be empty')
        x = np.concatenate([batch.x for batch in batches], axis=0)
        regionNames = np.concatenate([batch.regionNames for batch in batches], axis=0)
        normals = None
        if all(batch.normals is not None for batch in batches):
            normals = np.concatenate([batch.normals for batch in batches], axis=0)
        weights = None
        if all(batch.weights is not None for batch in batches):
            weights = np.concatenate([batch.weights for batch in batches], axis=0)
        return cls(x=x, regionNames=regionNames, normals=normals, weights=weights)
