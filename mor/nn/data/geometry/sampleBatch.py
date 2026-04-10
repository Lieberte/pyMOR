from dataclasses import dataclass
import numpy as np

@dataclass
class sampleBatch:
    x: np.ndarray
    regionNames: np.ndarray
    normals: np.ndarray | None = None
    weights: np.ndarray | None = None

    @property
    def size(self) -> int:
        return int(self.x.shape[0])

    def withWeights(self, weights: np.ndarray) -> 'sampleBatch':
        return sampleBatch(x=self.x, regionNames=self.regionNames, normals=self.normals, weights=weights)

    def withNormals(self, normals: np.ndarray) -> 'sampleBatch':
        return sampleBatch(x=self.x, regionNames=self.regionNames, normals=normals, weights=self.weights)

    def subset(self, mask: np.ndarray) -> 'sampleBatch':
        return sampleBatch(
            x=self.x[mask],
            regionNames=self.regionNames[mask],
            normals=None if self.normals is None else self.normals[mask],
            weights=None if self.weights is None else self.weights[mask],
        )

    def shuffled(self) -> 'sampleBatch':
        order = np.random.permutation(self.size)
        return self.subset(order)

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
