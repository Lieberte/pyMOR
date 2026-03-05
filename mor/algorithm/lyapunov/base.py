from mor.operators import matrixOperator
from mor.backends import backendRegistry

class backendLyapunovAlgorithm:
    def __init__(self, backendName: str | None = None, **kwargs):
        self.localBackend, self.options = backendRegistry.get(backendName), kwargs

    def prepareQ(self, B: matrixOperator):
        backend, bData = self.localBackend, B.data
        if backend.array.ndim(bData) == 1: bData = backend.array.reshape(bData, (-1, 1))
        return -backend.linalg.dot(bData, backend.linalg.transpose(bData))
