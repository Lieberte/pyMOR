from mor.operators import matrixOperator
from mor.algorithm.registry import registerAlgorithm
from mor.backends import backendRegistry

class backendLyapunovAlgorithm:
    def __init__(self, backendName: str | None = None, **kwargs):
        self.localBackend = backendRegistry.get(backendName)
        self.options = kwargs

    def prepareQ(self, B: matrixOperator):
        backend = self.localBackend
        bData = B.data
        if backend.array.ndim(bData) == 1:
            bData = backend.array.reshape(bData, (-1, 1))
        return backend.linalg.dot(bData, bData.T)

@registerAlgorithm('lyapunov', 'continuousHr')
class continuousHrAlgorithm(backendLyapunovAlgorithm):
    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator):
        return self.localBackend.lyapunov.solveContinuous(A.data, self.prepareQ(B))

@registerAlgorithm('lyapunov', 'continuousHrGeneralized')
class continuousHrGeneralizedAlgorithm(backendLyapunovAlgorithm):
    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator):
        return self.localBackend.lyapunov.solveContinuousGeneralized(A.data, E.data, self.prepareQ(B))

@registerAlgorithm('lyapunov', 'discreteHr')
class discreteHrAlgorithm(backendLyapunovAlgorithm):
    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator):
        return self.localBackend.lyapunov.solveDiscrete(A.data, self.prepareQ(B))

@registerAlgorithm('lyapunov', 'discreteHrGeneralized')
class discreteHrGeneralizedAlgorithm(backendLyapunovAlgorithm):
    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator):
        return self.localBackend.lyapunov.solveDiscreteGeneralized(A.data, E.data, self.prepareQ(B))
