from mor.operators import matrixOperator
from mor.algorithm.registry import registerAlgorithm
from mor.backends import backendRegistry

class backendLyapunovAlgorithm:
    def __init__(self, backendName: str | None = None, **kwargs):
        self.localBackend = backendRegistry.get(backendName)
        self.options = kwargs

    def _prepareQ(self, B: matrixOperator):
        backend = self.localBackend
        B_data = B.data
        if backend.ndim(B_data) == 1:
            B_data = backend.reshape(B_data, (-1, 1))
        return backend.linalg.dot(B_data, B_data.T)

@registerAlgorithm('lyapunov', 'continuousHr')
class continuousHrAlgorithm(backendLyapunovAlgorithm):
    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator):
        return self.localBackend.lyapunov.solveContinuous(A.data, self._prepareQ(B))

@registerAlgorithm('lyapunov', 'continuousHrGeneralized')
class continuousHrGeneralizedAlgorithm(backendLyapunovAlgorithm):
    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator):
        return self.localBackend.lyapunov.solveContinuousGeneralized(A.data, E.data, self._prepareQ(B))

@registerAlgorithm('lyapunov', 'discreteHr')
class discreteHrAlgorithm(backendLyapunovAlgorithm):
    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator):
        return self.localBackend.lyapunov.solveDiscrete(A.data, self._prepareQ(B))

@registerAlgorithm('lyapunov', 'discreteHrGeneralized')
class discreteHrGeneralizedAlgorithm(backendLyapunovAlgorithm):
    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator):
        return self.localBackend.lyapunov.solveDiscreteGeneralized(A.data, E.data, self._prepareQ(B))
