from .lyapunov import baseGeneralizedLyapunovSolver
from mor.operators import matrixOperator
from mor.algorithm.lyapunov import solveLyapunovLrDiscreteGeneralized

class discreteLrGeneralizedLyapunovSolver(baseGeneralizedLyapunovSolver):
    def solve(self,a: matrixOperator,e: matrixOperator,b: matrixOperator) -> matrixOperator:
        self._validateInputs(a, e, b)
        maxIter = self.options.get('maxIter', 200)
        tol = self.options.get('tol', 1e-10)
        maxRank = self.options.get('maxRank', None)
        zData = solveLyapunovLrDiscreteGeneralized( a, e, b,backendName=self.backendName,maxIter=maxIter,tol=tol,maxRank=maxRank)
        return matrixOperator(zData, backendName=self.backendName)
