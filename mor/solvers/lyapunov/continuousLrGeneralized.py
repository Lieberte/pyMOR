from .lyapunov import baseGeneralizedLyapunovSolver
from mor.operators import matrixOperator
from mor.algorithm.lyapunov import solveLyapunovLrGeneralized, shiftComputationOptions

class continuousLrGeneralizedLyapunovSolver(baseGeneralizedLyapunovSolver):
    def solve(self, a: matrixOperator,e: matrixOperator,b: matrixOperator) -> matrixOperator:
        self._validateInputs(a, e, b)
        tol = self.options.get('tol', 1e-10)
        maxIter = self.options.get('maxIter', 500)
        trans = self.options.get('trans', False)
        initMaxiter = self.options.get('initMaxiter', 20)
        subspaceColumns = self.options.get('subspaceColumns', 6)
        shiftOpts = shiftComputationOptions(initMaxiter=initMaxiter,subspaceColumns=subspaceColumns)
        zData = solveLyapunovLrGeneralized(a, e, b,trans=trans,backendName=self.backendName,tol=tol,maxIter=maxIter,shiftOptions=shiftOpts)
        return matrixOperator(zData, backendName=self.backendName)
