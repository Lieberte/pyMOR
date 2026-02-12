from .lyapunov import lyapunovSolverBase
from mor.operators import matrixOperator
from mor.algorithm.lyapunov import solveLyapunovLr, shiftComputationOptions


class continuousLrLyapunovSolver(lyapunovSolverBase):

    def solve(self, a: matrixOperator, b: matrixOperator) -> matrixOperator:
        self._validateInputs(a, b)

        tol = self.options.get('tol', 1e-10)
        maxIter = self.options.get('maxIter', 500)
        trans = self.options.get('trans', False)
        initMaxiter = self.options.get('initMaxiter', 20)
        subspaceColumns = self.options.get('subspaceColumns', 6)

        shiftOpts = shiftComputationOptions(
            initMaxiter=initMaxiter,
            subspaceColumns=subspaceColumns
        )

        zData = solveLyapunovLr(
            a, b,
            trans=trans,
            backendName=self.backendName,
            tol=tol,
            maxIter=maxIter,
            shiftOptions=shiftOpts
        )

        return matrixOperator(zData, backendName=self.backendName)
