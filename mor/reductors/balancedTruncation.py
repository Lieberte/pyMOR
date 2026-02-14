import numpy as np
from dataclasses import dataclass
from typing import Optional

from mor.algorithm import algorithmRegistry
from mor.backends import backendRegistry
from mor.operators import matrixOperator
from mor.solvers.lyapunov import lyapunovRegistry

# TODO: call flow for lower layers
# - lyapunovRegistry.get(name=None, A, E, B, isContinuous) -> mor.solvers.lyapunov.registry._selectSolver
# - lyapunovSolver.solveControllabilityAndObservability(A, E, B, C) -> mor.solvers.lyapunov.lyapunov (base classes)
# - algorithmRegistry.get('svd', 'auto', A, E, B) -> mor.algorithm.registry._selectSVDVariant
# - svdAlgorithm.decompose(mOp) -> mor.algorithm.decompose.svd.staticSVD

@dataclass
class reducedSystem:
    isContinuous: bool
    Ar: matrixOperator
    Br: matrixOperator
    Cr: matrixOperator
    Dr: matrixOperator
    Er: matrixOperator
    hsv: list
    order: int

class balancedTruncationReductor:
    def __init__(self, globalBackendName: Optional[str] = None, **kwargs):
        self.lyapunovSolver = None
        self.localBackend = None
        self.svdAlgorithm = None
        self._backendName = globalBackendName
        self.options = kwargs

    def _updateOptions(self, A: matrixOperator, B: matrixOperator, C: matrixOperator, D: Optional[matrixOperator] = None, E: Optional[matrixOperator] = None, isContinuous: bool = True, **kwargs):
        self.lyapunovSolver = lyapunovRegistry.get(name=None, A=A, E=E, B=B, isContinuous=isContinuous)
        self.localBackend = backendRegistry.get(self._backendName)
        self.svdAlgorithm = algorithmRegistry.get('svd', 'auto', A=A, E=E, B=B)

    def reduce(self, A: matrixOperator, B: matrixOperator, C: matrixOperator, D: Optional[matrixOperator] = None, E: Optional[matrixOperator] = None, *, order: Optional[int] = None, maxError: Optional[float] = None, isContinuous: bool = True) -> reducedSystem:
        self._updateOptions(A, B, C, D, E, isContinuous=isContinuous)
        backendName = self.localBackend.name
        Zc, Zo, Eeff = self.lyapunovSolver.solveControllabilityAndObservability(A, E, B, C)
        M = self.localBackend.linalg.dot(Zo.T, Zc)
        mOp = matrixOperator(M, backendName=backendName)
        U, S, Vt = self.svdAlgorithm.decompose(mOp, fullMatrices=False)
        hsv = S
        if order is None and maxError is not None:
            for r in range(len(hsv), 0, -1):
                if 2 * np.sum(hsv[r:]) <= maxError:
                    order = r
                    break
            else:
                order = 1
        if order is None:
            order = len(hsv)
        order = min(order, len(hsv))
        srInv = np.diag(1.0 / np.sqrt(S[:order]))
        vProj = self.localBackend.linalg.dot(Zc, self.localBackend.linalg.dot(Vt[:order].T, srInv))
        wProj = self.localBackend.linalg.dot(Zo, self.localBackend.linalg.dot(U[:, :order], srInv))
        bData = B.toNumpy()
        cData = C.toNumpy()
        if bData.ndim == 1:
            bData = bData[:, np.newaxis]
        if cData.ndim == 1:
            cData = cData[np.newaxis, :]
        AV = A.apply(vProj)
        Ar = self.localBackend.linalg.dot(wProj.T, AV)
        Br = self.localBackend.linalg.dot(wProj.T, bData)
        Cr = self.localBackend.linalg.dot(cData, vProj)
        nOut = C.shape[0]
        nIn = B.shape[1] if B.ndim > 1 else 1
        Dr = D.toNumpy() if D is not None else np.zeros((nOut, nIn), dtype=A.dtype)
        EV = Eeff.apply(vProj)
        Er = self.localBackend.linalg.dot(wProj.T, EV)
        return reducedSystem(isContinuous=isContinuous,Ar=matrixOperator(Ar, backendName=backendName),Br=matrixOperator(Br, backendName=backendName),Cr=matrixOperator(Cr, backendName=backendName),Dr=matrixOperator(Dr, backendName=backendName),Er=matrixOperator(Er, backendName=backendName),hsv=hsv[:order].tolist(),order=order)
        
