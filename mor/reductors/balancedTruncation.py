import numpy as np
from dataclasses import dataclass
from typing import Optional, Any

from mor.algorithm import algorithmRegistry
from mor.backends import backendRegistry
from mor.operators import matrixOperator
from mor.solvers import solverRegistry


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
    def __init__(self, globalBackendName: str | None = None, **kwargs):
        self.lyapunovSolver = None
        self.localBackend = None
        self.svdAlgorithm = None
        self._backendName = globalBackendName
        self.options = kwargs

    def _updateOptions(self, A: matrixOperator, B: matrixOperator, C: matrixOperator, E: matrixOperator | None = None, isContinuous: bool = True):
        self.lyapunovSolver = solverRegistry.get(
            solverType='lyapunov',
            variant='auto',
            forceOptions=self.options,
            A=A, E=E, B=B, C=C,
            isContinuous=isContinuous
        )
        self.localBackend = backendRegistry.get(self._backendName)
        self.svdAlgorithm = algorithmRegistry.get(
            category='svd',
            variant='auto',
            forceOptions=self.options,
            A=A, E=E, B=B
        )

    def reduce(self, A: matrixOperator, B: matrixOperator, C: matrixOperator, D: matrixOperator | None = None, E: matrixOperator | None = None, *, order: int | None = None, maxError: float | None = None, isContinuous: bool = True) -> reducedSystem:
        self._updateOptions(A, B, C, E, isContinuous=isContinuous)
        backend = self.localBackend
        Zc, Zo, eEff = self.lyapunovSolver.solveControllabilityAndObservability(A, E, B, C)
        # TODO: Ensure all intermediate calculations stay on the target backend device
        M = backend.linalg.dot(Zo.T, Zc)
        mOp = matrixOperator(M, backendName=backend.name)
        U, S, Vt = self.svdAlgorithm.decompose(mOp, fullMatrices=False)
        hsv = S
        # TODO: Add truncation strategy further as an algorithm
        if order is None and maxError is not None:
            for r in range(len(hsv), 0, -1):
                if 2 * backend.array.sum(hsv[r:]) <= maxError:
                    order = r
                    break
            else:
                raise ValueError("Maximum error threshold not met")
        order = min(order, len(hsv)) if order is not None else len(hsv)
        srInv = backend.array.diag(1.0 / backend.array.sqrt(S[:order]))
        vProj = backend.linalg.dot(Zc, backend.linalg.dot(Vt[:order].T, srInv))
        wProj = backend.linalg.dot(Zo, backend.linalg.dot(U[:, :order], srInv))
        # TODO: matrixOperator should provide a way to get data in backend-native format instead of toNumpy
        bData = B.data
        cData = C.data
        if backend.array.ndim(bData) == 1:
            bData = backend.reshape(bData, (-1, 1))
        if backend.array.ndim(cData) == 1:
            cData = backend.reshape(cData, (1, -1))
        av = A.apply(vProj)
        Ar = backend.linalg.dot(wProj.T, av)
        Br = backend.linalg.dot(wProj.T, bData)
        Cr = backend.linalg.dot(cData, vProj)
        nOut = C.shape[0]
        nIn = B.shape[1] if backend.array.ndim(B.data) > 1 else 1
        Dr = D.data if D is not None else backend.array.zeros((nOut, nIn), dtype=A.dtype)
        ev = eEff.apply(vProj)
        Er = backend.linalg.dot(wProj.T, ev)
        return reducedSystem(isContinuous=isContinuous, Ar=matrixOperator(Ar, backendName=backend.name), Br=matrixOperator(Br, backendName=backend.name), Cr=matrixOperator(Cr, backendName=backend.name), Dr=matrixOperator(Dr, backendName=backend.name), Er=matrixOperator(Er, backendName=backend.name), hsv=backend.array.toNumpy(hsv[:order]).tolist(), order=order)
