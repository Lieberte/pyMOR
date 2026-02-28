import dataclasses
from typing import Optional, Any

from mor.algorithm import algorithmRegistry
from mor.backends import backendRegistry
from mor.operators import matrixOperator
from mor.solvers import solverRegistry


@dataclasses.dataclass
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
        self.backendName = globalBackendName
        self.options = kwargs

    def updateOptions(self, A: matrixOperator, B: matrixOperator, C: matrixOperator, E: matrixOperator | None = None, isContinuous: bool = True):
        lyapOptions, svdOptions = self.options.copy(), self.options.copy()
        if 'variant' in self.options: svdOptions.pop('variant')
        self.lyapunovSolver = solverRegistry.get(solverType='lyapunov', variant='auto', forceOptions=lyapOptions, A=A, E=E, B=B, C=C, isContinuous=isContinuous)
        self.localBackend = backendRegistry.get(self.backendName)
        self.svdAlgorithm = algorithmRegistry.get(category='svd', variant='auto', forceOptions=svdOptions, A=A, E=E, B=B)

    def reduce(self, A: matrixOperator, B: matrixOperator, C: matrixOperator, D: matrixOperator | None = None, E: matrixOperator | None = None, *, order: int | None = None, maxError: float | None = None, isContinuous: bool = True) -> reducedSystem:
        self.updateOptions(A, B, C, E, isContinuous=isContinuous)
        backend = self.localBackend
        offset, n = self.options.get('offset', 1e-08), A.shape[0]
        if E is not None: ARegData = A.data - offset * E.data
        else: ARegData = A.data - offset * backend.array.eye(n, dtype=A.dtype)
        aNorm, scalingFactor = backend.linalg.norm(ARegData), 1.0
        if aNorm > 1e6:
            scalingFactor = float(aNorm)
            AScaled = matrixOperator(ARegData / scalingFactor, backendName=backend.name)
            BScaled = matrixOperator(B.data / backend.array.sqrt(scalingFactor), backendName=backend.name)
            CScaled = matrixOperator(C.data / backend.array.sqrt(scalingFactor), backendName=backend.name)
            EScaled = matrixOperator(E.data, backendName=backend.name) if E is not None else None
        else: AScaled, BScaled, CScaled, EScaled = matrixOperator(ARegData, backendName=backend.name), B, C, E
        ZcRaw, ZoRaw, eEff = self.lyapunovSolver.solveControllabilityAndObservability(AScaled, EScaled, BScaled, CScaled)
        Zc = backend.linalg.robustSqrtFactor(ZcRaw.data, name="Controllability Gramian")
        Zo = backend.linalg.robustSqrtFactor(ZoRaw.data, name="Observability Gramian")
        M = backend.linalg.dot(Zo.T, Zc)
        U, S, Vt = self.svdAlgorithm.decompose(matrixOperator(M, backendName=backend.name), fullMatrices=False)
        hsv = S
        # TODO: Add truncation algorithm
        if order is None and maxError is not None:
            for r in range(len(hsv), 0, -1):
                if 2 * backend.array.sum(hsv[r:]) <= maxError:
                    order = r
                    break
            else: raise ValueError("Maximum error threshold not met")
        order = min(order, len(hsv)) if order is not None else len(hsv)
        if order == 0: raise ValueError("All Hankel Singular Values were truncated. System might be unstable or too stiff.")
        srInv = backend.array.diag(1.0 / backend.array.sqrt(S[:order]))
        vProj = backend.linalg.dot(Zc, backend.linalg.dot(Vt[:order].T, srInv))
        wProj = backend.linalg.dot(Zo, backend.linalg.dot(U[:, :order], srInv))
        bData, cData = B.data, C.data
        if backend.array.ndim(bData) == 1: bData = backend.array.reshape(bData, (-1, 1))
        if backend.array.ndim(cData) == 1: cData = backend.array.reshape(cData, (1, -1))
        Ar, Br, Cr = backend.linalg.dot(wProj.T, A.apply(vProj)), backend.linalg.dot(wProj.T, bData), backend.linalg.dot(cData, vProj)
        nOut, nIn = C.shape[0], B.shape[1] if backend.array.ndim(B.data) > 1 else 1
        Dr = D.data if D is not None else backend.array.zeros((nOut, nIn), dtype=A.dtype)
        Er = backend.linalg.dot(wProj.T, E.apply(vProj)) if E is not None else backend.linalg.dot(wProj.T, vProj)
        return reducedSystem(isContinuous=isContinuous, Ar=matrixOperator(Ar, backendName=backend.name), Br=matrixOperator(Br, backendName=backend.name), Cr=matrixOperator(Cr, backendName=backend.name), Dr=matrixOperator(Dr, backendName=backend.name), Er=matrixOperator(Er, backendName=backend.name), hsv=backend.array.toNumpy(hsv[:order]).tolist(), order=order)
