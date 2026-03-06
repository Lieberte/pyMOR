import dataclasses
from dataclasses import dataclass

from mor.algorithm import algorithmRegistry
from mor.backends import backendRegistry
from mor.operators import matrixOperator
from mor.solvers import solverRegistry
from mor.models.lti import ltiModel


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
    solverInfo: dict = dataclasses.field(default_factory=dict)

class balancedTruncationReductor:
    def __init__(self, globalBackendName: str | None = None, **kwargs):
        self.lyapunovSolver = None
        self.localBackend = None
        self.svdAlgorithm = None
        self.backendName = globalBackendName
        self.options = kwargs

    def updateOptions(self, A: matrixOperator, B: matrixOperator, C: matrixOperator, E: matrixOperator | None = None, isContinuous: bool = True):
        backendName = self.backendName or A.backendName
        import mor.algorithm.lyapunov
        self.lyapunovSolver = solverRegistry.get(solverType='lyapunov', name='unified', forceOptions=self.options, backendName=backendName, A=A, E=E, B=B, C=C, isContinuous=isContinuous)
        self.localBackend = backendRegistry.get(backendName)
        self.svdAlgorithm = algorithmRegistry.get(category='svd', variant='auto', forceOptions=self.options, backendName=backendName, A=A, E=E, B=B)

    def reduce(self, model: ltiModel, *, order: int | None = None, maxError: float | None = None, isContinuous: bool = True) -> reducedSystem:
        self.updateOptions(model.A, model.B, model.C, model.E, isContinuous=isContinuous)
        backend = self.localBackend
        offset, n = self.options.get('offset', 1e-08), model.A.shape[0]
        if model.E is not None: aRegData = model.A.data - offset * model.E.data
        else: aRegData = model.A.data - offset * backend.array.eye(n, dtype=model.A.dtype)
        aNorm = backend.linalg.norm(aRegData)
        scalingFactor = float(aNorm) if aNorm > 1e6 else 1.0
        AScaled = matrixOperator(aRegData / scalingFactor, backendName=backend.name)
        BScaled = matrixOperator(model.B.data / backend.array.sqrt(scalingFactor), backendName=backend.name)
        CScaled = matrixOperator(model.C.data / backend.array.sqrt(scalingFactor), backendName=backend.name)
        EScaled = model.E
        ZcRaw = self.lyapunovSolver.solve(AScaled, EScaled, BScaled)
        ZoRaw = self.lyapunovSolver.solve(AScaled.T, EScaled.T if EScaled else None, CScaled.T)
        from mor.operators.lowRank import lowRankOperator
        Zc = ZcRaw.left if isinstance(ZcRaw, lowRankOperator) else backend.linalg.robustSqrtFactor(ZcRaw.data, name="Controllability Gramian")
        Zo = ZoRaw.left if isinstance(ZoRaw, lowRankOperator) else backend.linalg.robustSqrtFactor(ZoRaw.data, name="Observability Gramian")
        Zo_data = backend.array.toArray(Zo)
        Zc_data = backend.array.toArray(Zc)
        M = backend.linalg.dot(backend.linalg.transpose(Zo_data), Zc_data)
        if hasattr(M, 'to') and hasattr(model.A.data, 'dtype'): M = M.to(model.A.data.dtype)
        M_op = matrixOperator(M, backendName=backend.name)
        U, S, Vt = self.svdAlgorithm.decompose(M_op, rank=order, tol=maxError)
        hsv, order = S, backend.array.size(S)
        if order == 0: raise ValueError("All Hankel Singular Values were truncated.")
        srInv = backend.array.diag(1.0 / backend.array.sqrt(S))
        vProj = backend.linalg.dot(Zc, backend.linalg.dot(backend.linalg.transpose(Vt), srInv))
        wProj = backend.linalg.dot(Zo, backend.linalg.dot(U, srInv))
        wProjT = backend.linalg.transpose(wProj)
        bData, cData = model.B.data, model.C.data
        if backend.array.ndim(bData) == 1: bData = backend.array.reshape(bData, (-1, 1))
        if backend.array.ndim(cData) == 1: cData = backend.array.reshape(cData, (1, -1))
        if hasattr(bData, 'to') and hasattr(model.A.data, 'dtype'):
            bData, cData = bData.to(model.A.data.dtype), cData.to(model.A.data.dtype)
        Ar, Br, Cr = backend.linalg.dot(wProjT, model.A.apply(vProj)), backend.linalg.dot(wProjT, bData), backend.linalg.dot(cData, vProj)
        nOut, nIn = model.C.shape[0], model.B.shape[1] if backend.array.ndim(model.B.data) > 1 else 1
        Dr = model.D.data if model.D is not None else backend.array.zeros((nOut, nIn), dtype=model.A.dtype)
        if hasattr(Dr, 'to') and hasattr(model.A.data, 'dtype'): Dr = Dr.to(model.A.data.dtype)
        Er = backend.linalg.dot(wProjT, model.E.apply(vProj)) if model.E is not None else backend.linalg.dot(wProjT, vProj)
        return reducedSystem(
            isContinuous=isContinuous, 
            Ar=matrixOperator(Ar, backendName=backend.name), 
            Br=matrixOperator(Br, backendName=backend.name), 
            Cr=matrixOperator(Cr, backendName=backend.name), 
            Dr=matrixOperator(Dr, backendName=backend.name), 
            Er=matrixOperator(Er, backendName=backend.name), 
            hsv=backend.array.toNumpy(hsv).tolist(), 
            order=order,
            solverInfo={
                'backend': backend.name,
                'lyapunov': getattr(self.lyapunovSolver, '_lastAlgorithmName', 'unknown'),
                'svd': self.svdAlgorithm.__class__.__name__
            }
        )
