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

    @property
    def model(self) -> ltiModel:
        return ltiModel(
            self.Ar,
            self.Br,
            self.Cr,
            self.Dr,
            self.Er,
            backendName=self.Ar.backendName,
            isContinuous=self.isContinuous,
        )

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
        self.svdAlgorithm = None

    def reduce(self, model: ltiModel, *, order: int | None = None, maxError: float | None = None, isContinuous: bool | None = None) -> reducedSystem:
        isContinuous = model.isContinuous if isContinuous is None else isContinuous
        self.updateOptions(model.A, model.B, model.C, model.E, isContinuous=isContinuous)
        backend = self.localBackend
        offset, n = self.options.get('offset', 0.0), model.A.shape[0]
        aData = model.A.toBackendData()
        bData = model.B.toBackendData()
        cData = model.C.toBackendData()
        eData = model.E.toBackendData() if model.E is not None else None
        if offset != 0 and eData is not None:
            aRegData = aData - offset * eData
        elif offset != 0:
            aRegData = aData - offset * backend.array.eye(n, dtype=model.A.dtype)
        else:
            aRegData = aData
        aNorm = backend.linalg.norm(aRegData)
        scalingFactor = float(aNorm) if isContinuous and aNorm > 1e6 else 1.0
        AScaled = matrixOperator(aRegData / scalingFactor, backendName=backend.name)
        BScaled = matrixOperator(bData / backend.array.sqrt(scalingFactor), backendName=backend.name)
        CScaled = matrixOperator(cData / backend.array.sqrt(scalingFactor), backendName=backend.name)
        EScaled = model.E
        ZcRaw = self.lyapunovSolver.solve(AScaled, EScaled, BScaled)
        ZoRaw = self.lyapunovSolver.solve(AScaled.T, EScaled.T if EScaled is not None else None, CScaled.T)
        from mor.operators.lowRank import lowRankOperator
        Zc = ZcRaw.left if isinstance(ZcRaw, lowRankOperator) else backend.linalg.robustSqrtFactor(ZcRaw.toBackendData(), name="Controllability Gramian")
        Zo = ZoRaw.left if isinstance(ZoRaw, lowRankOperator) else backend.linalg.robustSqrtFactor(ZoRaw.toBackendData(), name="Observability Gramian")
        Zo_data = backend.array.toArray(Zo)
        Zc_data = backend.array.toArray(Zc)
        M = backend.linalg.dot(backend.linalg.transpose(Zo_data), Zc_data)
        if hasattr(M, 'to') and hasattr(aData, 'dtype'): M = M.to(aData.dtype)
        M_op = matrixOperator(M, backendName=backend.name)
        self.svdAlgorithm = algorithmRegistry.get(
            category='svd',
            variant='auto',
            forceOptions=self.options,
            backendName=backend.name,
            xOperator=M_op,
            rank=order,
        )
        U, S, Vt = self.svdAlgorithm.decompose(M_op, rank=order, tol=maxError)
        hsv, order = S, backend.array.size(S)
        if order == 0: raise ValueError("All Hankel Singular Values were truncated.")
        srInv = backend.array.diag(1.0 / backend.array.sqrt(S))
        vProj = backend.linalg.dot(Zc, backend.linalg.dot(backend.linalg.transpose(Vt), srInv))
        wProj = backend.linalg.dot(Zo, backend.linalg.dot(U, srInv))
        wProjT = backend.linalg.transpose(wProj)
        if backend.array.ndim(bData) == 1: bData = backend.array.reshape(bData, (-1, 1))
        if backend.array.ndim(cData) == 1: cData = backend.array.reshape(cData, (1, -1))
        if hasattr(bData, 'to') and hasattr(aData, 'dtype'):
            bData, cData = bData.to(aData.dtype), cData.to(aData.dtype)
        Ar, Br, Cr = backend.linalg.dot(wProjT, model.A.apply(vProj)), backend.linalg.dot(wProjT, bData), backend.linalg.dot(cData, vProj)
        nOut, nIn = model.outputDim, model.inputDim
        Dr = model.D.toBackendData() if model.D is not None else backend.array.zeros((nOut, nIn), dtype=model.A.dtype)
        if hasattr(Dr, 'to') and hasattr(aData, 'dtype'): Dr = Dr.to(aData.dtype)
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
