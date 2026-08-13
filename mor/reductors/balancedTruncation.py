import dataclasses
from numbers import Integral, Real
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
        self._validateReductionRequest(order, maxError)
        if isContinuous and self.options.get('unstableHandling') == 'preserve':
            splitResult = self._reduceSymmetricUnstableModel(model, order=order, maxError=maxError)
            if splitResult is not None:
                return splitResult
        self._validateStability(model, isContinuous)
        self.updateOptions(model.A, model.B, model.C, model.E, isContinuous=isContinuous)
        backend = self.localBackend
        offset, n = self.options.get('offset', 0.0), model.A.shape[0]
        bData = model.B.toBackendData()
        cData = model.C.toBackendData()
        canScaleWithoutChangingOperatorForm = isinstance(model.A, matrixOperator)
        needsMaterializedA = offset != 0 or canScaleWithoutChangingOperatorForm
        aData = model.A.toBackendData() if needsMaterializedA else None
        if offset != 0:
            eData = model.E.toBackendData() if model.E is not None else backend.array.eye(n, dtype=model.A.dtype)
            aRegData = aData - offset * eData
        else:
            aRegData = aData
        aNorm = backend.linalg.norm(aRegData) if aRegData is not None else 0.0
        scalingFactor = float(aNorm) if isContinuous and aNorm > 1e6 else 1.0
        if offset == 0 and scalingFactor == 1.0:
            AScaled, BScaled, CScaled = model.A, model.B, model.C
        else:
            AScaled = matrixOperator(aRegData / scalingFactor, backendName=backend.name)
            BScaled = matrixOperator(bData / backend.array.sqrt(scalingFactor), backendName=backend.name)
            CScaled = matrixOperator(cData / backend.array.sqrt(scalingFactor), backendName=backend.name)
        EScaled = model.E
        ZcRaw = self.lyapunovSolver.solve(AScaled, EScaled, BScaled)
        controllabilityInfo = dict(self.lyapunovSolver.lastInfo)
        ZoRaw = self.lyapunovSolver.solve(AScaled.T, EScaled.T if EScaled is not None else None, CScaled.T)
        observabilityInfo = dict(self.lyapunovSolver.lastInfo)
        self._validateGramianSolve('controllability', controllabilityInfo)
        self._validateGramianSolve('observability', observabilityInfo)
        from mor.operators.lowRank import lowRankOperator
        Zc = ZcRaw.left if isinstance(ZcRaw, lowRankOperator) else backend.linalg.robustSqrtFactor(ZcRaw.toBackendData(), name="Controllability Gramian")
        Zo = ZoRaw.left if isinstance(ZoRaw, lowRankOperator) else backend.linalg.robustSqrtFactor(ZoRaw.toBackendData(), name="Observability Gramian")
        Zo_data = backend.array.toArray(Zo)
        Zc_data = backend.array.toArray(Zc)
        weightedZc = model.E.apply(Zc_data) if model.E is not None else Zc_data
        M = backend.linalg.dot(backend.linalg.transpose(Zo_data), weightedZc)
        if hasattr(M, 'to') and model.A.dtype is not None: M = M.to(model.A.dtype)
        M_op = matrixOperator(M, backendName=backend.name)
        self.svdAlgorithm = algorithmRegistry.get(
            category='svd',
            variant='auto',
            forceOptions=self.options,
            backendName=backend.name,
            xOperator=M_op,
            rank=None,
        )
        U, S, Vt = self.svdAlgorithm.decompose(M_op, rank=None, tol=None)
        fullHsv = backend.array.toNumpy(S)
        numericalRank = self._numericalRank(fullHsv)
        if numericalRank == 0:
            raise ValueError("The system has no positive numerical Hankel singular values.")
        retainedOrder = self._selectOrder(fullHsv[:numericalRank], order, maxError)
        U, S, Vt = U[:, :retainedOrder], S[:retainedOrder], Vt[:retainedOrder, :]
        hsv, order = S, retainedOrder
        srInv = backend.array.diag(1.0 / backend.array.sqrt(S))
        vProj = backend.linalg.dot(Zc, backend.linalg.dot(backend.linalg.transpose(Vt), srInv))
        wProj = backend.linalg.dot(Zo, backend.linalg.dot(U, srInv))
        wProjT = backend.linalg.transpose(wProj)
        if backend.array.ndim(bData) == 1: bData = backend.array.reshape(bData, (-1, 1))
        if backend.array.ndim(cData) == 1: cData = backend.array.reshape(cData, (1, -1))
        if hasattr(bData, 'to') and model.A.dtype is not None:
            bData, cData = bData.to(model.A.dtype), cData.to(model.A.dtype)
        Ar, Br, Cr = backend.linalg.dot(wProjT, model.A.apply(vProj)), backend.linalg.dot(wProjT, bData), backend.linalg.dot(cData, vProj)
        nOut, nIn = model.outputDim, model.inputDim
        Dr = model.D.toBackendData() if model.D is not None else backend.array.zeros((nOut, nIn), dtype=model.A.dtype)
        if hasattr(Dr, 'to') and model.A.dtype is not None: Dr = Dr.to(model.A.dtype)
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
                'svd': self.svdAlgorithm.__class__.__name__,
                'controllability': controllabilityInfo,
                'observability': observabilityInfo,
                'hankelErrorBound': float(2.0 * fullHsv[retainedOrder:numericalRank].sum()),
                'descriptorKind': model.classifyDescriptor(
                    maxSize=self.options.get('descriptorRankCheckMaxSize', 512)
                ),
            }
        )

    def _reduceSymmetricUnstableModel(self, model: ltiModel, *, order: int | None, maxError: float | None):
        import numpy as np
        from scipy import linalg

        if self.backendName not in (None, 'scipy') or not isinstance(model.A, matrixOperator):
            raise NotImplementedError("Unstable-mode preservation currently requires scipy matrix operators.")
        A = np.asarray(model.A.toNumpy())
        E = np.eye(model.order, dtype=A.dtype) if model.E is None else np.asarray(model.E.toNumpy())
        symmetryTolerance = self.options.get('symmetryTolerance', 1e-10)
        if not np.allclose(A, A.T, rtol=symmetryTolerance, atol=symmetryTolerance):
            raise NotImplementedError("Unstable-mode preservation currently requires a symmetric A matrix.")
        if not np.allclose(E, E.T, rtol=symmetryTolerance, atol=symmetryTolerance):
            raise NotImplementedError("Unstable-mode preservation currently requires a symmetric E matrix.")
        try:
            eigenvalues, basis = linalg.eigh(A, E)
        except linalg.LinAlgError as error:
            raise NotImplementedError("Unstable-mode preservation requires a positive-definite E matrix.") from error
        unstableMask = eigenvalues >= 0
        unstableCount = int(np.count_nonzero(unstableMask))
        if unstableCount == 0:
            return None
        if order is None:
            raise ValueError("An explicit order is required when preserving unstable modes.")
        if order <= unstableCount:
            raise ValueError(
                f"Requested order {order} must exceed the {unstableCount} preserved unstable modes."
            )

        stableBasis = basis[:, ~unstableMask]
        unstableBasis = basis[:, unstableMask]
        B = np.asarray(model.B.toNumpy(), dtype=np.result_type(A.dtype, np.float64))
        C = np.asarray(model.C.toNumpy(), dtype=np.result_type(A.dtype, np.float64))
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        stableModel = ltiModel(
            matrixOperator(np.diag(eigenvalues[~unstableMask]), backendName='scipy'),
            matrixOperator(stableBasis.T @ B, backendName='scipy'),
            matrixOperator(C @ stableBasis, backendName='scipy'),
            backendName='scipy',
            isContinuous=True,
        )
        stableOptions = {
            key: value for key, value in self.options.items()
            if key not in ('unstableHandling', 'symmetryTolerance', 'checkStability')
        }
        stableResult = balancedTruncationReductor(
            globalBackendName='scipy', **stableOptions
        ).reduce(stableModel, order=order - unstableCount, maxError=maxError)

        unstableA = np.diag(eigenvalues[unstableMask])
        unstableB = unstableBasis.T @ B
        unstableC = C @ unstableBasis
        stableA = stableResult.Ar.toNumpy()
        stableE = stableResult.Er.toNumpy()
        Ar = linalg.block_diag(unstableA, stableA)
        Er = linalg.block_diag(np.eye(unstableCount), stableE)
        Br = np.vstack((unstableB, stableResult.Br.toNumpy()))
        Cr = np.hstack((unstableC, stableResult.Cr.toNumpy()))
        Dr = model.D.toNumpy() if model.D is not None else np.zeros((model.outputDim, model.inputDim))
        solverInfo = dict(stableResult.solverInfo)
        solverInfo.update(
            unstableHandling='preserve',
            preservedUnstableModes=unstableCount,
            stableReducedOrder=stableResult.order,
        )
        return reducedSystem(
            isContinuous=True,
            Ar=matrixOperator(Ar, backendName='scipy'),
            Br=matrixOperator(Br, backendName='scipy'),
            Cr=matrixOperator(Cr, backendName='scipy'),
            Dr=matrixOperator(Dr, backendName='scipy'),
            Er=matrixOperator(Er, backendName='scipy'),
            hsv=stableResult.hsv,
            order=order,
            solverInfo=solverInfo,
        )

    def _validateReductionRequest(self, order: int | None, maxError: float | None) -> None:
        if order is not None and maxError is not None:
            raise ValueError("Specify either order or maxError, not both.")
        if order is not None and (not isinstance(order, Integral) or isinstance(order, bool) or order <= 0):
            raise ValueError("order must be a positive integer.")
        if maxError is not None and (not isinstance(maxError, Real) or isinstance(maxError, bool) or maxError < 0):
            raise ValueError("maxError must be a non-negative real number.")

    def _validateStability(self, model: ltiModel, isContinuous: bool) -> None:
        if not self.options.get('checkStability', True):
            return
        if not isContinuous:
            from mor.algorithm.lyapunov_discrete import estimateDiscreteRadius
            radius = estimateDiscreteRadius(model.A, model.E)
            if radius is not None and radius >= 1.0:
                raise ValueError(f"Balanced truncation requires a stable discrete model, got spectral radius {radius:.6g}.")
            return
        n = model.order
        import numpy as np

        descriptorLimit = self.options.get('descriptorRankCheckMaxSize', 512)
        descriptorKind = model.classifyDescriptor(maxSize=descriptorLimit)
        if descriptorKind == 'singular':
            raise NotImplementedError(
                "Balanced truncation for singular descriptor systems requires spectral projectors; "
                "use a descriptor-specific or frequency-response reductor."
            )
        stabilityLimit = self.options.get('stabilityCheckMaxSize', 300)
        if n > stabilityLimit and not model.A.isDiagonal:
            return
        try:
            eigenvalues = model.poles()
            maxRealPart = float(np.max(np.real(eigenvalues)))
        except (ValueError, TypeError, np.linalg.LinAlgError):
            return
        if maxRealPart >= 0:
            raise ValueError(f"Balanced truncation requires a stable continuous model, got spectral abscissa {maxRealPart:.6g}.")

    def _validateGramianSolve(self, name: str, info: dict) -> None:
        if info.get('converged', True) or self.options.get('allowUnconverged', False):
            return
        raise RuntimeError(
            f"The {name} Gramian solve did not converge "
            f"(algorithm={info.get('algorithm')}, iterations={info.get('iterations')}, residual={info.get('residual')})."
        )

    def _numericalRank(self, singularValues) -> int:
        import numpy as np
        if singularValues.size == 0:
            return 0
        threshold = max(singularValues.shape) * np.finfo(singularValues.dtype).eps * singularValues[0]
        return int(np.count_nonzero(singularValues > threshold))

    def _selectOrder(self, singularValues, order: int | None, maxError: float | None) -> int:
        import numpy as np
        numericalRank = singularValues.size
        if order is not None:
            if order > numericalRank:
                raise ValueError(f"Requested order {order} exceeds numerical Hankel rank {numericalRank}.")
            return int(order)
        if maxError is None:
            return int(numericalRank)
        tailBounds = 2.0 * np.concatenate(([singularValues.sum()], np.cumsum(singularValues[::-1])[:-1][::-1], [0.0]))
        candidates = np.flatnonzero(tailBounds <= maxError)
        retainedOrder = int(candidates[0]) if candidates.size else numericalRank
        if retainedOrder == 0:
            raise ValueError("maxError truncates all positive Hankel singular values.")
        return retainedOrder
