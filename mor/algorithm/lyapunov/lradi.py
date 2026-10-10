from dataclasses import dataclass
from collections import OrderedDict
from numbers import Integral
from typing import Any, Optional

from mor.backends import backendRegistry
from mor.operators.operatorsBase import operatorBase
from mor.algorithm.registry import registerAlgorithm, algorithmRegistry
from .base import backendLyapunov

@dataclass(frozen=True, slots=True)
class shiftComputationOptions:
    initMaxIter: int = 20
    subspaceColumns: int = 6
    maxShifts: int = 20

@registerAlgorithm('lyapunov', 'lradi')
class lradi(backendLyapunov):
    # TODO: Implement more robust shift selection (e.g., Penzl's heuristic) for stiff systems
    def __init__(self, backendName: str | None = None, **kwargs):
        self.localBackend, self.options = backendRegistry.get(backendName), kwargs
        self.lastResidualHistory: list[float] = []
        self.lastIterations = 0
        self.lastShifts: Any = None
        self.lastConverged = False
        self.lastResidual: float | None = None
        self.lastTolerance: float | None = None

    def solve(self, A: operatorBase, E: operatorBase | None, B: operatorBase, trans: bool = False) -> Any:
        backend, n, bData = self.localBackend, A.shape[0], B.toBackendData()
        self.lastResidualHistory = []
        self.lastIterations = 0
        self.lastConverged = False
        self.lastResidual = None
        self.lastTolerance = None
        self.lastShifts = None
        self.validateIterationOptions()
        if trans:
            bData = backend.linalg.transpose(bData)
        if backend.array.ndim(bData) == 1: bData = backend.array.reshape(bData, (-1, 1))    
        tol = self.options.get('tol', 1e-10)
        maxIter = self.options.get('maxIter', 500)
        compressionColumns = self.options.get('compressionColumns', 32)
        self.validateColumnOption('compressionColumns', compressionColumns)
        factorCacheBytes = self.options.get('factorCacheBytes', 64 * 1024 ** 2)
        if not isinstance(factorCacheBytes, Integral) or isinstance(factorCacheBytes, bool) or factorCacheBytes < 0:
            raise ValueError('factorCacheBytes must be a non-negative integer')
        shiftOpts = self.options.get('shiftOptions')
        if not isinstance(shiftOpts, shiftComputationOptions): shiftOpts = shiftComputationOptions(**(shiftOpts if isinstance(shiftOpts, dict) else {}))
        for name in ('initMaxIter', 'subspaceColumns', 'maxShifts'):
            self.validateColumnOption(name, getattr(shiftOpts, name))
        shifts, zColumns, W, j, jShift = self.computeInitialShifts(A, E, bData, shiftOpts), [], backend.array.copy(bData), 0, 0
        shifts = shifts[:shiftOpts.maxShifts]
        self.lastShifts = shifts
        res = backend.specialized.gramMatrixNorm(W, backend)
        bTol = res * tol
        self.lastTolerance = float(bTol)
        self.lastResidualHistory = [float(res)]
        factorSolvers = OrderedDict()
        cachedBytes = 0
        while res > bTol and j < maxIter:
            sigma = shifts[jShift]
            sigmaReal, sigmaImag = backend.array.real(sigma), backend.array.imag(sigma)
            if backend.array.abs(sigmaImag) < 1e-14 * backend.array.abs(sigmaReal): sigma, sigmaImag = sigmaReal, 0.0
            shiftKey = complex(sigma)
            if shiftKey in factorSolvers:
                solveShift = factorSolvers.pop(shiftKey)
                factorSolvers[shiftKey] = solveShift
            else:
                linearSolver = algorithmRegistry.get(category='linear', variant=self.options.get('linearVariant', 'auto'), backendName=backend.name, A=A, E=E, shift=sigma, trans=trans)
                if hasattr(linearSolver, 'factorize'):
                    if len(factorSolvers) >= shiftOpts.maxShifts:
                        _, removed = factorSolvers.popitem(last=False)
                        cachedBytes -= removed.factorBytes
                    solveShift = linearSolver.factorize(A, E=E, shift=sigma, trans=trans)
                    factorBytes = getattr(solveShift, 'factorBytes', factorCacheBytes + 1)
                    while factorSolvers and cachedBytes + factorBytes > factorCacheBytes:
                        _, removed = factorSolvers.popitem(last=False)
                        cachedBytes -= removed.factorBytes
                    if factorBytes <= factorCacheBytes:
                        factorSolvers[shiftKey] = solveShift
                        cachedBytes += factorBytes
                else:
                    solveShift = lambda rhs: linearSolver.solve(A, rhs, E=E, shift=sigma, trans=trans)
            vData = solveShift(W)
            if sigmaImag == 0:
                s = sigmaReal
                W = W - (2 * s) * (E.apply(vData, trans=trans) if E is not None else vData)
                zColumns.append(vData * backend.array.sqrt(-2 * s))
                j += 1
            else:
                gs, d = -4 * sigmaReal, sigmaReal / sigmaImag if backend.array.abs(sigmaImag) > 1e-14 else 0.0
                if trans: vData = backend.array.conj(vData)
                U = backend.array.real(vData) + backend.array.imag(vData) * d
                W = W + gs * (E.apply(U, trans=trans) if E is not None else U)
                g = backend.array.sqrt(gs)
                zColumns.append(U * g)
                zColumns.append(backend.array.imag(vData) * (g * backend.array.sqrt(d**2 + 1)))
                j += 2
            jShift += 1
            res = backend.specialized.gramMatrixNorm(W, backend)
            self.lastResidualHistory.append(float(res))
            if jShift >= backend.array.size(shifts) and res > bTol and j < maxIter:
                shifts, jShift = self.updateShifts(A, E, vData, zColumns, shifts, shiftOpts, trans=trans), 0
            if jShift == 0:
                shifts = shifts[:shiftOpts.maxShifts]
                self.lastShifts = shifts
            totalColumns = sum(col.shape[1] for col in zColumns)
            if totalColumns > compressionColumns:
                Z_temp = self.compressFactor(backend.array.hstack(zColumns), n)
                zColumns = [Z_temp]
        self.lastIterations = j
        self.lastResidual = float(res)
        self.lastConverged = bool(res <= bTol)
        from mor.operators.lowRank import lowRankOperator
        return lowRankOperator(backend.array.hstack(zColumns) if zColumns else backend.array.zeros((n, 0), dtype=A.dtype), backendName=backend.name)

    def computeInitialShifts(self, A: operatorBase, E: operatorBase | None, bData: Any, options: shiftComputationOptions) -> Any:
        backend = self.localBackend
        def projectAndGetShifts(B: Any) -> Any:
            Q = backend.decomposition.qrOrthogonalize(B, backend)
            aProj = backend.linalg.dot(backend.linalg.transpose(Q), A.apply(Q))
            eProj = backend.linalg.dot(backend.linalg.transpose(Q), E.apply(Q)) if E is not None else backend.array.eyeLike(aProj)
            return self.filterStableShifts(backend.eigen.eigvalsGeneralized(aProj, eProj))
        shifts = projectAndGetShifts(bData)
        if backend.array.size(shifts) > 0: return shifts
        for _ in range(options.initMaxIter - 1):
            expanded = [bData]
            for alpha in (-1e-2, -1e-1, -1.0, -10.0):
                try:
                    linearSolver = algorithmRegistry.get(category='linear', variant='shifted', backendName=backend.name, A=A, E=E, shift=alpha, trans=False)
                    vData = linearSolver.solve(A, bData, E=E, shift=alpha, trans=False)
                    if backend.array.all(backend.array.isfinite(vData)): expanded.append(vData)
                except (backend.linalg.LinAlgError, NotImplementedError):
                    continue
            if len(expanded) <= 1: break
            shifts = projectAndGetShifts(backend.array.hstack(expanded))
            if backend.array.size(shifts) > 0: return shifts[:options.maxShifts]
        return self.computeHeuristicShifts(A, E)

    def filterStableShifts(self, shifts: Any, tol: float = 1e-14, selectPositiveImag: bool = False) -> Any:
        backend = self.localBackend
        realParts, absShifts = backend.array.real(shifts), backend.array.abs(shifts)
        mask = (realParts < 0) & (absShifts > tol) & (backend.array.abs(realParts) > tol)
        if selectPositiveImag: mask &= (backend.array.imag(shifts) >= 0)
        return shifts[mask]

    def computeHeuristicShifts(self, A: operatorBase, E: operatorBase | None) -> Any:
        backend, n = self.localBackend, A.shape[0]
        if n <= 256:
            try:
                aData = A.toBackendData()
                sh = self.filterStableShifts(backend.eigen.eigvalsGeneralized(aData, E.toBackendData() if E is not None else backend.array.eyeLike(aData)))
                if backend.array.size(sh) > 0: return sh[backend.array.argsort(backend.array.abs(sh))][:20]
            except (backend.linalg.LinAlgError, TypeError, ValueError, NotImplementedError):
                pass
        return backend.array.array([-1e-3, -1e-2, -1e-1, -1.0, -10.0], dtype=A.dtype)

    def updateShifts(self, A: operatorBase, E: operatorBase | None, vData: Any, zColumns: list[Any], prevShifts: Any, options: shiftComputationOptions, trans: bool = False) -> Any:
        backend, nc = self.localBackend, options.subspaceColumns
        if nc == 1: Q = backend.decomposition.qrOrthogonalize(backend.array.hstack([backend.array.real(vData), backend.array.imag(vData)]) if backend.array.iscomplexobj(vData) else vData, backend)
        else:
            numCols = min(nc * vData.shape[1], sum(column.shape[1] for column in zColumns))
            if numCols == 0: return prevShifts
            tail, remaining = [], numCols
            for column in reversed(zColumns):
                take = min(remaining, column.shape[1])
                if take:
                    tail.append(column[:, -take:])
                    remaining -= take
                if remaining == 0: break
            subspace = tail[0] if len(tail) == 1 else backend.array.hstack(tail[::-1])
            Q = backend.decomposition.qrOrthogonalize(subspace, backend)
        aProj = backend.linalg.dot(backend.linalg.transpose(Q), A.apply(Q, trans=trans))
        eProj = backend.linalg.dot(backend.linalg.transpose(Q), E.apply(Q, trans=trans)) if E is not None else backend.array.eyeLike(aProj)
        shifts = self.filterStableShifts(backend.eigen.eigvalsGeneralized(aProj, eProj), selectPositiveImag=True)
        if backend.array.size(shifts) == 0: return prevShifts
        re = backend.array.abs(backend.array.real(shifts))
        mask = (re > 1e-14) & (backend.array.abs(backend.array.imag(shifts)) < 1e-12 * re)
        shifts = backend.array.copy(shifts)
        shiftsImag = backend.array.imag(shifts)
        shiftsImag[mask] = 0
        return shifts[backend.array.argsort(backend.array.abs(shifts))]
