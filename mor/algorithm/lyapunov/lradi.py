from dataclasses import dataclass
from typing import Any, Optional

from mor.backends import backendRegistry
from mor.operators import matrixOperator
from mor.algorithm.registry import registerAlgorithm, algorithmRegistry
from .base import backendLyapunov

@dataclass(frozen=True, slots=True)
class shiftComputationOptions:
    initMaxIter, subspaceColumns = 20, 6

@registerAlgorithm('lyapunov', 'lradi')
class lradi(backendLyapunov):
    # TODO: Implement more robust shift selection (e.g., Penzl's heuristic) for stiff systems
    def __init__(self, backendName: str | None = None, **kwargs):
        self.localBackend, self.options = backendRegistry.get(backendName), kwargs

    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator, trans: bool = False) -> Any:
        backend, n, bData = self.localBackend, A.shape[0], B.data
        if backend.array.ndim(bData) == 1: bData = backend.array.reshape(bData, (-1, 1))    
        tol, maxIter, shiftOpts = self.options.get('tol', 1e-10), self.options.get('maxIter', 500), self.options.get('shiftOptions')
        if not isinstance(shiftOpts, shiftComputationOptions): shiftOpts = shiftComputationOptions(**(shiftOpts if isinstance(shiftOpts, dict) else {}))
        shifts, zColumns, W, j, jShift = self.computeInitialShifts(A, E, bData, shiftOpts), [], backend.array.copy(bData), 0, 0
        res = backend.specialized.gramMatrixNorm(W, backend)
        bTol = res * tol
        while res > bTol and j < maxIter:
            sigma = shifts[jShift]
            sigmaReal, sigmaImag = backend.array.real(sigma), backend.array.imag(sigma)
            if backend.array.abs(sigmaImag) < 1e-14 * backend.array.abs(sigmaReal): sigma, sigmaImag = sigmaReal, 0.0
            linearSolver = algorithmRegistry.get(category='linear', variant=self.options.get('linearVariant', 'auto'), backendName=backend.name, A=A, E=E, shift=sigma, trans=trans)
            vData = linearSolver.solve(A, W, E=E, shift=sigma, trans=trans)
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
            if jShift >= backend.array.size(shifts): shifts, jShift = self.updateShifts(A, E, vData, zColumns, shifts, shiftOpts, trans=trans), 0
            if len(zColumns) > 4:
                Z_temp = self.compressFactor(backend.array.hstack(zColumns), n)
                zColumns = [Z_temp]
        from mor.operators.lowRank import lowRankOperator
        return lowRankOperator(backend.array.hstack(zColumns) if zColumns else backend.array.zeros((n, 0), dtype=A.dtype), backendName=backend.name)

    def computeInitialShifts(self, A: matrixOperator, E: matrixOperator | None, bData: Any, options: shiftComputationOptions) -> Any:
        backend = self.localBackend
        def projectAndGetShifts(B: Any) -> Any:
            Q = backend.decomposition.qrOrthogonalize(B, backend)
            aProj, eProj = backend.linalg.dot(backend.linalg.transpose(Q), A.apply(Q)), backend.linalg.dot(backend.linalg.transpose(Q), E.apply(Q)) if E is not None else backend.array.eye(Q.shape[1], dtype=A.dtype)
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
                except Exception: continue
            if len(expanded) <= 1: break
            shifts = projectAndGetShifts(backend.array.hstack(expanded))
            if backend.array.size(shifts) > 0: return shifts
        return self.computeHeuristicShifts(A, E)

    def filterStableShifts(self, shifts: Any, tol: float = 1e-14, selectPositiveImag: bool = False) -> Any:
        backend = self.localBackend
        realParts, absShifts = backend.array.real(shifts), backend.array.abs(shifts)
        mask = (realParts < 0) & (absShifts > tol) & (backend.array.abs(realParts) > tol)
        if selectPositiveImag: mask &= (backend.array.imag(shifts) >= 0)
        return shifts[mask]

    def computeHeuristicShifts(self, A: matrixOperator, E: matrixOperator | None) -> Any:
        backend, n = self.localBackend, A.shape[0]
        if n <= 256:
            try:
                sh = self.filterStableShifts(backend.eigen.eigvalsGeneralized(A.data, E.data if E is not None else backend.array.eye(n, dtype=A.dtype)))
                if backend.array.size(sh) > 0: return sh[backend.array.argsort(backend.array.abs(sh))][:20]
            except Exception: pass
        return backend.array.array([-1e-3, -1e-2, -1e-1, -1.0, -10.0], dtype=A.dtype)

    def updateShifts(self, A: matrixOperator, E: matrixOperator | None, vData: Any, zColumns: list[Any], prevShifts: Any, options: shiftComputationOptions, trans: bool = False) -> Any:
        backend, nc = self.localBackend, options.subspaceColumns
        if nc == 1: Q = backend.decomposition.qrOrthogonalize(backend.array.hstack([backend.array.real(vData), backend.array.imag(vData)]) if backend.array.iscomplexobj(vData) else vData, backend)
        else:
            ZAll = backend.array.hstack(zColumns)
            numCols = min(nc * vData.shape[1], ZAll.shape[1])
            if numCols == 0: return prevShifts
            Q = backend.decomposition.qrOrthogonalize(ZAll[:, -numCols:], backend)
        aProj, eProj = backend.linalg.dot(backend.linalg.transpose(Q), A.apply(Q, trans=trans)), backend.linalg.dot(backend.linalg.transpose(Q), E.apply(Q, trans=trans)) if E is not None else backend.array.eye(Q.shape[1], dtype=A.dtype)
        shifts = self.filterStableShifts(backend.eigen.eigvalsGeneralized(aProj, eProj), selectPositiveImag=True)
        if backend.array.size(shifts) == 0: return prevShifts
        re = backend.array.abs(backend.array.real(shifts))
        mask = (re > 1e-14) & (backend.array.abs(backend.array.imag(shifts)) < 1e-12 * re)
        shifts = backend.array.copy(shifts)
        shiftsImag = backend.array.imag(shifts)
        shiftsImag[mask] = 0
        return shifts[backend.array.argsort(backend.array.abs(shifts))]
