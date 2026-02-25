from dataclasses import dataclass
from typing import Any, Optional

from mor.backends import backendRegistry
from mor.operators import matrixOperator
from mor.algorithm.registry import registerAlgorithm

@dataclass(frozen=True, slots=True)
class shiftComputationOptions:
    initMaxiter: int = 20
    subspaceColumns: int = 6

@registerAlgorithm('lyapunov', 'lradi')
class lradiAlgorithm:
    def __init__(self, backendName: str | None = None, **kwargs):
        self.localBackend = backendRegistry.get(backendName)
        self.options = kwargs

    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator) -> Any:
        backend = self.localBackend
        n = A.shape[0]
        bData = B.data
        if backend.array.ndim(bData) == 1:
            bData = backend.array.reshape(bData, (-1, 1))    
        trans = self.options.get('trans', False)
        tol = self.options.get('tol', 1e-10)
        maxIter = self.options.get('maxIter', 500)
        shiftOpts = self.options.get('shiftOptions')
        if not isinstance(shiftOpts, shiftComputationOptions):
            shiftOpts = shiftComputationOptions(**(shiftOpts if isinstance(shiftOpts, dict) else {}))
        shifts = self._computeInitialShifts(A, E, bData, shiftOpts)
        zColumns = []
        w = backend.array.copy(bData)
        j = 0
        jShift = 0
        res = backend.specialized.gramMatrixNorm(w, backend)
        bTol = res * tol
        while res > bTol and j < maxIter:
            sigma = shifts[jShift]
            sigma_real = backend.array.real(sigma)
            sigma_imag = backend.array.imag(sigma)
            if backend.array.abs(sigma_imag) < 1e-14 * backend.array.abs(sigma_real):
                sigma = sigma_real
                sigma_imag = 0.0
            # Linear solver for (A + sigma*E)v = w
            linear_solver = algorithmRegistry.get(
                category='linear',
                variant=self.options.get('linearVariant', 'auto'),
                backendName=backend.name,
                A=A, E=E, shift=sigma, trans=trans
            )
            vData = linear_solver.solve(A, w, E=E, shift=sigma, trans=trans)
            
            if sigma_imag == 0:
                s = sigma_real
                if E is not None:
                    ev = E.apply(vData, trans=trans)
                    w = w - (2 * s) * ev
                else:
                    w = w - (2 * s) * vData
                zColumns.append(vData * backend.array.sqrt(-2 * s))
                j += 1
            else:
                gs = -4 * sigma_real
                d = sigma_real / sigma_imag if backend.array.abs(sigma_imag) > 1e-14 else 0.0
                if trans:
                    vData = backend.array.conj(vData)
                
                u = backend.array.real(vData) + backend.array.imag(vData) * d
                if E is not None:
                    eu = E.apply(u, trans=trans)
                    w = w + gs * eu
                else:
                    w = w + gs * u
                g = backend.array.sqrt(gs)
                zColumns.append(u * g)
                zColumns.append(backend.array.imag(vData) * (g * backend.array.sqrt(d**2 + 1)))
                j += 2
            jShift += 1
            res = backend.specialized.gramMatrixNorm(w, backend)
            if jShift >= backend.array.size(shifts):
                shifts = self._updateShifts(A, E, vData, zColumns, shifts, shiftOpts)
                jShift = 0
        if len(zColumns) == 0:
            return backend.array.zeros((n, 0), dtype=A.dtype)
        return backend.array.hstack(zColumns)

    def _computeInitialShifts(self, A: matrixOperator, E: matrixOperator | None, bData: Any, options: shiftComputationOptions) -> Any:
        backend = self.localBackend
        def projectAndGetShifts(b: Any) -> Any:
            Q = backend.decomposition.qrOrthogonalize(b, backend)
            aProj = backend.linalg.dot(Q.T, A.apply(Q))
            eProj = backend.linalg.dot(Q.T, E.apply(Q)) if E is not None else backend.array.eye(Q.shape[1], dtype=A.dtype)
            sh = backend.eigen.eigvalsGeneralized(aProj, eProj)
            return self._filterStableShifts(sh)
        shifts = projectAndGetShifts(bData)
        if backend.array.size(shifts) > 0:
            return shifts
        for _ in range(options.initMaxiter - 1):
            expanded = [bData]
            for alpha in (-1e-2, -1e-1, -1.0, -10.0):
                try:
                    linear_solver = algorithmRegistry.get(category='linear',variant='shifted',backendName=backend.name,A=A, E=E, shift=alpha, trans=False)
                    vData = linear_solver.solve(A, bData, E=E, shift=alpha, trans=False)
                    if backend.array.all(backend.array.isfinite(vData)):
                        expanded.append(vData)
                except Exception:
                    continue
            if len(expanded) <= 1:
                break
            bExp = backend.array.hstack(expanded)
            shifts = projectAndGetShifts(bExp)
            if backend.array.size(shifts) > 0:
                return shifts
        return self._computeHeuristicShifts(A, E)

    def _filterStableShifts(self, shifts: Any, tol: float = 1e-14, selectPositiveImag: bool = False) -> Any:
        backend = self.localBackend
        real_parts = backend.array.real(shifts)
        abs_shifts = backend.array.abs(shifts)
        mask = (real_parts < 0) & (abs_shifts > tol) & (backend.array.abs(real_parts) > tol)
        if selectPositiveImag:
            mask &= (backend.array.imag(shifts) >= 0)
        return shifts[mask]

    def _computeHeuristicShifts(self, A: matrixOperator, E: matrixOperator | None) -> Any:
        backend = self.localBackend
        n = A.shape[0]
        if n <= 256:
            try:
                aData = A.data
                eData = E.data if E is not None else backend.array.eye(n, dtype=A.dtype)
                sh = self._filterStableShifts(backend.eigen.eigvalsGeneralized(aData, eData))
                if backend.array.size(sh) > 0:
                    # Use backend sorting if available, or stay on backend
                    abs_sh = backend.array.abs(sh)
                    idx = backend.array.argsort(abs_sh)
                    return sh[idx][:20]
            except Exception:
                pass
        return backend.array.array([-1e-3, -1e-2, -1e-1, -1.0, -10.0], dtype=A.dtype)

    def _updateShifts(self, A: matrixOperator, E: matrixOperator | None, vData: Any, zColumns: list[Any], prevShifts: Any, options: shiftComputationOptions) -> Any:
        backend = self.localBackend
        nc = options.subspaceColumns
        if nc == 1:
            if backend.array.iscomplexobj(vData):
                q = backend.decomposition.qrOrthogonalize(backend.array.hstack([backend.array.real(vData), backend.array.imag(vData)]), backend)
            else:
                q = backend.decomposition.qrOrthogonalize(vData, backend)
        else:
            zAll = backend.array.hstack(zColumns)
            numCols = min(nc * vData.shape[1], zAll.shape[1])
            if numCols == 0:
                return prevShifts
            z = zAll[:, -numCols:]
            q = backend.decomposition.qrOrthogonalize(z, backend)
        aProj = backend.linalg.dot(q.T, A.apply(q))
        eProj = backend.linalg.dot(q.T, E.apply(q)) if E is not None else backend.array.eye(q.shape[1], dtype=A.dtype)
        shifts = self._filterStableShifts(backend.eigen.eigvalsGeneralized(aProj, eProj), selectPositiveImag=True)
        if backend.array.size(shifts) == 0:
            return prevShifts
        re = backend.array.abs(backend.array.real(shifts))
        mask = (re > 1e-14) & (backend.array.abs(backend.array.imag(shifts)) < 1e-12 * re)
        shifts = backend.array.copy(shifts)
        # Note: some backends might need specific handling for item assignment
        shifts_imag = backend.array.imag(shifts)
        shifts_imag[mask] = 0
        # Reconstruct or update shifts if necessary depending on backend API
        abs_sh = backend.array.abs(shifts)
        return shifts[backend.array.argsort(abs_sh)]
