import numpy as np
from dataclasses import dataclass
from typing import Any

from mor.backends import backendRegistry
from mor.operators import matrixOperator

_identity = lambda x: x

@dataclass(frozen=True, slots=True)
class shiftComputationOptions:
    initMaxiter: int = 20
    subspaceColumns: int = 6

def solveLyapunovLr( A: matrixOperator, B: matrixOperator,*,trans: bool = False,backendName: str = 'numpy',tol: float = 1e-10, maxIter: int = 500,shiftOptions: shiftComputationOptions | None = None,) -> np.ndarray:
    opts = shiftOptions or shiftComputationOptions()
    return _solveLyapunovLrCore(A, B, E=None, trans=trans, backendName=backendName,tol=tol, maxIter=maxIter, shiftOptions=opts)

def solveLyapunovLrGeneralized(A: matrixOperator, E: matrixOperator, B: matrixOperator, *, trans: bool = False, backendName: str = 'numpy', tol: float = 1e-10, maxIter: int = 500, shiftOptions: shiftComputationOptions | None = None, ) -> np.ndarray:
    opts = shiftOptions or shiftComputationOptions()
    return _solveLyapunovLrCore(A, B, E=E, trans=trans, backendName=backendName,tol=tol, maxIter=maxIter, shiftOptions=opts)

def _solveLyapunovLrCore( A: matrixOperator, B: matrixOperator, E: matrixOperator | None, *, trans: bool, backendName: str, tol: float, maxIter: int, shiftOptions: shiftComputationOptions,) -> np.ndarray:
    backend = backendRegistry.get(backendName)
    n = A.shape[0]
    bData = B.toNumpy()
    if bData.ndim == 1:
        bData = bData[:, np.newaxis]
    if E is not None:
        eTransposed = E.T
        applyE = E.apply
        applyET = eTransposed.apply
    else:
        applyE = applyET = _identity
    shifts = _computeInitialShifts(A, E, bData, backend, shiftOptions)
    zColumns: list[np.ndarray] = []
    w = bData.copy()
    j = 0
    jShift = 0
    res = backend.specialized.gramMatrixNorm(w, backend)
    bTol = res * tol
    while res > bTol and j < maxIter: 
        sigma = shifts[jShift]
        if np.abs(sigma.imag) < 1e-14 * np.abs(sigma.real):
            sigma = sigma.real
        v = A.solveShifted(E, sigma, w, trans=trans)
        if sigma.imag == 0:
            s = sigma.real
            w = w - (2 * s) * (applyET(v) if trans else applyE(v))
            zColumns.append(v * np.sqrt(-2 * s))
            j += 1
        else:
            gs = -4 * sigma.real
            d = sigma.real / sigma.imag if np.abs(sigma.imag) > 1e-14 else 0.0
            if trans:
                v = v.conj()
            u = v.real + v.imag * d
            w = w + gs * (applyET(u) if trans else applyE(u))
            g = np.sqrt(gs)
            zColumns.append(u * g)
            zColumns.append(v.imag * (g * np.sqrt(d**2 + 1)))
            j += 2
        jShift += 1
        res = backend.specialized.gramMatrixNorm(w, backend)
        if jShift >= shifts.size:
            shifts = _updateShifts(A, E, v, zColumns, shifts, shiftOptions, backend)
            jShift = 0
    if len(zColumns) == 0:
        return backend.array.zeros((n, 0), dtype=A.dtype)
    return backend.array.hstack(zColumns)

def _computeGeneralizedEigenvalues(backend: Any, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return backend.eigen.eigvalsGeneralized(A, B)

def _filterStableShifts( shifts: np.ndarray, *, tol: float = 1e-14, selectPositiveImag: bool = False, ) -> np.ndarray:
    mask = (shifts.real < 0) & (np.abs(shifts) > tol) & (np.abs(shifts.real) > tol)
    if selectPositiveImag:
        mask &= (shifts.imag >= 0)
    return shifts[mask]

def _computeInitialShifts( A: matrixOperator, E: matrixOperator | None, bData: np.ndarray, backend: Any, options: shiftComputationOptions,) -> np.ndarray:
    def projectAndGetShifts(b: np.ndarray) -> np.ndarray:
        Q = backend.decomposition.qrOrthogonalize(b, backend) # TODO： sofar is qr, and maybe lsqr and gram_schmidt can be used use operators control
        aProj = backend.linalg.dot(Q.T, A.apply(Q))
        eProj = backend.linalg.dot(Q.T, E.apply(Q)) if E is not None else backend.array.eye(Q.shape[1], dtype=A.dtype)
        sh = _computeGeneralizedEigenvalues(backend, aProj, eProj)
        return _filterStableShifts(sh)
    shifts = projectAndGetShifts(bData)
    if shifts.size > 0:
        return shifts
    for _ in range(options.initMaxiter - 1):
        expanded = [bData]
        for alpha in (-1e-2, -1e-1, -1.0, -10.0):
            try:
                v = A.solveShifted(E, alpha, bData, trans=False)
                if np.all(np.isfinite(v)):
                    expanded.append(v)
            except (np.linalg.LinAlgError, RuntimeError):
                continue
        if len(expanded) <= 1:
            break
        bExp = backend.array.hstack(expanded)
        shifts = projectAndGetShifts(bExp)
        if shifts.size > 0:
            return shifts      #TODO：check for better performance here
    return _computeHeuristicShifts(A, E, backend)

def _computeHeuristicShifts(A: matrixOperator, E: matrixOperator | None, backend: Any) -> np.ndarray:
    n = A.shape[0]
    aData = A.toNumpy()
    eData = E.toNumpy() if E is not None else backend.array.eye(n, dtype=A.dtype)
    if n <= 256: #TODO: is this very necessary to set the limitation to 256? options: 1. remove the limitation 2. make it a parameter 3. reset it with performanceTest
        try:
            sh = _filterStableShifts(_computeGeneralizedEigenvalues(backend, aData, eData))
            if sh.size > 0:
                return sh[np.argsort(np.abs(sh))][:20]
        except Exception:
            pass
    try:
        diagA = np.diag(aData)
        diagE = np.diag(eData)
        mask = np.abs(diagE) > 1e-14
        if np.any(mask):
            ratios = diagA[mask] / diagE[mask]
            ratios = ratios[ratios < 0]
            if ratios.size >= 2:
                lambdaMin = np.min(ratios)
                lambdaMax = np.max(ratios)
                if lambdaMin < lambdaMax:
                    return np.geomspace(-lambdaMax, -lambdaMin, 10)
    except Exception:
        pass
    return np.array([-1e-3, -1e-2, -1e-1, -1.0, -10.0], dtype=np.float64) #TODO： here use 3 strategies to set the shiftsties, necessary?priority?bable?


def _updateShifts( A: matrixOperator, E: matrixOperator | None, v: np.ndarray, zColumns: list[np.ndarray], prevShifts: np.ndarray, options: shiftComputationOptions, backend: Any, ) -> np.ndarray:
    nc = options.subspaceColumns
    if nc == 1:
        if np.iscomplexobj(v):
            q = backend.decomposition.qrOrthogonalize(backend.array.hstack([v.real, v.imag]), backend)
        else:
            q = backend.decomposition.qrOrthogonalize(v, backend)
    else:
        zAll = backend.array.hstack(zColumns)
        numCols = min(nc * v.shape[1], zAll.shape[1])
        if numCols == 0:
            return prevShifts
        z = zAll[:, -numCols:]
        q = backend.decomposition.qrOrthogonalize(z, backend)
    aProj = backend.linalg.dot(q.T, A.apply(q))
    eProj = backend.linalg.dot(q.T, E.apply(q)) if E is not None else backend.array.eye(q.shape[1], dtype=A.dtype)
    shifts = _filterStableShifts(_computeGeneralizedEigenvalues(backend, aProj, eProj), selectPositiveImag=True)
    if shifts.size == 0:
        return prevShifts
    re = np.abs(shifts.real)
    mask = (re > 1e-14) & (np.abs(shifts.imag) < 1e-12 * re)
    shifts = shifts.copy()
    shifts.imag[mask] = 0
    return shifts[np.argsort(np.abs(shifts))]
