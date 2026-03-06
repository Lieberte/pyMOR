import math
from typing import Any
from mor.operators import matrixOperator
from mor.operators.lowRank import lowRankOperator
from mor.algorithm.registry import registerAlgorithm
from .base import backendLyapunov

@registerAlgorithm('lyapunov', 'sign')
class sign(backendLyapunov):
    # TODO: Improve numerical stability for stiff systems (e.g., hplant) by refining log-det scaling
    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator, trans: bool = False) -> Any:
        backend, n = self.localBackend, A.shape[0]
        maxIter, tol = self.options.get('maxIter', 80), self.options.get('tol', 1e-14)
        aData = backend.linalg.transpose(A.data) if trans else A.data
        if E is not None:
            Q = self.prepareQ(B, trans=trans)
            eData = backend.linalg.transpose(E.data) if trans else E.data
            X = backend.linalg.solveGeneralizedContinuousLyapunov(aData, eData, Q)
            return lowRankOperator(backend.linalg.robustSqrtFactor(X, name='SignFallback'), backendName=backend.name)
        Aj = backend.array.copy(aData)
        Zj = backend.array.copy(B.data)
        if backend.array.ndim(Zj) == 1: Zj = backend.array.reshape(Zj, (-1, 1))
        if hasattr(Zj, 'to') and hasattr(Aj, 'dtype'): Zj = Zj.to(Aj.dtype)
        eye = backend.array.eye(n, dtype=A.dtype)
        converged = False
        for _ in range(maxIter):
            _, logDet = backend.linalg.slogdet(Aj)
            logGamma = logDet / n
            logGammaAbs = float(backend.array.abs(logGamma))
            if backend.array.isfinite(logGamma) and logGammaAbs < 30:
                gamma = backend.array.exp(logGamma)
            else:
                gamma = 1.0
            try: AjInv = backend.linalg.solve(Aj, eye)
            except Exception: break
            c1 = backend.array.sqrt(gamma / 2.0)
            c2 = backend.array.sqrt(1.0 / (2.0 * gamma))
            Zj = backend.array.hstack([c1 * Zj, c2 * backend.linalg.solve(Aj, Zj)])
            Aj = (gamma * Aj + AjInv / gamma) / 2.0
            diff = float(backend.linalg.norm(Aj + eye)) / n
            if not math.isfinite(diff): break
            if diff < tol:
                converged = True
                break
            if Zj.shape[1] > 4 * n: Zj = self.compressFactor(Zj, n)
        ZFactor = self.compressFactor(Zj / math.sqrt(2.0), n)
        zNorm = float(backend.linalg.norm(ZFactor))
        XApprox = backend.linalg.dot(ZFactor, backend.linalg.transpose(ZFactor))
        QRef = self.prepareQ(B, trans=trans)
        residual = backend.linalg.dot(aData, XApprox) + backend.linalg.dot(XApprox, backend.linalg.transpose(aData)) + QRef
        resNorm = float(backend.linalg.norm(residual))
        qNorm = float(backend.linalg.norm(QRef))
        relRes = resNorm / max(qNorm, 1e-16)
        if converged and math.isfinite(zNorm) and math.isfinite(relRes) and relRes < self.options.get('signResidualTol', 1e-6):
            return lowRankOperator(ZFactor, backendName=backend.name)
        Q = self.prepareQ(B, trans=trans)
        X = backend.linalg.solveContinuousLyapunov(aData, Q)
        return lowRankOperator(backend.linalg.robustSqrtFactor(X, name='SignFallback'), backendName=backend.name)
