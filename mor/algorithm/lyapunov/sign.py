import math
from typing import Any
from mor.operators import matrixOperator
from mor.operators.lowRank import lowRankOperator
from mor.algorithm.registry import registerAlgorithm
from .base import backendLyapunov

@registerAlgorithm('lyapunov', 'sign')
class sign(backendLyapunov):
    def solve(self, A: matrixOperator, E: matrixOperator | None, B: matrixOperator, trans: bool = False) -> Any:
        self.validateIterationOptions()
        self.lastIterations, self.lastFallback, self.lastRelativeResidual = 0, None, None
        backend, n = self.localBackend, A.shape[0]
        maxIter, tol = self.options.get('maxIter', 80), self.options.get('tol', 1e-14)
        aData = A.toBackendData()
        if trans:
            aData = backend.linalg.transpose(aData)
        if E is not None:
            self.lastFallback = 'generalized_direct'
            Q = self.prepareQ(B, trans=trans)
            eData = E.toBackendData()
            if trans:
                eData = backend.linalg.transpose(eData)
            X = backend.linalg.solveGeneralizedContinuousLyapunov(aData, eData, Q)
            return lowRankOperator(backend.linalg.robustSqrtFactor(X, name='SignFallback'), backendName=backend.name)
        Aj = backend.array.copy(aData)
        Zj = backend.array.copy(B.toBackendData())
        if trans:
            Zj = backend.linalg.transpose(Zj)
        if backend.array.ndim(Zj) == 1: Zj = backend.array.reshape(Zj, (-1, 1))
        if hasattr(Zj, 'to') and hasattr(Aj, 'dtype'): Zj = Zj.to(Aj.dtype)
        eye = backend.array.eyeLike(Aj)
        converged = False
        for iteration in range(maxIter):
            self.lastIterations = iteration + 1
            _, logDet = backend.linalg.slogdet(Aj)
            # gamma scales Aj toward unit determinant, so use the reciprocal.
            logGamma = -float(logDet) / n
            if math.isfinite(logGamma):
                gamma = math.exp(max(-300.0, min(300.0, logGamma)))
            else:
                gamma = 1.0
            try: AjInv = backend.linalg.solve(Aj, eye)
            except backend.linalg.LinAlgError: break
            c1 = math.sqrt(gamma / 2.0)
            c2 = math.sqrt(1.0 / (2.0 * gamma))
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
        self.lastRelativeResidual = relRes
        if converged and math.isfinite(zNorm) and math.isfinite(relRes) and relRes < self.options.get('signResidualTol', 1e-6):
            return lowRankOperator(ZFactor, backendName=backend.name)
        Q = self.prepareQ(B, trans=trans)
        self.lastFallback = 'continuous_direct'
        X = backend.linalg.solveContinuousLyapunov(aData, Q)
        return lowRankOperator(backend.linalg.robustSqrtFactor(X, name='SignFallback'), backendName=backend.name)
