import numpy as np
import scipy.linalg as la
from typing import Tuple

from .balancedTruncation import balancedTruncation
from mor.utils.lyapunov import lyapunovSolver
from temp.utils.svd import svdMethod, svdFactory


class srBalancedTruncation(balancedTruncation):

    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray,D: np.ndarray | None = None, ts: float = -1.0,
                 svdMethodType: svdMethod = svdMethod.static):
        super().__init__()

        self._initializeSystem(A, B, C, D, ts)
        self.Wc: np.ndarray | None = None
        self.Wo: np.ndarray | None = None
        self.Lc: np.ndarray | None = None
        self.Lo: np.ndarray | None = None

        self._svdMethod = svdMethodType
        self._svdSolver = svdFactory.create(svdMethodType)
        self._isSVDComputed: bool  = False

    def reduce(self, r: int) -> bool:
        if not self._validateReductionOrder(r):
            return False
        success = self._performSrBt(r)
        if success:
            self.isReduced = True
            self._logInfo(f"SR-BT reduction successful: {self.n} → {self.r}")
        else:
            self._logError("SR-BT reduction failed")
        return success

    def reduceByEnergy(self, energyThreshold: float = 0.99) -> bool:
        if not self._validateEnergyThreshold(energyThreshold):
            return False
        if not self._computeHankelSingularValues():
            return False
        r = self._selectOrderByEnergy(energyThreshold)
        if r == 0:
            self._logError("Failed to select reduction order")
            return False
        self._logInfo(f"Selected order r={r} for energy threshold {energyThreshold:.2%}")
        return self._performTruncation(r)

    def getReducedA(self) -> np.ndarray | None:
        return self.Ar

    def getReducedB(self) -> np.ndarray | None:
        return self.Br

    def getReducedC(self) -> np.ndarray | None:
        return self.Cr

    def getReducedD(self) -> np.ndarray | None:
        return self.Dr

    def getHankelSingularValues(self) -> np.ndarray | None:
        return self.hankelSv

    def computeErrorBound(self, r: int) -> float:
        if self.hankelSv is None or r >= len(self.hankelSv):
            return np.inf
        return 2.0 * np.sum(self.hankelSv[r:])

    def getSVDInfo(self) -> dict:
        if not self._isSVDComputed:
            return {
                'method': self._svdMethod.name,
                'status': 'not computed',
                'isFitted': False
            }
        return self._svdSolver.getInfo()

    def getGramianInfo(self) -> dict:
        info = {}
        if self.Wc is not None:
            info['controllability'] = {
                'trace': np.trace(self.Wc),
                'condition': np.linalg.cond(self.Wc),
                'min_eigenvalue': np.min(np.linalg.eigvalsh(self.Wc))
            }
        if self.Wo is not None:
            info['observability'] = {
                'trace': np.trace(self.Wo),
                'condition': np.linalg.cond(self.Wo),
                'min_eigenvalue': np.min(np.linalg.eigvalsh(self.Wo))
            }
        return info

    def getEnergyRetention(self,r:int) -> float:
        if self.hankelSv is None or r >= len(self.hankelSv):
            return 0.0
        return np.sum(self.hankelSv[:r])/np.sum(self.hankelSv) if np.sum(self.hankelSv) > 0 else 0.0

    def _solveLyapunovContinuous(self, A: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, bool]:
        X, converged = lyapunovSolver.solveContinuous(
            A, Q,
            maxIter=self.tol['maxLyapunovIterations'],
            tol=self.tol['lyapunovConvergence']
        )
        return X, converged

    def _solveLyapunovDiscrete(self, A: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, bool]:
        X, converged = lyapunovSolver.solveDiscrete(
            A, Q,
            maxIter=self.tol['maxLyapunovIterations'],
            tol=self.tol['lyapunovConvergence']
        )
        return X, converged

    def _performSrBt(self, r: int) -> bool:
        if not self._computeControllabilityGramian():
            return False
        if not self._computeObservabilityGramian():
            return False
        if not self._validateGramians():
            return False
        if not self._computeCholeskyFactorization(self.Wc, 'Lc'):
            return False
        if not self._computeCholeskyFactorization(self.Wo, 'Lo'):
            return False
        if not self._computeHankelSingularValues():
            return False
        if not self._performTruncation(r):
            return False
        return True

    def _computeControllabilityGramian(self) -> bool:
        Q = self._computeSymmetricProduct(self.B, self.B)
        Wc, converged = self._solveLyapunov(self.A, Q)

        if not converged:
            self._logWarning("Controllability Gramian computation did not fully converge")

        self.Wc = Wc
        return True

    def _computeObservabilityGramian(self) -> bool:
        Q = self._computeSymmetricProduct(self.C.T, self.C.T, transpose2=False)
        Wo, converged = self._solveLyapunov(self.A.T, Q)

        if not converged:
            self._logWarning("Observability Gramian computation did not fully converge")

        self.Wo = Wo
        return True

    def _computeCholeskyFactorization(self, M: np.ndarray, attrName: str) -> bool:
        try:
            L = la.cholesky(M, lower=True)
            setattr(self, attrName, L)
            return True
        except la.LinAlgError:
            self._logError(f"Cholesky factorization failed for {attrName}")
            try:
                eps = 1e-10 * np.trace(M) / M.shape[0]
                mReg = M + eps * np.eye(M.shape[0])
                L = la.cholesky(mReg, lower=True)
                setattr(self, attrName, L)
                self._logWarning(f"Used regularized Cholesky for {attrName}")
                return True
            except Exception as e:
                self._logError(f"Regularized Cholesky failed for {attrName}: {e}")
                return False

    def _computeHankelSingularValues(self) -> bool:
        if self.Lc is None or self.Lo is None:
            self._logError("Cholesky factors not computed")
            return False

        try:
            M = self.Lo.T @ self.Lc
            self._svdSolver.fit(M)
            self._isSVDComputed = True
            self.hankelSv = self._svdSolver.S.copy()
            threshold = self.tol.get('singularValueThreshold',1e-10)
            validIdx = self.hankelSv > threshold
            nFiltered = np.sum(~validIdx)
            if nFiltered > 0:
                self._logWarning(f"Filtered {nFiltered} singular values below threshold {threshold:.2e}")
            self.hankelSv = self.hankelSv[validIdx]
            if len(self.hankelSv) == 0:
                self._logError("No valid singular value found")
                return False
            return  True
        except Exception as e:
            self._logError(f"Hankel SVD failed: {e}")
            return False

    def _performTruncation(self, r: int) -> bool:
        if r > len(self.hankelSv):
            r = len(self.hankelSv)
            self._logWarning(f"Reducing order to r={r} (max available)")
        self.r = r
        try:
            sigmaRInvSqrt = np.diag(1.0 / np.sqrt(self.hankelSv[:r]))
            U = self._svdSolver.U[:,:r]
            Vt = self._svdSolver.Vt[:r,:]
            V = Vt.T
            T = self.Lc @ V @ sigmaRInvSqrt
            TInv = sigmaRInvSqrt @ U.T @ self.Lo.T
            self.Ar = TInv @ self.A @ T
            self.Br = TInv @ self.B
            self.Cr = self.C @ T
            self.Dr = self.D.copy()
            if not self._validateReducedSystem():
                return False
            return True
        except Exception as e:
            self._logError(f"Truncation failed: {e}")
            return False

    def _validateReductionOrder(self, r:int)->bool:
        if r<=0:
            self._logError(f"Reduction order must be positive, get {r}")
            return False
        if r>self.n:
            self._logError(f"Reduction order must be less than n={self.n}, get {r}")
            return False
        return True

    def _validateEnergyThreshold(self, threshold: float) -> bool:
        if not (0 < threshold <= 1):
            self._logError(f"Energy threshold must be in (0,1], get {threshold}")
            return False
        return True

    def _validateGramians(self) -> bool:
        if self.Wc is None or self.Wo is None:
            self._logError("Gramians not computed")
            return False
        if not np.allclose(self.Wc, self.Wc.T, rtol=1e-6):
            self._logWarning("Controllability Gramian is not symmetric")
        if not np.allclose(self.Wo, self.Wo.T, rtol=1e-6):
            self._logWarning("Observability Gramian is not symmetric")
        eigvalsWc = np.linalg.eigvalsh(self.Wc)
        eigvalsWo = np.linalg.eigvalsh(self.Wo)
        threshold = -1e-10
        if np.min(eigvalsWc) < threshold:
            self._logError("Controllability Gramian is not positive semi-definite")
            return False
        if np.min(eigvalsWo) < threshold:
            self._logError("Observability Gramian is not positive semi-definite")
            return False

        return True

    def _validateReducedSystem(self) -> bool:
        if self.Ar.shape != (self.r, self.r):
            self._logError(f"Ar dimension mismatch: expected ({self.r}, {self.r}), got {self.Ar.shape}")
            return False
        if self.Br.shape != (self.r, self.m):
            self._logError(f"Br dimension mismatch: expected ({self.r}, {self.m}), got {self.Br.shape}")
            return False
        if self.Cr.shape != (self.p, self.r):
            self._logError(f"Cr dimension mismatch: expected ({self.p}, {self.r}), got {self.Cr.shape}")
            return False
        if self.Dr.shape != (self.p, self.m):
            self._logError(f"Dr dimension mismatch: expected ({self.p}, {self.m}), got {self.Dr.shape}")
            return False
        return True

    def _selectOrderByEnergy(self, threshold: float) -> int:
        if self.hankelSv is None:
            return 0
        totalEnergy = np.sum(self.hankelSv)
        cumulativeEnergy = np.cumsum(self.hankelSv) / totalEnergy
        r = np.searchsorted(cumulativeEnergy, threshold) + 1
        if r > len(self.hankelSv):
            r = len(self.hankelSv)
        return r
