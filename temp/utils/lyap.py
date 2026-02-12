import numpy as np
import scipy.linalg as la
from enum import Enum
from typing import Tuple, Any

class LyapunovMethod(Enum):
    SCIPY = 'scipy'
    ADI = 'adi'
    RICHARDSON = 'richardson'
    DOUBLING = 'doubling'
    LRADI = 'lradi'

class EquationType(Enum):
    CONT = 'continuous'
    DISC = 'discrete'

class lyapSolver:
    """
    Lyapunov solver framework following the style of lyapunov.py,
    integrated with generalized and low-rank features.
    """
    @staticmethod
    def solve(A: np.ndarray, B: np.ndarray, 
              E: np.ndarray | None = None,
              equationType: EquationType = EquationType.CONT,
              trans: bool = False,
              isLRCF: bool = False,
              method: LyapunovMethod | None = None,
              **kwargs) -> Any:
        """
        Unified interface to solve Lyapunov equations.
        
        A X + X A^T + B B^T = 0 (Cont, E=None)
        A X E^T + E X A^T + B B^T = 0 (Cont, E exists)
        A X A^T - X + B B^T = 0 (Disc, E=None)
        A X A^T - E X E^T + B B^T = 0 (Disc, E exists)
        """
        if equationType == EquationType.CONT:
            return lyapSolver.solveContinuous(A, B, E, trans, isLRCF, method, **kwargs)
        else:
            return lyapSolver.solveDiscrete(A, B, E, trans, isLRCF, method, **kwargs)

    @staticmethod
    def solveContinuous(A: np.ndarray, B: np.ndarray, 
                         E: np.ndarray | None = None,
                         trans: bool = False,
                         isLRCF: bool = False,
                         method: LyapunovMethod | None = None,
                         **kwargs) -> Any:
        # Step 1: Pre-process matrices for E and trans
        Ah, Bh = A, B
        if trans:
            Ah = A.T
            Bh = B.T if B.ndim > 1 else B.reshape(-1, 1).T
        
        if E is not None:
            Eh = E.T if trans else E
            EhInv = la.inv(Eh)
            Ah = EhInv @ Ah
            Bh = EhInv @ Bh
        
        # Ensure B is in the right shape for BB^T
        if Bh.ndim == 1:
            Bh = Bh.reshape(-1, 1)
        
        # Step 2: Choose backend and solve
        if method is None:
            method = LyapunovMethod.SCIPY
        
        X = None
        if method == LyapunovMethod.SCIPY:
            X = la.solve_continuous_lyapunov(Ah, -(Bh @ Bh.T))
        elif method == LyapunovMethod.ADI:
            X, _ = lyapSolver._solveContinuousAdi(Ah, Bh @ Bh.T, **kwargs)
        elif method == LyapunovMethod.RICHARDSON:
            X, _ = lyapSolver._solveContinuousRichardson(Ah, Bh @ Bh.T, **kwargs)
        else:
            raise ValueError(f"Continuous method {method} not supported yet")
        
        # Step 3: Return requested type
        return lyapSolver._getSquareRoot(X) if isLRCF else X

    @staticmethod
    def _solveContinuousRichardson(A, Q, maxIter=1000, tol=1e-8):
        n = A.shape[0]
        X = np.zeros((n, n))
        eigvals = la.eigvals(A + A.T)
        omega = 2.0 / (np.min(np.real(eigvals)) + np.max(np.real(eigvals)) + 1e-10)
        for k in range(maxIter):
            R = A @ X + X @ A.T + Q
            X = X - omega * R
            if k % 20 == 0:
                X = 0.5 * (X + X.T)
                if la.norm(R, 'fro') < tol:
                    return X, True
        return 0.5 * (X + X.T), False

    @staticmethod
    def _solveContinuousAdi(A, Q, maxIter=100, tol=1e-8, numShifts=5):
        n = A.shape[0]
        X = np.zeros((n, n))
        eigvals = la.eigvals(A)
        eigReal = np.real(eigvals)
        pMin, pMax = np.min(np.abs(eigReal)), np.max(np.abs(eigReal))
        shifts = -np.logspace(np.log10(pMin + 1e-6), np.log10(pMax + 1e-6), numShifts)
        
        luFactors, amIList = [], []
        for p in shifts:
            try:
                lu, piv = la.lu_factor(A + p * np.eye(n))
                luFactors.append((lu, piv))
                amIList.append(A - p * np.eye(n))
            except: return X, False

        for k in range(maxIter):
            idx = k % numShifts
            lu, piv, amI = luFactors[idx][0], luFactors[idx][1], amIList[idx]
            try:
                V = la.lu_solve((lu, piv), -Q - X @ amI.T)
                X = la.lu_solve((lu, piv), -Q - V.T @ amI.T).T
                X = 0.5 * (X + X.T)
            except: return X, False
            
            if k % numShifts == numShifts - 1:
                if la.norm(A @ X + X @ A.T + Q, 'fro') < tol:
                    return X, True
        return X, False

    @staticmethod
    def solveDiscrete(A: np.ndarray, B: np.ndarray, 
                       E: np.ndarray | None = None,
                       trans: bool = False,
                       isLRCF: bool = False,
                       method: LyapunovMethod | None = None,
                       **kwargs) -> Any:
        # Step 1: Pre-process matrices for E and trans
        Ah, Bh = A, B
        if trans:
            Ah = A.T
            Bh = B.T if B.ndim > 1 else B.reshape(-1, 1).T
        
        if E is not None:
            Eh = E.T if trans else E
            EhInv = la.inv(Eh)
            Ah = EhInv @ Ah
            Bh = EhInv @ Bh
        
        # Ensure B is in the right shape for BB^T
        if Bh.ndim == 1:
            Bh = Bh.reshape(-1, 1)
        
        # Step 2: Choose backend and solve
        if method is None:
            method = LyapunovMethod.DOUBLING
        
        X = None
        if method == LyapunovMethod.SCIPY:
            X = la.solve_discrete_lyapunov(Ah, Bh @ Bh.T)
        elif method == LyapunovMethod.DOUBLING:
            X, _ = lyapSolver._solveDiscreteDoubling(Ah, Bh @ Bh.T, **kwargs)
        elif method == LyapunovMethod.RICHARDSON:
            X, _ = lyapSolver._solveDiscreteRichardson(Ah, Bh @ Bh.T, **kwargs)
        else:
            raise ValueError(f"Discrete method {method} not supported yet")
        
        # Step 3: Return requested type
        return lyapSolver._getSquareRoot(X) if isLRCF else X

    @staticmethod
    def _solveDiscreteDoubling(A, Q, maxIter=50, tol=1e-8):
        ak, qk, X = A.copy(), Q.copy(), Q.copy()
        for k in range(maxIter):
            qkNew = qk + ak @ qk @ ak.T
            akNew = ak @ ak
            err = la.norm(qkNew - qk, 'fro') / (la.norm(qkNew, 'fro') + 1e-10)
            qk, ak, X = 0.5 * (qkNew + qkNew.T), akNew, qkNew
            if err < tol:
                return X, True
        return X, False

    @staticmethod
    def _solveDiscreteRichardson(A, Q, maxIter=1000, tol=1e-8):
        n = A.shape[0]
        X = np.zeros((n, n))
        for k in range(maxIter):
            xNew = A @ X @ A.T + Q
            xNew = 0.5 * (xNew + xNew.T)
            if k % 10 == 0:
                if la.norm(xNew - X, 'fro') / (la.norm(xNew, 'fro') + 1e-10) < tol:
                    return xNew, True
            X = xNew
        return X, False

    @staticmethod
    def _getSquareRoot(M: np.ndarray) -> np.ndarray:
        """SVD-based Cholesky decomposition for potentially singular matrices."""
        U, s, _ = la.svd(M)
        return U * np.sqrt(np.maximum(s, 0))


def solveContinuous(A: np.ndarray, B: np.ndarray, E: np.ndarray | None = None, 
                    trans: bool = False, isLRCF: bool = False, **kwargs) -> Any:
    """Convenience wrapper for continuous Lyapunov equations."""
    return lyapSolver.solve(A, B, E, EquationType.CONT, trans, isLRCF, **kwargs)


def solveDiscrete(A: np.ndarray, B: np.ndarray, E: np.ndarray | None = None, 
                  trans: bool = False, isLRCF: bool = False, **kwargs) -> Any:
    """Convenience wrapper for discrete Lyapunov equations."""
    return lyapSolver.solve(A, B, E, EquationType.DISC, trans, isLRCF, **kwargs)
