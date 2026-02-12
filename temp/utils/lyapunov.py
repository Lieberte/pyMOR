import numpy as np
import scipy.linalg as la
from typing import Tuple, Literal


MethodType = Literal['scipy', 'adi', 'richardson', 'doubling']


def solve_continuous_lyapunov(a: np.ndarray, q: np.ndarray,
                              method: MethodType = 'scipy',
                              maxIter: int = 1000,
                              tol: float = 1e-8) -> Tuple[np.ndarray, bool]:
    """
    Solve continuous Lyapunov equation AX + XA^T + Q = 0
    """
    if method == 'scipy':
        try:
            return la.solve_continuous_lyapunov(a, -q), True
        except:
            method = 'adi'

    if method == 'adi':
        x, iters, success = _solve_continuous_adi_impl(a, q, maxIter=50, tol=tol)
        if success:
            return x, True
        method = 'richardson'

    if method == 'richardson':
        x, iters, success = _solve_continuous_richardson_impl(a, q, maxIter=maxIter, tol=tol)
        return x, success

    raise ValueError(f"Unknown method: {method}")


def solve_discrete_lyapunov(a: np.ndarray, q: np.ndarray,
                            method: MethodType = 'doubling',
                            maxIter: int = 1000,
                            tol: float = 1e-8) -> Tuple[np.ndarray, bool]:
    """
    Solve discrete Lyapunov equation AXA^T - X + Q = 0
    """
    if method == 'doubling':
        x, iters, success = _solve_discrete_doubling_impl(a, q, maxIter=50, tol=tol)
        if success:
            return x, True
        method = 'richardson'

    if method == 'richardson':
        x, iters, success = _solve_discrete_richardson_impl(a, q, maxIter=maxIter, tol=tol)
        if success:
            return x, True
        method = 'scipy'

    if method == 'scipy':
        try:
            return la.solve_discrete_lyapunov(a, q), True
        except:
            return q, False

    raise ValueError(f"Unknown method: {method}")


# 底层实现函数
def _solve_continuous_richardson_impl(A, Q, maxIter, tol):
    n = A.shape[0]
    X = np.zeros((n, n))
    eigvals = np.linalg.eigvals(A + A.T)
    omega = 2.0 / (np.min(np.real(eigvals)) + np.max(np.real(eigvals)) + 1e-10)
    for k in range(maxIter):
        R = A @ X + X @ A.T + Q
        X = X - omega * R
        if k % 20 == 0:
            X = 0.5 * (X + X.T)
            if np.linalg.norm(R, 'fro') < tol:
                return X, k + 1, True
    return 0.5 * (X + X.T), maxIter, False


def _solve_continuous_adi_impl(A, Q, maxIter, tol, numShifts=5):
    n = A.shape[0]
    X = np.zeros((n, n))
    eigvals = np.linalg.eigvals(A)
    eigvals_real = np.real(eigvals)
    p_min, p_max = np.min(np.abs(eigvals_real)), np.max(np.abs(eigvals_real))
    shifts = -np.logspace(np.log10(p_min + 1e-6), np.log10(p_max + 1e-6), numShifts)
    lu_factors, am_i_list = [], []
    for p in shifts:
        try:
            lu, piv = la.lu_factor(A + p * np.eye(n))
            lu_factors.append((lu, piv))
            am_i_list.append(A - p * np.eye(n))
        except: return X, 0, False
    for k in range(maxIter):
        idx = k % numShifts
        lu, piv, am_i = lu_factors[idx][0], lu_factors[idx][1], am_i_list[idx]
        try:
            V = la.lu_solve((lu, piv), -Q - X @ am_i.T)
            X = la.lu_solve((lu, piv), -Q - V.T @ am_i.T).T
            X = 0.5 * (X + X.T)
        except: return X, k + 1, False
        if k % numShifts == numShifts - 1:
            if np.linalg.norm(A @ X + X @ A.T + Q, 'fro') < tol:
                return X, k + 1, True
    return X, maxIter, False


def _solve_discrete_doubling_impl(A, Q, maxIter, tol):
    ak, qk, X = A.copy(), Q.copy(), Q.copy()
    for k in range(maxIter):
        qk_new = qk + ak @ qk @ ak.T
        ak_new = ak @ ak
        err = np.linalg.norm(qk_new - qk, 'fro') / (np.linalg.norm(qk_new, 'fro') + 1e-10)
        qk, ak, X = 0.5 * (qk_new + qk_new.T), ak_new, qk_new
        if err < tol:
            return X, k + 1, True
    return X, maxIter, False


def _solve_discrete_richardson_impl(A, Q, maxIter, tol):
    n = A.shape[0]
    X = np.zeros((n, n))
    for k in range(maxIter):
        x_new = A @ X @ A.T + Q
        x_new = 0.5 * (x_new + x_new.T)
        if k % 10 == 0:
            if np.linalg.norm(x_new - X, 'fro') / (np.linalg.norm(x_new, 'fro') + 1e-10) < tol:
                return x_new, k + 1, True
        X = x_new
    return X, maxIter, False


def solveLyapunovContinuous(a: np.ndarray, q: np.ndarray, **kwargs) -> np.ndarray:
    x, converged = solve_continuous_lyapunov(a, q, **kwargs)
    if not converged:
        print("[WARNING] Lyapunov solver did not converge")
    return x


def solveLyapunovDiscrete(a: np.ndarray, q: np.ndarray, **kwargs) -> np.ndarray:
    x, converged = solve_discrete_lyapunov(a, q, **kwargs)
    if not converged:
        print("[WARNING] Lyapunov solver did not converge")
    return x