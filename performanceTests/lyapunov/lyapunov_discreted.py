import numpy as np
import time
from scipy import linalg as la
from scipy.sparse.linalg import gmres, LinearOperator
import warnings
import gc
import os

# 强制单线程，避免多线程干扰
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

warnings.filterwarnings('ignore')


# ============================================================================
# Iterative Solvers for Discrete Lyapunov: AXA^T - X = -Q
# ============================================================================

def solveLyapunovMcg(A, Q, maxIter=5000, tol=1e-10):
    """CGNR for discrete Lyapunov: X - AXA^T = Q"""
    n = A.shape[0]
    X = np.zeros((n, n))

    def L(Mat):
        return Mat - A @ Mat @ A.T

    def Lt(Mat):
        return Mat - A.T @ Mat @ A

    R = Q - L(X)
    Z = Lt(R)
    P = Z.copy()
    zNormOld = np.sum(Z ** 2)

    for k in range(maxIter):
        resNorm = np.linalg.norm(R, 'fro')
        if resNorm < tol:
            return X, k + 1, True

        LP = L(P)
        alpha = zNormOld / (np.sum(LP ** 2) + 1e-20)
        X = X + alpha * P
        X = 0.5 * (X + X.T)
        R = R - alpha * LP

        Z = Lt(R)
        zNormNew = np.sum(Z ** 2)
        if zNormNew < tol ** 2:
            return X, k + 1, True

        beta = zNormNew / (zNormOld + 1e-20)
        P = Z + beta * P
        zNormOld = zNormNew

    return X, maxIter, False


def solveLyapunovRichardson(A, Q, maxIter=1000, tol=1e-8):
    """Smith Iteration: X_{k+1} = A X_k A^T + Q"""
    n = A.shape[0]
    X = np.zeros((n, n))

    for k in range(maxIter):
        XNew = A @ X @ A.T + Q
        XNew = 0.5 * (XNew + XNew.T)

        if k % 10 == 0:
            resNorm = np.linalg.norm(XNew - X, 'fro') / (np.linalg.norm(XNew, 'fro') + 1e-10)
            if resNorm < tol:
                return XNew, k + 1, True

        X = XNew
        if np.any(np.isnan(X)) or np.linalg.norm(X, 'fro') > 1e10:
            return X, k + 1, False

    return X, maxIter, False


def solveLyapunovDoubling(A, Q, maxIter=50, tol=1e-8):
    """Doubling Algorithm (Squared Smith)"""
    n = A.shape[0]
    Ak = A.copy()
    Qk = Q.copy()
    X = Q.copy()

    for k in range(maxIter):
        QkNew = Qk + Ak @ Qk @ Ak.T
        AkNew = Ak @ Ak

        err = np.linalg.norm(QkNew - Qk, 'fro') / (np.linalg.norm(QkNew, 'fro') + 1e-10)
        Qk = 0.5 * (QkNew + QkNew.T)
        Ak = AkNew
        X = Qk

        if err < tol:
            return X, k + 1, True
        if np.any(np.isnan(X)) or np.linalg.norm(X, 'fro') > 1e10:
            return X, k + 1, False

    return X, maxIter, False


def solveLyapunovAdi(A, Q, maxIter=100, tol=1e-8, numShifts=5):
    """Discrete ADI: X = M X M^T + Qp"""
    n = A.shape[0]
    X = np.zeros((n, n))
    eigvals = np.linalg.eigvals(A)
    rho = np.abs(eigvals)
    shifts = np.linspace(np.min(rho), np.max(rho), numShifts)

    factors = []
    for p in shifts:
        try:
            ImPA = np.eye(n) - p * A
            AmPI = A - p * np.eye(n)
            lu, piv = la.lu_factor(ImPA)
            Y = la.lu_solve((lu, piv), Q)
            Qp = (1 - p ** 2) * la.lu_solve((lu, piv), Y.T).T
            factors.append((lu, piv, AmPI, Qp))
        except:
            continue

    if not factors:
        return X, 0, False

    for k in range(maxIter):
        lu, piv, AmPI, Qp = factors[k % len(factors)]
        try:
            V = la.lu_solve((lu, piv), X)
            X = AmPI @ la.lu_solve((lu, piv), V.T).T @ AmPI.T + Qp
            X = 0.5 * (X + X.T)
        except:
            return X, k + 1, False

        if k % len(factors) == len(factors) - 1:
            resid = np.linalg.norm(X - A @ X @ A.T - Q, 'fro')
            if resid < tol * np.linalg.norm(Q, 'fro'):
                return X, k + 1, True

    return X, maxIter, False


def solveLyapunovGmres(A, Q, maxIter=200, tol=1e-10):
    """GMRES for vectorized discrete Lyapunov"""
    n = A.shape[0]

    def lOp(v):
        X = v.reshape((n, n))
        return (A @ X @ A.T - X).ravel()

    L = LinearOperator((n * n, n * n), matvec=lOp)

    try:
        x, info = gmres(L, -Q.ravel(), maxiter=maxIter, atol=tol, rtol=tol, restart=50)

        if info >= 0:
            X = x.reshape((n, n))
            X = 0.5 * (X + X.T)
            return X, info if info > 0 else maxIter, True
        else:
            return None, 0, False
    except:
        return None, 0, False


# ============================================================================
# Utility Functions
# ============================================================================

def createSystem(n, rhoMax=0.9):
    """Create stable discrete system with independent random seed"""
    np.random.seed(42 + n * 1000)  # 独立种子
    A = np.random.randn(n, n) * 0.3
    eigvals = np.linalg.eigvals(A)
    A = A * (rhoMax / (np.max(np.abs(eigvals)) + 0.1))
    Q = np.eye(n)
    return A, Q


def computeResidual(A, X, Q):
    """Compute relative residual error (not timed)"""
    try:
        if X is None:
            return None
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            return None
        residual = A @ X @ A.T - X + Q
        relError = np.linalg.norm(residual, 'fro') / np.linalg.norm(Q, 'fro')
        return relError if not (np.isnan(relError) or np.isinf(relError)) else None
    except:
        return None


# ============================================================================
# Diagnostic Functions
# ============================================================================

def diagnose_scipy():
    """诊断 SciPy 性能，多次运行取平均"""
    print("\n" + "=" * 70)
    print("SciPy Performance Diagnosis (5 trials per size)")
    print("=" * 70)

    t_ref = None
    n_ref = 50

    for n in [50, 100, 200, 500]:
        np.random.seed(42 + n * 1000)

        # 创建系统
        A = np.random.randn(n, n) * 0.3
        eigvals = np.linalg.eigvals(A)
        rho_actual = np.max(np.abs(eigvals))
        A = A * (0.9 / (rho_actual + 0.1))
        Q = np.eye(n)

        rho_final = np.max(np.abs(np.linalg.eigvals(A)))

        # 预热
        _ = la.solve_discrete_lyapunov(A[:10, :10], Q[:10, :10])

        # 多次测试
        times = []
        for trial in range(5):
            t0 = time.perf_counter()
            X = la.solve_discrete_lyapunov(A, Q)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        # 计算误差（不计时）
        residual = A @ X @ A.T - X + Q
        err = np.linalg.norm(residual, 'fro') / np.linalg.norm(Q, 'fro')

        # 统计
        mean_t = np.mean(times)
        std_t = np.std(times)
        min_t = np.min(times)

        print(f"\nn = {n:3d}, ρ(A) = {rho_final:.4f}")
        print(f"  Time:  {mean_t:.4f} ± {std_t:.4f} s  (min: {min_t:.4f})")
        print(f"  Error: {err:.2e}")

        # 理论时间比
        if n == n_ref:
            t_ref = mean_t
        else:
            ratio_actual = mean_t / t_ref
            ratio_theory = (n / n_ref) ** 3
            print(f"  Speedup ratio: {ratio_actual:.2f}x (theory O(n³): {ratio_theory:.2f}x)")

        del X, A, Q
        gc.collect()


def compare_methods_detailed(n):
    """详细对比各方法的性能"""
    print(f"\n{'=' * 70}")
    print(f"Detailed Comparison for n = {n}")
    print(f"{'=' * 70}")

    np.random.seed(42 + n * 1000)
    A = np.random.randn(n, n) * 0.3
    eigvals = np.linalg.eigvals(A)
    A = A * (0.9 / (np.max(np.abs(eigvals)) + 0.1))
    Q = np.eye(n)

    rho = np.max(np.abs(np.linalg.eigvals(A)))
    print(f"System: n={n}, ρ(A)={rho:.4f}")
    print(f"\n{'Method':<15} {'Time(s)':<12} {'Error':<14} {'Iters':<10}")
    print("-" * 70)

    # SciPy (多次取平均)
    times_scipy = []
    for _ in range(3):
        t0 = time.perf_counter()
        X_scipy = la.solve_discrete_lyapunov(A, Q)
        times_scipy.append(time.perf_counter() - t0)

    t_scipy = np.mean(times_scipy)
    err_scipy = computeResidual(A, X_scipy, Q)
    print(f"{'SciPy':<15} {t_scipy:<12.4f} {err_scipy:<14.2e} {'-':<10}")

    # Richardson
    t0 = time.perf_counter()
    X_rich, iters_rich, success_rich = solveLyapunovRichardson(A, Q, maxIter=1000, tol=1e-8)
    t_rich = time.perf_counter() - t0
    err_rich = computeResidual(A, X_rich, Q)
    print(f"{'Richardson':<15} {t_rich:<12.4f} {err_rich:<14.2e} {iters_rich:<10}")

    # Doubling
    t0 = time.perf_counter()
    X_doub, iters_doub, success_doub = solveLyapunovDoubling(A, Q, maxIter=50, tol=1e-8)
    t_doub = time.perf_counter() - t0
    err_doub = computeResidual(A, X_doub, Q)
    print(f"{'Doubling':<15} {t_doub:<12.4f} {err_doub:<14.2e} {iters_doub:<10}")

    # MCG
    if n <= 200:  # MCG 对大系统太慢
        t0 = time.perf_counter()
        X_mcg, iters_mcg, success_mcg = solveLyapunovMcg(A, Q, maxIter=5000, tol=1e-10)
        t_mcg = time.perf_counter() - t0
        err_mcg = computeResidual(A, X_mcg, Q)
        print(f"{'MCG':<15} {t_mcg:<12.4f} {err_mcg:<14.2e} {iters_mcg:<10}")

    print(f"\nTime ratios (vs Doubling):")
    print(f"  SciPy/Doubling:     {t_scipy / t_doub:6.2f}x")
    print(f"  Richardson/Doubling: {t_rich / t_doub:6.2f}x")

    del X_scipy, X_rich, X_doub, A, Q
    gc.collect()


# ============================================================================
# Full Benchmark
# ============================================================================

def benchmark(n):
    """Run benchmark for all methods"""
    A, Q = createSystem(n)
    results = {}

    # 预热
    _ = la.solve_discrete_lyapunov(A[:10, :10], Q[:10, :10])

    # -------------------------------------------------------------------------
    # Reference solution (SciPy)
    # -------------------------------------------------------------------------
    try:
        # 多次运行取最小值（减少噪声）
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            XRef = la.solve_discrete_lyapunov(A, Q)
            times.append(time.perf_counter() - t0)

        tScipy = np.min(times)  # 取最小值

        # 误差计算（不计时）
        errScipy = computeResidual(A, XRef, Q)

        results['scipy'] = {
            'time': tScipy,
            'error': errScipy,
            'iters': '-',
            'success': True
        }
    except Exception as e:
        XRef = None
        results['scipy'] = {
            'time': None,
            'error': None,
            'iters': '-',
            'success': False
        }

    # -------------------------------------------------------------------------
    # Iterative methods
    # -------------------------------------------------------------------------
    methods = {
        'mcg': (solveLyapunovMcg, {'maxIter': 5000, 'tol': 1e-10}),
        'richardson': (solveLyapunovRichardson, {'maxIter': 1000, 'tol': 1e-8}),
        'gmres': (solveLyapunovGmres, {'maxIter': 200, 'tol': 1e-10}),
        'doubling': (solveLyapunovDoubling, {'maxIter': 50, 'tol': 1e-8}),
        'adi': (solveLyapunovAdi, {'maxIter': 300, 'tol': 1e-8})
    }

    for name, (func, kwargs) in methods.items():
        # MCG 对大系统太慢，跳过
        if name == 'mcg' and n > 200:
            results[name] = {'time': None, 'error': None, 'iters': '-', 'success': False}
            continue

        try:
            t0 = time.perf_counter()
            X, iters, success = func(A, Q, **kwargs)
            tSolve = time.perf_counter() - t0

            # 误差计算（不计时）
            err = computeResidual(A, X, Q)

            results[name] = {
                'time': tSolve,
                'error': err,
                'iters': iters,
                'success': success and err is not None
            }

            del X
            gc.collect()

        except Exception as e:
            results[name] = {
                'time': None,
                'error': None,
                'iters': '-',
                'success': False
            }

    # Cleanup
    del A, Q, XRef
    gc.collect()

    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # 首先运行诊断
    diagnose_scipy()

    # 详细对比
    for n in [50, 100, 200]:
        compare_methods_detailed(n)

    # 完整 Benchmark
    print("\n" + "=" * 80)
    print("Full Benchmark: Discrete Lyapunov Equation (AXA^T - X = -Q)")
    print("=" * 80)
    print(f"{'n':<8} {'Method':<14} {'Time(s)':<12} {'Error':<14} {'Iters':<10} {'Status':<8}")
    print("-" * 80)

    for n in [50, 100, 200, 500]:
        print(f"\n[n = {n}]")
        res = benchmark(n)

        for method in ['scipy', 'mcg', 'richardson', 'gmres', 'doubling', 'adi']:
            r = res[method]

            if r['time'] is not None:
                timeStr = f"{r['time']:.4f}"
                errorStr = f"{r['error']:.2e}" if r['error'] is not None else "NaN"
                itersStr = str(r['iters'])
                statusStr = '✓' if r['success'] else '✗'
            else:
                timeStr = "SKIP" if method == 'mcg' and n > 200 else "FAIL"
                errorStr = "-"
                itersStr = "-"
                statusStr = '-' if method == 'mcg' and n > 200 else '✗'

            print(f"{'':8} {method:<14} {timeStr:<12} {errorStr:<14} {itersStr:<10} {statusStr:<8}")

        print("-" * 80)

    print("\n" + "=" * 80)
    print("Benchmark Complete")
    print("=" * 80)