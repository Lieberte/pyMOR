import numpy as np
import time
from scipy import linalg as la
from scipy.sparse.linalg import gmres, LinearOperator, splu
import warnings
import gc

warnings.filterwarnings('ignore')


# Iterative Solvers

def solveLyapunovMcg(A, Q, maxIter=5000, tol=1e-10):
    """Modified Conjugate Gradient"""
    n = A.shape[0]
    X = np.zeros((n, n))
    R = A @ X + X @ A.T + Q
    Z = -(A.T @ R + R @ A)

    for k in range(maxIter):
        resNorm = np.linalg.norm(R, 'fro')
        if resNorm < tol:
            return X, k + 1, True
        if np.any(np.isnan(X)) or resNorm > 1e10:
            return X, k + 1, False

        azZat = A @ Z + Z @ A.T
        denom = np.sum(azZat**2)
        if denom < 1e-20:
            return X, k + 1, False

        alpha = -np.sum(R * azZat) / denom
        X = X + alpha * Z
        X = 0.5 * (X + X.T)

        RNew = A @ X + X @ A.T + Q
        gradNew = A.T @ RNew + RNew @ A
        gradOld = A.T @ R + R @ A

        beta = np.sum(gradNew * (gradNew - gradOld)) / (np.sum(gradOld**2) + 1e-20)
        Z = -gradNew + max(0, beta) * Z
        R = RNew

    return X, maxIter, False


def solveLyapunovRichardson(A, Q, maxIter=1000, tol=1e-8):
    """Richardson Iteration"""
    n = A.shape[0]
    X = np.zeros((n, n))

    eigvals = np.linalg.eigvals(A + A.T)
    omega = 2.0 / (np.min(np.real(eigvals)) + np.max(np.real(eigvals)) + 1e-10)

    for k in range(maxIter):
        R = A @ X + X @ A.T + Q
        X = X - omega * R
        
        if k % 20 == 0:
            X = 0.5 * (X + X.T)
            resNorm = np.linalg.norm(R, 'fro')
            if resNorm < tol:
                return X, k + 1, True
            if np.any(np.isnan(X)) or resNorm > 1e10:
                return X, k + 1, False

    return 0.5 * (X + X.T), maxIter, False


def solveLyapunovAdiOptimized(A, Q, maxIter=50, tol=1e-8, numShifts=5):
    """ADI with LU reuse"""
    n = A.shape[0]
    X = np.zeros((n, n))

    eigvals = np.linalg.eigvals(A)
    eigReal = np.real(eigvals)
    pMin = np.min(np.abs(eigReal))
    pMax = np.max(np.abs(eigReal))

    shifts = -np.logspace(np.log10(pMin + 1e-6), np.log10(pMax + 1e-6), numShifts)

    luFactors = []
    AmI_list = []

    for p in shifts:
        ApI = A + p * np.eye(n)
        AmI = A - p * np.eye(n)
        try:
            lu, piv = la.lu_factor(ApI)
            luFactors.append((lu, piv))
            AmI_list.append(AmI)
        except:
            return X, 0, False

    for k in range(maxIter):
        idx = k % numShifts
        lu, piv = luFactors[idx]
        AmI = AmI_list[idx]

        try:
            rhs1 = -Q - X @ AmI.T
            V = la.lu_solve((lu, piv), rhs1)
            rhs2 = -Q - V.T @ AmI.T
            X = la.lu_solve((lu, piv), rhs2).T
            X = 0.5 * (X + X.T)
        except:
            return X, k + 1, False

        if k % numShifts == numShifts - 1:
            resNorm = np.linalg.norm(A @ X + X @ A.T + Q, 'fro')
            if resNorm < tol:
                return X, k + 1, True
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                return X, k + 1, False

    return X, maxIter, False


def solveLyapunovGmres(A, Q, maxIter=200, tol=1e-10):
    """GMRES for vectorized Lyapunov"""
    n = A.shape[0]

    def lOp(v):
        X = v.reshape((n, n))
        return (A @ X + X @ A.T).ravel()

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


# Benchmark

def createSystem(n, stableShift=2.0):
    """Create stable test system"""
    np.random.seed(42 + n)
    A = np.random.randn(n, n) * 0.5
    eigvals = np.linalg.eigvals(A)
    A = A - np.eye(n) * (np.max(np.real(eigvals)) + stableShift)
    Q = np.eye(n)
    return A, Q


def safeError(X, XRef):
    """Compute relative error safely"""
    try:
        if X is None or XRef is None:
            return None
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            return None
        err = np.linalg.norm(X - XRef, 'fro') / (np.linalg.norm(XRef, 'fro') + 1e-10)
        return err if not (np.isnan(err) or np.isinf(err)) else None
    except:
        return None


def benchmark(n):
    """Run benchmark for all methods"""
    A, Q = createSystem(n)
    results = {}

    # Reference solution (SciPy)
    try:
        t0 = time.time()
        XRef = la.solve_continuous_lyapunov(A, -Q)
        tScipy = time.time() - t0
        errScipy = np.linalg.norm(A @ XRef + XRef @ A.T + Q, 'fro') / np.linalg.norm(Q, 'fro')
        results['scipy'] = {
            'time': tScipy,
            'error': errScipy,
            'iters': '-',
            'success': True
        }
    except:
        XRef = None
        results['scipy'] = {'time': None, 'error': None, 'iters': '-', 'success': False}

    # Test iterative methods
    methods = {
        'mcg': (solveLyapunovMcg, {'maxIter': 5000, 'tol': 1e-10}),
        'richardson': (solveLyapunovRichardson, {'maxIter': 1000, 'tol': 1e-8}),
        'gmres': (solveLyapunovGmres, {'maxIter': 200, 'tol': 1e-10}),
        'adi': (solveLyapunovAdiOptimized, {'maxIter': 50, 'tol': 1e-8, 'numShifts': 5})
    }

    for name, (func, kwargs) in methods.items():
        try:
            t0 = time.time()
            X, iters, success = func(A, Q, **kwargs)
            tSolve = time.time() - t0

            err = safeError(X, XRef)

            results[name] = {
                'time': tSolve,
                'error': err,
                'iters': iters,
                'success': success and err is not None
            }

            del X
            gc.collect()
        except Exception as e:
            results[name] = {'time': None, 'error': None, 'iters': '-', 'success': False}

    # Cleanup
    del A, Q, XRef
    gc.collect()

    return results


# Main

if __name__ == "__main__":
    print(f"{'n':<6} {'Method':<14} {'Time(s)':<12} {'Error':<12} {'Iters':<10} {'Status':<8}")
    print("-" * 65)

    for n in [50, 100, 200, 500]:
        res = benchmark(n)
        first = True
        for method in ['scipy', 'mcg', 'richardson', 'gmres', 'adi']:
            r = res[method]
            nStr = str(n) if first else ''
            first = False

            if r['time'] is not None:
                timeStr = f"{r['time']:.4f}"
                errorStr = f"{r['error']:.2e}" if r['error'] is not None else "NaN"
                itersStr = str(r['iters'])
                statusStr = '✓' if r['success'] else '✗'
            else:
                timeStr = "FAIL"
                errorStr = "-"
                itersStr = "-"
                statusStr = '✗'

            print(f"{nStr:<6} {method:<14} {timeStr:<12} {errorStr:<12} {itersStr:<10} {statusStr:<8}")
        print()