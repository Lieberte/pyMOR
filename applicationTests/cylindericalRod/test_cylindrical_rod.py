"""
Test cylindrical rod benchmark, mirroring MATLAB cylindricalRod.m:

  load cylindricalRod.mat
  sys = sparss(A,B,C,D,E);
  R = reducespec(sys,"balanced");
  R = process(R)
  rsys = getrom(R,MaxError=1e-6,Method="truncate");

We test the Lyapunov/Gramian step (balanced truncation prep) with our mor solvers.
"""
import pytest
import numpy as np

pytest.importorskip("scipy")

from scipy.io import loadmat

from mor.operators import matrixOperator
from mor.solvers.lyapunov import lyapunovRegistry
from mor.reductors import BalancedTruncationReductor


def _extract(x):
    if isinstance(x, np.ndarray) and x.ndim == 0 and x.dtype == object:
        return x.item()
    return x


def _loadCylindricalRod():
    import os
    path = os.path.join(os.path.dirname(__file__), "cylindricalRod.mat")
    if not os.path.isfile(path):
        pytest.skip("cylindricalRod.mat not found")
    data = loadmat(path)
    a = _extract(data.get("A", data.get("a", None)))
    b = _extract(data.get("B", data.get("b", None)))
    c = _extract(data.get("C", data.get("c", None)))
    e = _extract(data.get("E", data.get("e", None)))
    if a is None or b is None or c is None:
        pytest.skip("cylindricalRod.mat missing A,B,C")
    if e is None:
        e = np.eye(a.shape[0], dtype=np.float64)
    if hasattr(b, "ndim") and b.ndim == 1:
        b = b[:, np.newaxis]
    if hasattr(c, "ndim") and c.ndim == 1:
        c = c[np.newaxis, :]
    return a, b, c, e

class TestCylindricalRod:
    @pytest.fixture
    def system(self):
        return _loadCylindricalRod()

    def test_load_mat(self, system):
        a, b, c, e = system
        assert a.shape[0] == a.shape[1]
        assert e.shape == a.shape
        assert b.shape[0] == a.shape[0]
        assert c.shape[1] == a.shape[0]

    def test_controllability_gramian_lr_generalized(self, system):
        """Controllability: A P E' + E P A' + B B' = 0 (low-rank)"""
        a, b, c, e = system
        A = matrixOperator(a, backendName="scipy")
        E = matrixOperator(e, backendName="scipy")
        B = matrixOperator(b, backendName="scipy")
        solver = lyapunovRegistry.get("continuousLrGeneralized", backendName="scipy")
        P = solver.solve(A, E, B)
        z = P.toNumpy()
        P_full = z @ z.T
        res = a @ P_full @ e.T + e @ P_full @ a.T + b @ b.T
        np.testing.assert_allclose(res, np.zeros_like(res), atol=1e-6)

    def test_controllability_gramian_hr_generalized(self, system):
        """Controllability: A P E' + E P A' + B B' = 0 (high-rank)"""
        a, b, c, e = system
        A = matrixOperator(a, backendName="scipy")
        E = matrixOperator(e, backendName="scipy")
        B = matrixOperator(b, backendName="scipy")
        solver = lyapunovRegistry.get("continuousHrGeneralized", backendName="scipy")
        P = solver.solve(A, E, B)
        pData = P.toNumpy()
        res = a @ pData @ e.T + e @ pData @ a.T + b @ b.T
        np.testing.assert_allclose(res, np.zeros_like(res), atol=1e-6)

    def test_observability_gramian_lr_generalized(self, system):
        """Observability: A' Q E + E' Q A + C' C = 0 (dual of generalized Lyapunov)"""
        a, b, c, e = system
        A = matrixOperator(a.T, backendName="scipy")
        E = matrixOperator(e.T, backendName="scipy")
        C = matrixOperator(c.T, backendName="scipy")
        solver = lyapunovRegistry.get("continuousLrGeneralized", backendName="scipy")
        Q = solver.solve(A, E, C)
        z = Q.toNumpy()
        Q_full = z @ z.T
        res = a.T @ Q_full @ e + e.T @ Q_full @ a + c.T @ c
        np.testing.assert_allclose(res, np.zeros_like(res), atol=1e-6)

    def test_observability_gramian_hr_generalized(self, system):
        """Observability: A' Q E + E' Q A + C' C = 0 (high-rank)"""
        a, b, c, e = system
        A = matrixOperator(a.T, backendName="scipy")
        E = matrixOperator(e.T, backendName="scipy")
        C = matrixOperator(c.T, backendName="scipy")
        solver = lyapunovRegistry.get("continuousHrGeneralized", backendName="scipy")
        Q = solver.solve(A, E, C)
        qData = Q.toNumpy()
        res = a.T @ qData @ e + e.T @ qData @ a + c.T @ c
        np.testing.assert_allclose(res, np.zeros_like(res), atol=1e-6)

    def test_lr_hr_gramian_consistency(self, system):
        """LR and HR controllability Gramians should match (up to truncation)"""
        a, b, c, e = system
        A = matrixOperator(a, backendName="scipy")
        E = matrixOperator(e, backendName="scipy")
        B = matrixOperator(b, backendName="scipy")
        solverLr = lyapunovRegistry.get("continuousLrGeneralized", backendName="scipy")
        solverHr = lyapunovRegistry.get("continuousHrGeneralized", backendName="scipy")
        P_lr = solverLr.solve(A, E, B).toNumpy()
        P_hr = solverHr.solve(A, E, B).toNumpy()
        P_lr_full = P_lr @ P_lr.T
        diff = np.linalg.norm(P_hr - P_lr_full, "fro")
        rel = np.linalg.norm(P_hr, "fro")
        rel = np.maximum(rel, 1e-14)
        assert diff / rel < 0.1

    def test_balanced_truncation_small(self):
        """Full BT on small synthetic system."""
        np.random.seed(42)
        n = 20
        A = -np.eye(n) + np.random.randn(n, n) * 0.1
        B = np.random.randn(n, 2)
        C = np.random.randn(3, n)
        reductor = BalancedTruncationReductor(backendName="scipy")
        rsys = reductor.reduce(
            matrixOperator(A, backendName="scipy"),
            matrixOperator(B, backendName="scipy"),
            matrixOperator(C, backendName="scipy"),
            order=5,
        )
        assert rsys.Ar.shape[0] == rsys.Ar.shape[1] == 5
        assert rsys.Br.shape == (5, 2)
        assert rsys.Cr.shape == (3, 5)
        assert len(rsys.hsv) == 5
        assert rsys.order == 5

    @pytest.mark.slow
    def test_balanced_truncation_cylindrical(self, system):
        """Full BT on cylindrical rod (order=10, ~7522 states)."""
        a, b, c, e = system
        A = matrixOperator(a, backendName="scipy")
        B = matrixOperator(b, backendName="scipy")
        C = matrixOperator(c, backendName="scipy")
        E = matrixOperator(e, backendName="scipy")
        reductor = BalancedTruncationReductor(backendName="scipy")
        rsys = reductor.reduce(A, B, C, E=E, order=10)
        assert rsys.Ar.shape[0] == rsys.Ar.shape[1] == 10
        assert rsys.Br.shape[1] == B.shape[1]
        assert rsys.Cr.shape[0] == C.shape[0]
        assert rsys.Er.shape == (10, 10)
