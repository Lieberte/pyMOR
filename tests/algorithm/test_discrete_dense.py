import torch
import numpy as np
import pytest
from scipy import linalg

from mor.algorithm.registry import algorithmRegistry
from mor.operators import matrixOperator


@pytest.mark.parametrize('trans', [False, True])
@pytest.mark.parametrize('descriptor', [False, True])
@pytest.mark.parametrize('radius', [0.6, 0.999])
def test_dense_discrete_generalized_equation_and_reference(trans, descriptor, radius):
    a = np.array([[radius, 0.05], [0., 0.3]])
    e = np.array([[1., 0.1], [0., 2.]]) if descriptor else np.eye(2)
    a = e @ a
    b = np.array([[1., 0.2], [0.1, 2.]])
    solver = algorithmRegistry.get('lyapunov', 'discreteBartelsStewart', backendName='scipy')
    result = solver.solve(matrixOperator(a, 'scipy'), matrixOperator(e, 'scipy') if descriptor else None,
                          matrixOperator(b.T if trans else b, 'scipy'), trans=trans).toBackendData()
    at, et = (a.T, e.T) if trans else (a, e)
    aa, bb = np.linalg.solve(et, at), np.linalg.solve(et, b)
    reference = linalg.solve_discrete_lyapunov(aa, bb @ bb.T, method='direct')
    np.testing.assert_allclose(result, reference, rtol=1e-10)
    assert np.linalg.norm(at @ result @ at.T - et @ result @ et.T + b @ b.T) / np.linalg.norm(b @ b.T) < 1e-9
    assert solver.lastConverged


@pytest.mark.parametrize('radius', [1., 1.1])
def test_dense_discrete_rejects_unstable_system(radius):
    solver = algorithmRegistry.get('lyapunov', 'discreteBartelsStewart', backendName='scipy')
    with pytest.raises(ValueError, match='stable'):
        solver.solve(matrixOperator(np.eye(2) * radius, 'scipy'), None, matrixOperator(np.ones((2, 1)), 'scipy'))


def test_dense_discrete_rejects_singular_descriptor():
    solver = algorithmRegistry.get('lyapunov', 'discreteBartelsStewart', backendName='scipy')
    with pytest.raises((linalg.LinAlgWarning, linalg.LinAlgError)):
        solver.solve(matrixOperator(np.eye(2) * 0.3, 'scipy'), matrixOperator(np.zeros((2, 2)), 'scipy'),
                     matrixOperator(np.ones((2, 1)), 'scipy'))


def test_dense_discrete_auto_accepts_composed_operators_through_unified_solver():
    from mor.solvers.lyapunov import lyapunovSolver
    a = matrixOperator(np.diag([0.2, 0.3]), 'scipy') + matrixOperator(np.array([[0., 0.1], [0., 0.]]), 'scipy')
    e = matrixOperator(np.eye(2), 'scipy') * 2.
    b = matrixOperator(np.ones((2, 1)), 'scipy')
    solver = lyapunovSolver('scipy', isContinuous=False)
    x = solver.solve(a, e, b).toBackendData()
    aa, ee, bb = a.toBackendData(), e.toBackendData(), b.toBackendData()
    np.testing.assert_allclose(aa @ x @ aa.T - ee @ x @ ee.T + bb @ bb.T, 0., atol=1e-12)
    assert solver.lastInfo['algorithm'] == 'discreteBartelsStewart'


@pytest.mark.parametrize('trans', [False, True])
def test_dense_torch_discrete_auto_matches_scipy(trans):
    from mor.solvers.lyapunov import lyapunovSolver
    a, b = np.diag([0.2, 0.3]), np.ones((2, 1))
    solver = lyapunovSolver('torch', isContinuous=False, tol=1e-12)
    x = solver.solve(matrixOperator(torch.from_numpy(a), 'torch'), None,
                     matrixOperator(torch.from_numpy(b.T if trans else b), 'torch'), trans=trans).toBackendData().numpy()
    np.testing.assert_allclose(x, linalg.solve_discrete_lyapunov(a, b @ b.T), atol=1e-10)
    assert solver.lastInfo['converged']
    assert solver.lastInfo['algorithm'] == 'lrsmith'
