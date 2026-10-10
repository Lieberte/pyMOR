import numpy as np
import pytest
from scipy import linalg
from scipy.sparse import diags

from mor.algorithm.registry import algorithmRegistry
from mor.backends import backendRegistry
from mor.operators import lowRankOperator, matrixOperator
from scripts.standard_benchmark_cases import matrix_depot_poisson_1d


def _discrete_relative_residual(a_data, x_data, b_data):
    q_data = b_data @ b_data.T
    residual = a_data @ x_data @ a_data.T - x_data + q_data
    return np.linalg.norm(residual) / np.linalg.norm(q_data)


def test_backend_array_max_is_available_for_lrsmith_scipy():
    backend = backendRegistry.get('scipy')
    assert backend.array.max(backend.array.array([1.0, 3.0, 2.0])) == 3.0


def test_lrsmith_scipy_executes_one_iteration_without_missing_array_max():
    A = matrixOperator(np.zeros((3, 3)), backendName='scipy')
    B = matrixOperator(np.ones((3, 1)), backendName='scipy')
    smith = algorithmRegistry.get(category='lyapunov', variant='lrsmith', backendName='scipy', maxIter=1)

    z = smith.solve(A, None, B)

    assert isinstance(z, lowRankOperator)
    assert z.shape == (3, 3)
    assert z.left.shape == (3, 1)
    np.testing.assert_allclose(z.left, B.toBackendData())


def test_lrsmith_single_iteration_only_applies_operator_for_residual(monkeypatch):
    A = matrixOperator(np.zeros((3, 3)), 'scipy')
    B = matrixOperator(np.ones((3, 1)), 'scipy')
    original = A.apply
    calls = []
    def apply(values, trans=False):
        calls.append(values.shape)
        return original(values, trans=trans)
    monkeypatch.setattr(A, 'apply', apply)
    smith = algorithmRegistry.get(category='lyapunov', variant='lrsmith', backendName='scipy',
                                  maxIter=1, checkStability=False)
    result = smith.solve(A, None, B)
    assert calls == [(3, 1)]
    np.testing.assert_allclose(result.toBackendData(), np.ones((3, 3)))


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('trans', [False, True])
def test_descriptor_factor_is_reused_only_within_one_solve(sparse, trans, monkeypatch):
    import scipy.linalg
    import scipy.sparse.linalg
    from scipy.sparse import csr_matrix
    Adata = np.array([[0.3, 0.1], [0., 0.4]])
    Edata = np.array([[1., 0.2], [0., 2.]])
    Bdata = np.ones((2, 1))
    A = matrixOperator(csr_matrix(Adata) if sparse else Adata, 'scipy')
    B = matrixOperator(Bdata.T if trans else Bdata, 'scipy')
    calls = []
    module = scipy.sparse.linalg if sparse else scipy.linalg
    name = 'splu' if sparse else 'lu_factor'
    original = getattr(module, name)
    def factor(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, name, factor)
    smith = algorithmRegistry.get(category='lyapunov', variant='lrsmith', backendName='scipy',
                                  maxIter=80, tol=1e-12, checkStability=False)
    for scale in [1., 2.]:
        E = matrixOperator(csr_matrix(Edata * scale) if sparse else Edata * scale, 'scipy')
        result = smith.solve(A, E, B, trans=trans).toBackendData()
        descriptor = Edata.T * scale if trans else Edata * scale
        effectiveA = np.linalg.solve(descriptor, Adata.T if trans else Adata)
        effectiveB = np.linalg.solve(descriptor, Bdata)
        reference = scipy.linalg.solve_discrete_lyapunov(effectiveA, effectiveB @ effectiveB.T)
        np.testing.assert_allclose(result, reference, atol=1e-10)
    assert len(calls) == 2


def test_lrsmith_scipy_solves_stable_discrete_residual():
    A = matrixOperator(np.diag([0.2, 0.4, 0.6]), backendName='scipy')
    B = matrixOperator(np.ones((3, 1)), backendName='scipy')
    smith = algorithmRegistry.get(category='lyapunov', variant='lrsmith', backendName='scipy', maxIter=100, tol=1e-12)

    z = smith.solve(A, None, B)
    x = z.toBackendData()
    q = B.toBackendData() @ B.toBackendData().T
    residual = A.toBackendData() @ x @ A.toBackendData().T - x + q

    assert isinstance(z, lowRankOperator)
    assert smith.lastIterations > 1
    assert smith.lastConverged
    assert smith.lastResidual <= smith.lastTolerance
    assert len(smith.lastResidualHistory) == smith.lastIterations
    assert np.linalg.norm(residual) / np.linalg.norm(q) < 1e-10


def test_lrsmith_scipy_solves_matrix_depot_poisson_discrete_case():
    a_data = 0.2 * matrix_depot_poisson_1d(100).toarray()
    b_data = np.ones((100, 1))
    A = matrixOperator(a_data, backendName='scipy')
    B = matrixOperator(b_data, backendName='scipy')
    smith = algorithmRegistry.get(category='lyapunov', variant='lrsmith', backendName='scipy', maxIter=160, tol=1e-12)

    z = smith.solve(A, None, B)
    x = z.toBackendData()
    x_ref = linalg.solve_discrete_lyapunov(a_data, b_data @ b_data.T)

    assert isinstance(z, lowRankOperator)
    assert smith.lastIterations < 160
    assert _discrete_relative_residual(a_data, x, b_data) < 1e-10
    assert np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref) < 1e-10


def test_lrsmith_scipy_solves_multi_input_poisson_against_dense_reference():
    a_data = 0.2 * matrix_depot_poisson_1d(100).toarray()
    b_data = np.column_stack([np.ones(100), np.linspace(0.5, 1.5, 100)])
    A = matrixOperator(a_data, backendName='scipy')
    B = matrixOperator(b_data, backendName='scipy')
    smith = algorithmRegistry.get(category='lyapunov', variant='lrsmith', backendName='scipy', maxIter=160, tol=1e-12)

    z = smith.solve(A, None, B)
    x = z.toBackendData()
    x_ref = linalg.solve_discrete_lyapunov(a_data, b_data @ b_data.T)

    assert isinstance(z, lowRankOperator)
    assert z.left.shape[1] >= b_data.shape[1]
    assert smith.lastIterations < 160
    assert _discrete_relative_residual(a_data, x, b_data) < 1e-10
    assert np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref) < 1e-10


def test_lrsmith_near_unit_poisson_hits_iteration_budget_before_accuracy():
    poisson = matrix_depot_poisson_1d(100).toarray()
    a_data = np.eye(100) - 0.2 * poisson
    b_data = np.ones((100, 1))
    A = matrixOperator(a_data, backendName='scipy')
    B = matrixOperator(b_data, backendName='scipy')
    smith = algorithmRegistry.get(category='lyapunov', variant='lrsmith', backendName='scipy', maxIter=80, tol=1e-12)

    z = smith.solve(A, None, B)
    x = z.toBackendData()

    assert max(abs(np.linalg.eigvals(a_data))) > 0.99
    assert smith.lastIterations == 80
    assert not smith.lastConverged
    assert _discrete_relative_residual(a_data, x, b_data) > 1e-2


@pytest.mark.parametrize(
    ("rho", "max_iter", "expected_residual_bound"),
    [
        (0.8, 120, 1e-10),
        (0.95, 320, 1e-10),
        (0.995, 320, 1e-5),
    ],
)
def test_lrsmith_diagonal_convergence_degrades_near_unit_circle(rho, max_iter, expected_residual_bound):
    a_data = np.diag(np.linspace(0.1, rho, 80))
    b_data = np.ones((80, 1))
    A = matrixOperator(a_data, backendName='scipy')
    B = matrixOperator(b_data, backendName='scipy')
    smith = algorithmRegistry.get(category='lyapunov', variant='lrsmith', backendName='scipy', maxIter=max_iter, tol=1e-12)

    z = smith.solve(A, None, B)
    x = z.toBackendData()
    residual = _discrete_relative_residual(a_data, x, b_data)

    assert smith.lastIterations <= max_iter
    if rho < 0.99:
        assert residual < expected_residual_bound
        assert smith.lastConverged
    else:
        assert smith.lastIterations == max_iter
        assert not smith.lastConverged
        assert residual > expected_residual_bound


def test_lrsmith_unstable_discrete_case_is_rejected_when_radius_is_available():
    a_data = np.diag([0.8, 1.01, 1.05])
    b_data = np.ones((3, 1))
    A = matrixOperator(a_data, backendName='scipy')
    B = matrixOperator(b_data, backendName='scipy')
    smith = algorithmRegistry.get(category='lyapunov', variant='lrsmith', backendName='scipy', maxIter=40, tol=1e-12)

    with pytest.raises(ValueError, match='stable discrete operator'):
        smith.solve(A, None, B)

    assert smith.lastSpectralRadiusEstimate > 1.0
    assert not smith.lastConverged


def test_lrsmith_descriptor_case_matches_transformed_discrete_problem():
    a_data = np.diag([0.2, 0.3, 0.4])
    e_data = np.diag([1.0, 2.0, 4.0])
    f_data = np.linalg.solve(e_data, a_data)
    b_data = np.ones((3, 1))
    A = matrixOperator(a_data, backendName='scipy')
    E = matrixOperator(e_data, backendName='scipy')
    B = matrixOperator(b_data, backendName='scipy')
    smith = algorithmRegistry.get(category='lyapunov', variant='lrsmith', backendName='scipy', maxIter=100, tol=1e-12)

    z = smith.solve(A, E, B)
    x = z.toBackendData()
    g_data = np.linalg.solve(e_data, b_data)
    x_ref = linalg.solve_discrete_lyapunov(f_data, g_data @ g_data.T)

    assert _discrete_relative_residual(f_data, x, g_data) < 1e-10
    assert np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref) < 1e-10


def test_lrsmith_descriptor_solution_satisfies_generalized_equation():
    a_data = np.diag([0.2, 0.3, 0.4])
    e_data = np.diag([1.0, 2.0, 4.0])
    b_data = np.ones((3, 1))
    smith = algorithmRegistry.get(category='lyapunov', variant='lrsmith', backendName='scipy', maxIter=100, tol=1e-12)

    z = smith.solve(
        matrixOperator(a_data, backendName='scipy'),
        matrixOperator(e_data, backendName='scipy'),
        matrixOperator(b_data, backendName='scipy'),
    )
    x = z.toBackendData()
    residual = a_data @ x @ a_data.T - e_data @ x @ e_data.T + b_data @ b_data.T

    assert np.linalg.norm(residual) / np.linalg.norm(b_data @ b_data.T) < 1e-10
    assert smith.lastConverged


def test_lrsmith_max_rank_uses_true_residual_for_convergence():
    a_data = np.diag(np.linspace(0.1, 0.8, 20))
    b_data = np.column_stack([np.ones(20), np.linspace(0.5, 1.5, 20)])
    smith = algorithmRegistry.get(
        category='lyapunov', variant='lrsmith', backendName='scipy', maxIter=80, tol=1e-12, maxRank=2
    )

    z = smith.solve(
        matrixOperator(a_data, backendName='scipy'),
        None,
        matrixOperator(b_data, backendName='scipy'),
    )
    actual = _discrete_relative_residual(a_data, z.toBackendData(), b_data)

    assert actual > 1e-4
    assert not smith.lastConverged


def test_lrsmith_sparse_operator_path_avoids_dense_materialization_for_iterations():
    n = 1000
    a_data = 0.2 * matrix_depot_poisson_1d(n)
    b_data = np.column_stack([np.ones(n), np.linspace(0.5, 1.5, n)])
    A = matrixOperator(a_data, backendName='scipy')
    B = matrixOperator(b_data, backendName='scipy')
    smith = algorithmRegistry.get(category='lyapunov', variant='lrsmith', backendName='scipy', maxIter=180, tol=1e-12)

    z = smith.solve(A, None, B)
    az = A.apply(z.left)
    low_rank_residual = smith.discreteLyapunovResidualNorm(az, z.left, b_data) / np.linalg.norm(b_data.T @ b_data)

    assert A.isSparse
    assert z.left.shape[0] == n
    assert smith.lastIterations < 180
    assert smith.lastConverged
    assert low_rank_residual < 1e-7


def test_lrsmith_zero_rhs_returns_zero_solution_factor():
    A = matrixOperator(diags([0.2, 0.3, 0.4], format='csr'), backendName='scipy')
    B = matrixOperator(np.zeros((3, 1)), backendName='scipy')
    smith = algorithmRegistry.get(category='lyapunov', variant='lrsmith', backendName='scipy', maxIter=20, tol=1e-12)

    z = smith.solve(A, None, B)

    assert isinstance(z, lowRankOperator)
    assert smith.lastConverged
    np.testing.assert_allclose(z.toBackendData(), np.zeros((3, 3)))
    assert smith.lastIterations == 1


def test_lrsmith_scipy_accepts_transpose_contract():
    A = matrixOperator(0.5 * np.eye(3), backendName='scipy')
    B = matrixOperator(np.ones((2, 3)), backendName='scipy')
    smith = algorithmRegistry.get(category='lyapunov', variant='lrsmith', backendName='scipy', maxIter=1)

    z = smith.solve(A, None, B, trans=True)

    assert isinstance(z, lowRankOperator)
    assert z.left.shape == (3, 2)


def test_lrsmith_scipy_accepts_descriptor_operator_contract():
    A = matrixOperator(0.5 * np.eye(3), backendName='scipy')
    E = matrixOperator(np.eye(3), backendName='scipy') * 1.0
    B = matrixOperator(np.ones((3, 1)), backendName='scipy')
    smith = algorithmRegistry.get(category='lyapunov', variant='lrsmith', backendName='scipy', maxIter=1)

    z = smith.solve(A, E, B)

    assert isinstance(z, lowRankOperator)
    assert z.left.shape == (3, 1)


def test_low_rank_operator_rejects_mismatched_rank_factors_early():
    left = np.zeros((6, 2))
    right = np.zeros((6, 3))

    with pytest.raises(ValueError, match='same number of columns'):
        lowRankOperator(left, right, backendName='scipy')


def test_low_rank_operator_to_backend_data_matches_dense_general_factor():
    left = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    right = np.array([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]])

    op = lowRankOperator(left, right, backendName='scipy')

    assert op.shape == (3, 3)
    np.testing.assert_allclose(op.toBackendData(), left @ right.T)


def test_low_rank_operator_general_shape_tracks_both_factor_row_counts():
    left = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    right = np.array([[0.5, 1.0], [1.5, 2.0]])

    op = lowRankOperator(left, right, backendName='scipy')

    assert op.shape == (3, 2)
    np.testing.assert_allclose(op.toBackendData(), left @ right.T)


def test_low_rank_apply_accepts_operator_input():
    left = np.array([[1.0], [2.0], [3.0]])
    op = lowRankOperator(left, backendName='scipy')
    x = matrixOperator(np.ones((3, 2)), backendName='scipy')

    np.testing.assert_allclose(op.apply(x), (left @ left.T) @ x.toBackendData())
