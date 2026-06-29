import numpy as np
import pytest

from mor.algorithm.registry import algorithmRegistry
from mor.backends import backendRegistry
from mor.operators import lowRankOperator, matrixOperator


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
    np.testing.assert_allclose(np.linalg.norm(z.left), 1.0)


def test_lrsmith_scipy_accepts_transpose_contract():
    A = matrixOperator(np.eye(3), backendName='scipy')
    B = matrixOperator(np.ones((2, 3)), backendName='scipy')
    smith = algorithmRegistry.get(category='lyapunov', variant='lrsmith', backendName='scipy', maxIter=1)

    z = smith.solve(A, None, B, trans=True)

    assert isinstance(z, lowRankOperator)
    assert z.left.shape == (3, 2)


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
