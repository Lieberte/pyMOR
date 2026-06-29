"""Operator contract tests: lowRankOperator / sumOperator must be instantiable
(they implement the abstract toBackendData) and lowRankOperator.apply must work
(it previously referenced an un-imported ``matrixOperator`` -> NameError).
"""
import numpy as np
import pytest

from mor.operators import matrixOperator, lowRankOperator


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_low_rank_operator_instantiates(rng):
    """lowRankOperator was uninstantiable: it did not override abstract toBackendData."""
    L = rng.standard_normal((6, 2))
    op = lowRankOperator(L, backendName='scipy')
    assert op.shape == (6, 6)
    assert op.left.shape == (6, 2)


def test_sum_operator_instantiates_and_to_backend_data(rng):
    """sumOperator (A + B) was uninstantiable for the same reason."""
    M1 = matrixOperator(rng.standard_normal((6, 4)), backendName='scipy')
    M2 = matrixOperator(rng.standard_normal((6, 4)), backendName='scipy')
    composed = M1 + M2
    dense_ref = M1.toBackendData() + M2.toBackendData()
    np.testing.assert_allclose(composed.toBackendData(), dense_ref)


def test_low_rank_apply_matches_dense(rng):
    """lowRankOperator.apply previously raised NameError on every call."""
    L = rng.standard_normal((6, 2))
    lr = lowRankOperator(L, backendName='scipy')
    x = matrixOperator(rng.standard_normal((6, 1)), backendName='scipy')
    y = lr.apply(x)
    np.testing.assert_allclose(y, (L @ L.T) @ x.toBackendData())


def test_low_rank_apply_accepts_matrix_operator_input(rng):
    """apply must route any operatorBase through toBackendData, not raw .data."""
    L = rng.standard_normal((5, 2))
    lr = lowRankOperator(L, backendName='scipy')
    # feeding another operatorBase (not a raw array) must not AttributeError
    x_op = matrixOperator(rng.standard_normal((5, 3)), backendName='scipy')
    out = lr.apply(x_op)
    assert out.shape == (5, 3)


def test_low_rank_to_backend_data_symmetric_and_general(rng):
    L = rng.standard_normal((6, 2))
    R = rng.standard_normal((6, 2))  # general factor L R^T needs matching inner dim
    np.testing.assert_allclose(
        lowRankOperator(L, backendName='scipy').toBackendData(),
        L @ L.T,
    )
    np.testing.assert_allclose(
        lowRankOperator(L, R, backendName='scipy').toBackendData(),
        L @ R.T,
    )
