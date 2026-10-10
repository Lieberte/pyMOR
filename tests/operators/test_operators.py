"""Operator contract tests: lowRankOperator / sumOperator must be instantiable
(they implement the abstract toBackendData) and lowRankOperator.apply must work
(it previously referenced an un-imported ``matrixOperator`` -> NameError).
"""
import torch
import numpy as np
import pytest
from scipy.sparse import diags, issparse

from mor.operators import matrixOperator, lowRankOperator, scaledOperator, sumOperator


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_low_rank_operator_instantiates(rng):
    """lowRankOperator was uninstantiable: it did not override abstract toBackendData."""
    L = rng.standard_normal((6, 2))
    op = lowRankOperator(L, backendName='scipy')
    assert op.shape == (6, 6)
    assert op.left.shape == (6, 2)
    assert op.isLowRank
    assert op.metadata['materializationCost'] == 12


def test_sum_operator_instantiates_and_to_backend_data(rng):
    """sumOperator (A + B) was uninstantiable for the same reason."""
    M1 = matrixOperator(rng.standard_normal((6, 4)), backendName='scipy')
    M2 = matrixOperator(rng.standard_normal((6, 4)), backendName='scipy')
    composed = M1 + M2
    dense_ref = M1.toBackendData() + M2.toBackendData()
    np.testing.assert_allclose(composed.toBackendData(), dense_ref)
    assert composed.isComposed
    assert 'composed' in composed.metadata['structureTags']


def test_matrix_operator_preserves_scipy_sparse_data():
    sparse = diags([-1.0, -2.0, -3.0], format='csr')
    op = matrixOperator(sparse, backendName='scipy')

    assert op.shape == (3, 3)
    assert op.isSparse
    assert issparse(op.toBackendData())
    np.testing.assert_allclose(op.toNumpy(), sparse.toarray())


def test_matrix_operator_metadata_detects_diagonal_dense_and_sparse():
    dense = matrixOperator(np.diag([1.0, 2.0, 3.0]), backendName='scipy')
    sparse = matrixOperator(diags([1.0, 2.0, 3.0], format='csr'), backendName='scipy')
    general = matrixOperator(np.ones((3, 3)), backendName='scipy')

    assert dense.isDiagonal
    assert sparse.isDiagonal
    assert not general.isDiagonal
    assert 'diagonal' in dense.metadata['structureTags']
    assert 'diagonal' in sparse.metadata['structureTags']
    assert dense.metadata['materializationCost'] == 9


@pytest.mark.parametrize('scale', [1., 1e-10, 1e-100])
def test_diagonal_metadata_preserves_small_off_diagonal_entries(scale):
    data = np.array([[scale, scale / 10], [0., scale]])
    operator = matrixOperator(data, backendName='scipy')
    assert not operator.isDiagonal
    assert 'diagonal' not in operator.metadata['structureTags']
    with pytest.raises(ValueError, match='not diagonal'):
        operator.diagonal()


def test_diagonal_detection_is_lazy_and_cached(monkeypatch):
    calls = []
    original = matrixOperator._detectDiagonal

    def detect(operator):
        calls.append(operator)
        return original(operator)

    monkeypatch.setattr(matrixOperator, '_detectDiagonal', detect)
    operator = matrixOperator(np.eye(3), backendName='scipy')
    transposed = operator.T
    scaled = operator * 2
    assert calls == []
    assert operator.isDiagonal
    assert operator.metadata['isDiagonal']
    assert calls == [operator]
    assert transposed.isDiagonal
    assert scaled.isDiagonal


def test_dense_diagonal_detection_does_not_build_diagonal_matrix(monkeypatch):
    operator = matrixOperator(np.eye(4, dtype=np.complex128), backendName='scipy')
    monkeypatch.setattr(np, 'diag', lambda *args, **kwargs: pytest.fail('allocated diagonal matrix'))
    assert operator.isDiagonal


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


@pytest.mark.parametrize('operator_factory', [
    lambda: matrixOperator(np.eye(2), backendName='scipy'),
    lambda: lowRankOperator(np.eye(2), backendName='scipy'),
    lambda: matrixOperator(np.eye(2), backendName='scipy') + matrixOperator(np.eye(2), backendName='scipy'),
])
def test_operator_apply_rejects_input_operator_from_another_backend(operator_factory):
    operator = operator_factory()
    x = matrixOperator(torch.ones((2, 1)), backendName='torch')

    with pytest.raises(ValueError, match='Input operator backend torch does not match operator backend scipy'):
        operator.apply(x)


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


def test_complex_low_rank_uses_adjoint_factors_and_preserves_transpose_contract(rng):
    left = rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2))
    right = rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2))
    symmetric = lowRankOperator(left, backendName='scipy')
    general = lowRankOperator(left, right, backendName='scipy')
    x = rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2))

    np.testing.assert_allclose(symmetric.toBackendData(), left @ left.conj().T)
    np.testing.assert_allclose(general.toBackendData(), left @ right.conj().T)
    np.testing.assert_allclose(general.T.toBackendData(), general.toBackendData().T)
    np.testing.assert_allclose(general.apply(x, trans=True), general.toBackendData().T @ x)


def test_complex_symmetric_low_rank_transpose_is_not_mistaken_for_adjoint(rng):
    left = rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2))
    op = lowRankOperator(left, backendName='scipy')

    np.testing.assert_allclose(op.T.toBackendData(), op.toBackendData().T)
    np.testing.assert_allclose(op.toBackendData(), op.toBackendData().conj().T)


def test_low_rank_transpose_apply_does_not_construct_temporary_operator(monkeypatch, rng):
    left = rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2))
    right = rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2))
    op = lowRankOperator(left, right, backendName='scipy')
    x = rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2))
    monkeypatch.setattr(
        lowRankOperator,
        'T',
        property(lambda self: (_ for _ in ()).throw(AssertionError('constructed transpose operator'))),
    )

    result = op.apply(x, trans=True)

    np.testing.assert_allclose(result, (left @ right.conj().T).T @ x)


def test_torch_complex_low_rank_uses_adjoint_and_promoted_dtype():
    left = torch.tensor([[1.0 + 2.0j], [3.0 - 1.0j]], dtype=torch.complex64)
    right = torch.tensor([[2.0 - 1.0j], [1.0 + 4.0j], [0.5 + 0.0j]], dtype=torch.complex128)
    op = lowRankOperator(left, right, backendName='torch')

    expected = left.to(torch.complex128) @ right.conj().mT

    assert op.dtype == op.toBackendData().dtype == torch.complex128
    torch.testing.assert_close(op.toBackendData(), expected)
    torch.testing.assert_close(op.T.toBackendData(), expected.mT)


def test_low_rank_negative_scaling_preserves_operator_value(rng):
    L = rng.standard_normal((6, 2))
    op = lowRankOperator(L, backendName='scipy')

    scaled = -2.0 * op

    np.testing.assert_allclose(scaled.toBackendData(), -2.0 * (L @ L.T))
    assert np.isfinite(scaled.toBackendData()).all()


def test_left_scalar_multiplication_matches_right_multiplication(rng):
    data = rng.standard_normal((4, 3))
    op = matrixOperator(data, backendName='scipy')

    np.testing.assert_allclose((2.0 * op).toBackendData(), (op * 2.0).toBackendData())


def test_sum_operator_rejects_non_operator_entry(rng):
    op = matrixOperator(rng.standard_normal((3, 3)), backendName='scipy')

    with pytest.raises(TypeError, match='operatorBase'):
        _ = op + np.eye(3)


@pytest.mark.parametrize('operator_type', [sumOperator, scaledOperator])
def test_composed_operator_rejects_explicit_backend_mismatch(operator_type):
    op = matrixOperator(np.eye(2), backendName='scipy')

    with pytest.raises(ValueError, match='Requested backend torch does not match operator backend scipy'):
        if operator_type is sumOperator:
            operator_type([op], backendName='torch')
        else:
            operator_type(op, 2.0, backendName='torch')


def test_scipy_composed_operator_dtype_matches_promoted_data():
    float32 = matrixOperator(np.eye(2, dtype=np.float32), backendName='scipy')
    float64 = matrixOperator(np.eye(2, dtype=np.float64), backendName='scipy')

    summed = float32 + float64
    scaled = scaledOperator(float32, np.float64(0.5))

    assert summed.dtype == summed.toBackendData().dtype == np.dtype(np.float64)
    assert scaled.dtype == scaled.toBackendData().dtype == np.dtype(np.float64)


def test_torch_composed_operator_dtype_matches_promoted_data():
    float32 = matrixOperator(torch.eye(2, dtype=torch.float32), backendName='torch')
    float64 = matrixOperator(torch.eye(2, dtype=torch.float64), backendName='torch')

    summed = float32 + float64
    scaled = scaledOperator(float32, np.float64(0.5))

    assert summed.dtype == summed.toBackendData().dtype == torch.float64
    assert scaled.dtype == scaled.toBackendData().dtype == torch.float32


def test_nested_sum_operators_are_flattened(rng):
    operators = [
        matrixOperator(rng.standard_normal((3, 3)), backendName='scipy')
        for _ in range(3)
    ]

    nested = (operators[0] + operators[1]) + operators[2]

    assert len(nested.operators) == 3
    assert all(not isinstance(operator, sumOperator) for operator in nested.operators)
    np.testing.assert_allclose(nested.toBackendData(), sum(operator.toBackendData() for operator in operators))


def test_nested_scaled_operators_merge_scalars(rng):
    operator = matrixOperator(rng.standard_normal((3, 3)), backendName='scipy')

    scaled = scaledOperator(scaledOperator(operator, 2.0), -3.0)

    assert scaled.operator is operator
    assert scaled.scalar == -6.0
    np.testing.assert_allclose(scaled.toBackendData(), -6.0 * operator.toBackendData())
