"""SVD algorithm contract tests.

Covers:
  * economic/dual/incremental variants must accept a COMPOSED operator
    (sumOperator) -- they previously read ``xOperator.data`` directly and
    raised AttributeError on anything but a matrixOperator.
  * dualSVD must work on the torch backend -- it previously used
    ``argsort(evals)[::-1]`` which torch tensors do not support.
"""
import torch
import numpy as np
import pytest
from scipy.sparse import diags

from mor.algorithm.auto.svd import selectSVDVariant
from mor.operators import matrixOperator
from mor.algorithm.decompose.truncation.truncation import truncate
from mor.backends import backendRegistry
from mor.algorithm.registry import algorithmRegistry


@pytest.fixture
def rng():
    return np.random.default_rng(1)


@pytest.fixture
def composed(rng):
    M1 = matrixOperator(rng.standard_normal((6, 4)), backendName='scipy')
    M2 = matrixOperator(rng.standard_normal((6, 4)), backendName='scipy')
    return M1 + M2, (M1.toBackendData() + M2.toBackendData())


@pytest.mark.parametrize('variant', ['economic', 'dual', 'static', 'qrSVD'])
def test_svd_on_composed_operator(variant, composed):
    op, dense = composed
    algo = algorithmRegistry.get(category='svd', variant=variant, backendName='scipy')
    U, S, Vt = algo.decompose(op, rank=3)
    ref = np.linalg.svd(dense, compute_uv=False)
    np.testing.assert_allclose(np.sort(S)[::-1], np.sort(ref[:3])[::-1])
    # factor shapes are consistent
    assert U.shape[1] == 3 and Vt.shape[0] == 3


def test_dual_svd_descending_order(rng):
    """dualSVD must return singular values in descending order (truncation relies on it)."""
    M = matrixOperator(rng.standard_normal((8, 5)), backendName='scipy')
    algo = algorithmRegistry.get(category='svd', variant='dual', backendName='scipy')
    _, S, _ = algo.decompose(M, rank=5)
    assert np.all(np.diff(S) <= 1e-10), f"not descending: {S}"


def test_dual_svd_torch_backend():
    """torch tensors do not support negative-stride slicing ([::-1]); argsort(-evals) is safe."""
    rng = np.random.default_rng(2)
    M = matrixOperator(rng.standard_normal((6, 4)), backendName='torch')
    algo = algorithmRegistry.get(category='svd', variant='dual', backendName='torch')
    U, S, Vt = algo.decompose(M, rank=3)
    dense = M.toBackendData()
    ref = torch.linalg.svdvals(dense)[:3].cpu().numpy()
    np.testing.assert_allclose(np.sort(S.cpu().numpy())[::-1], np.sort(ref)[::-1])


def test_incremental_standard_on_composed_operator(composed):
    op, dense = composed
    algo = algorithmRegistry.get(category='svd', variant='incrementalStandard', backendName='scipy')
    U, S, Vt = algo.decompose(op, rank=3)
    ref = np.linalg.svd(dense, compute_uv=False)
    np.testing.assert_allclose(np.sort(S)[::-1], np.sort(ref[:3])[::-1])


def test_svd_auto_uses_static_when_centering_is_requested(rng):
    op = matrixOperator(rng.standard_normal((10, 4)), backendName='scipy')
    assert selectSVDVariant(backendName='scipy', xOperator=op, centerSnapshots=True) == 'static'


def test_svd_auto_uses_incremental_standard_for_streaming_mode(rng):
    op = matrixOperator(rng.standard_normal((10, 4)), backendName='scipy')
    assert selectSVDVariant(backendName='scipy', xOperator=op, incremental=True) == 'incrementalStandard'


def test_svd_auto_prefers_qrsvd_for_tall_skinny_scipy_operator(rng):
    op = matrixOperator(rng.standard_normal((40, 8)), backendName='scipy')
    assert selectSVDVariant(backendName='scipy', xOperator=op) == 'qrSVD'


def test_svd_auto_prefers_dual_for_large_tall_skinny_scipy_operator(rng):
    op = matrixOperator(rng.standard_normal((1000, 20)), backendName='scipy')
    assert selectSVDVariant(backendName='scipy', xOperator=op) == 'dual'


def test_svd_auto_prefers_economic_for_general_dense_scipy_operator(rng):
    op = matrixOperator(rng.standard_normal((8, 8)), backendName='scipy')
    assert selectSVDVariant(backendName='scipy', xOperator=op) == 'economic'


def test_svd_auto_uses_operator_metadata_for_medium_diagonal_scipy_operator():
    op = matrixOperator(np.diag(np.logspace(0.0, -12.0, 100)), backendName='scipy')
    assert op.isDiagonal
    assert selectSVDVariant(backendName='scipy', xOperator=op, rank=100) == 'dual'


def test_svd_auto_keeps_economic_for_large_diagonal_scipy_operator():
    op = matrixOperator(np.diag(np.logspace(0.0, -12.0, 300)), backendName='scipy')
    assert op.isDiagonal
    assert selectSVDVariant(backendName='scipy', xOperator=op, rank=300) == 'economic'


def test_svd_auto_uses_operator_metadata_for_medium_composed_scipy_operator(rng):
    op = (
        matrixOperator(rng.standard_normal((100, 100)), backendName='scipy')
        + matrixOperator(rng.standard_normal((100, 100)), backendName='scipy')
    )
    assert op.isComposed
    assert selectSVDVariant(backendName='scipy', xOperator=op, rank=100) == 'dual'


def test_svd_auto_keeps_economic_for_large_composed_scipy_operator(rng):
    op = (
        matrixOperator(rng.standard_normal((196, 196)), backendName='scipy')
        + matrixOperator(rng.standard_normal((196, 196)), backendName='scipy')
    )
    assert op.isComposed
    assert selectSVDVariant(backendName='scipy', xOperator=op, rank=196) == 'economic'


def test_svd_auto_prefers_economic_for_small_torch_operator():
    op = matrixOperator(torch.randn(12, 4), backendName='torch')
    assert selectSVDVariant(backendName='torch', xOperator=op) == 'economic'


def test_svd_auto_prefers_dual_for_large_torch_cpu_tall_skinny_operator():
    op = matrixOperator(torch.randn(1000, 20, dtype=torch.float64, device='cpu'), backendName='torch')
    assert selectSVDVariant(backendName='torch', xOperator=op) == 'dual'


def test_svd_auto_prefers_dual_for_large_torch_cuda_tall_skinny_operator():
    if not torch.cuda.is_available():
        pytest.skip('CUDA torch is not available')
    op = matrixOperator(torch.randn(1000, 20, dtype=torch.float64, device='cuda'), backendName='torch')
    assert selectSVDVariant(backendName='torch', xOperator=op) == 'dual'


def test_svd_auto_prefers_dual_for_cuda_tall_skinny_operator_even_when_small():
    if not torch.cuda.is_available():
        pytest.skip('CUDA torch is not available')
    op = matrixOperator(torch.randn(100, 10, dtype=torch.float64, device='cuda'), backendName='torch')
    assert selectSVDVariant(backendName='torch', xOperator=op) == 'dual'


def test_svd_auto_uses_dual_for_weighted_product_svd(rng):
    op = matrixOperator(rng.standard_normal((40, 8)), backendName='scipy')
    product = matrixOperator(np.eye(40), backendName='scipy')
    assert selectSVDVariant(backendName='scipy', xOperator=op, product=product) == 'dual'


@pytest.mark.parametrize('rank', [-1, 1.5, True])
def test_truncation_rejects_invalid_rank(rank):
    with pytest.raises(ValueError, match='rank'):
        truncate(np.array([3.0, 2.0, 1.0]), rank=rank, backend=backendRegistry.get('scipy'))


@pytest.mark.parametrize('tol', [-0.1, 1.1, True])
def test_truncation_rejects_invalid_tolerance(tol):
    with pytest.raises(ValueError, match='tol'):
        truncate(np.array([3.0, 2.0, 1.0]), tol=tol, backend=backendRegistry.get('scipy'))


def test_explicit_svd_variant_is_not_overridden_by_force_options(rng):
    op = matrixOperator(rng.standard_normal((12, 5)), backendName='scipy')

    algorithm = algorithmRegistry.get(
        category='svd', variant='economic', forceOptions={'variant': 'dual'}, backendName='scipy', xOperator=op
    )

    assert algorithm.__class__.__name__ == 'economicSVD'


def test_svd_auto_uses_economic_for_sparse_tall_operator():
    op = matrixOperator(diags(np.ones(100), format='csr')[:, :20], backendName='scipy')
    assert selectSVDVariant(backendName='scipy', xOperator=op) == 'economic'


def test_svd_auto_uses_economic_for_sparse_wide_operator():
    op = matrixOperator(diags(np.ones(20), offsets=0, shape=(20, 100), format='csr'), backendName='scipy')
    assert selectSVDVariant(backendName='scipy', xOperator=op) == 'economic'


@pytest.mark.parametrize('shape', [(12, 5), (5, 12)])
@pytest.mark.parametrize('dtype', [np.int64, np.float64, np.complex128])
def test_sparse_truncated_svd_preserves_sparse_input(shape, dtype, monkeypatch):
    from scipy.sparse import csr_matrix

    data = diags(np.arange(1, 6), shape=shape, format='csr', dtype=dtype)
    if dtype == np.complex128:
        data = data * (1 + 2j)
    dense = data.toarray()
    monkeypatch.setattr(csr_matrix, 'toarray', lambda *a, **k: pytest.fail('densified input'))
    operator = matrixOperator(data, backendName='scipy')
    algorithm = algorithmRegistry.get(category='svd', variant='auto', backendName='scipy', xOperator=operator)
    U, S, Vt = algorithm.decompose(operator, rank=2)
    np.testing.assert_allclose(S, np.linalg.svd(dense, compute_uv=False)[:2], atol=1e-12)
    np.testing.assert_allclose(U.conj().T @ U, np.eye(2), atol=1e-12)
    refU, refS, refVt = np.linalg.svd(dense, full_matrices=False)
    np.testing.assert_allclose(U @ np.diag(S) @ Vt,
                               (refU[:, :2] * refS[:2]) @ refVt[:2], atol=1e-12)


@pytest.mark.parametrize('variant', ['economic', 'dual'])
def test_sparse_full_svd_and_energy_truncation(variant):
    data = diags([4.0, 3.0, 1.0], shape=(8, 3), format='csr')
    operator = matrixOperator(data, backendName='scipy')
    algorithm = algorithmRegistry.get(category='svd', variant=variant, backendName='scipy')
    U, S, Vt = algorithm.decompose(operator)
    np.testing.assert_allclose(U @ np.diag(S) @ Vt, data.toarray(), atol=1e-12)
    _, S, _ = algorithm.decompose(operator, tol=0.05)
    np.testing.assert_allclose(S, [4.0, 3.0])


def test_sparse_full_decomposition_rejects_unbounded_dense_conversion(monkeypatch):
    import mor.backends.scipyBackend as scipyModule

    monkeypatch.setattr(scipyModule, '_MAX_DENSE_DECOMPOSITION_BYTES', 16)
    operator = matrixOperator(diags([4.0, 3.0, 1.0], format='csr'), backendName='scipy')
    algorithm = algorithmRegistry.get(category='svd', variant='economic', backendName='scipy')
    with pytest.raises(ValueError, match='dense conversion limit'):
        algorithm.decompose(operator)
    _, S, _ = algorithm.decompose(operator, rank=1)
    np.testing.assert_allclose(S, [4.0])


@pytest.mark.parametrize('diagonal', [[0., 0., 0., 0.], [4., 0., 0., 0.], [1., 1e-6, 1e-12, 1e-14],
                                     [4e-200, 3e-200, 2e-200, 1e-200],
                                     [4e200, 3e200, 2e200, 1e200]])
def test_sparse_truncated_svd_handles_degenerate_spectra(diagonal):
    operator = matrixOperator(diags(diagonal, shape=(8, 4), format='csr'), 'scipy')
    algorithm = algorithmRegistry.get(category='svd', variant='auto', backendName='scipy', xOperator=operator)
    U, S, Vt = algorithm.decompose(operator, rank=2)
    np.testing.assert_allclose(S, np.sort(diagonal)[::-1][:2], atol=1e-214, rtol=1e-6)
    assert np.isfinite(U).all()
    assert np.isfinite(Vt).all()


def test_sparse_svd_accepts_composed_operator():
    data = diags([4., 3., 2., 1.], shape=(8, 4), format='csr')
    operator = matrixOperator(data, 'scipy') + matrixOperator(data, 'scipy')
    algorithm = algorithmRegistry.get(category='svd', variant='auto', backendName='scipy', xOperator=operator)
    _, S, _ = algorithm.decompose(operator, rank=2)
    np.testing.assert_allclose(S, [8., 6.], atol=1e-12)


@pytest.mark.parametrize('shape', [(5, 5), (129, 129), (10, 4)])
def test_svd_auto_skips_unused_structure_detection(shape, monkeypatch):
    operator = matrixOperator(np.ones(shape), 'scipy')
    monkeypatch.setattr(operator, '_detectDiagonal', lambda: pytest.fail('unneeded structure scan'))
    assert selectSVDVariant(backendName='scipy', xOperator=operator) in ('economic', 'qrSVD')


@pytest.mark.parametrize('backendName', ['scipy', 'torch'])
@pytest.mark.parametrize('complexData', [False, True])
def test_wide_dual_svd_matches_dense_svd_with_small_gram(backendName, complexData, monkeypatch):
    rng = np.random.default_rng(31)
    data = rng.normal(size=(6, 25))
    if complexData:
        data = data + 1j * rng.normal(size=data.shape)
    if backendName == 'torch':
        dataForBackend = torch.from_numpy(data)
    else:
        dataForBackend = data
    operator = matrixOperator(dataForBackend, backendName)
    algorithm = algorithmRegistry.get(category='svd', variant='dual', backendName=backendName)
    original = algorithm.localBackend.eigen.eigh

    def smallGram(gram):
        assert gram.shape == (6, 6)
        return original(gram)

    monkeypatch.setattr(algorithm.localBackend.eigen, 'eigh', smallGram)
    U, S, Vt = algorithm.decompose(operator, rank=3)
    convert = algorithm.localBackend.array.toNumpy
    U, S, Vt = convert(U), convert(S), convert(Vt)
    refU, refS, refVt = np.linalg.svd(data, full_matrices=False)
    np.testing.assert_allclose(S, refS[:3], rtol=1e-12)
    np.testing.assert_allclose((U * S) @ Vt, (refU[:, :3] * refS[:3]) @ refVt[:3], atol=1e-12)
    np.testing.assert_allclose(U.conj().T @ U, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(Vt @ Vt.conj().T, np.eye(3), atol=1e-12)


@pytest.mark.parametrize(
    ('kwargs', 'expected'),
    [
        ({'centerSnapshots': True, 'product': object(), 'incremental': True}, 'incrementalStandard'),
        ({'centerSnapshots': True, 'product': object()}, 'static'),
        ({'product': object()}, 'dual'),
    ],
)
def test_svd_auto_dispatch_semantic_precedence(rng, kwargs, expected):
    op = matrixOperator(rng.standard_normal((1000, 20)), backendName='scipy')
    assert selectSVDVariant(backendName='scipy', xOperator=op, **kwargs) == expected


def test_svd_registry_auto_uses_qrsvd_for_tall_skinny_operator(rng):
    dense = rng.standard_normal((40, 8))
    op = matrixOperator(dense, backendName='scipy')
    algo = algorithmRegistry.get(category='svd', variant='auto', backendName='scipy', xOperator=op)

    assert algo.__class__.__name__ == 'qrSVD'


def test_svd_registry_auto_uses_dual_for_large_tall_skinny_operator(rng):
    dense = rng.standard_normal((1000, 20))
    op = matrixOperator(dense, backendName='scipy')
    algo = algorithmRegistry.get(category='svd', variant='auto', backendName='scipy', xOperator=op)

    assert algo.__class__.__name__ == 'dualSVD'


def test_svd_registry_auto_uses_static_without_operator_context():
    algo = algorithmRegistry.get(category='svd', variant='auto', backendName='scipy')

    assert algo.__class__.__name__ == 'staticSVD'


@pytest.mark.parametrize('variant', ['incrementalFast', 'incrementalBrand', 'incrementalStandard'])
def test_incremental_svd_reconstructs_cumulative_matrix_after_second_batch(variant, rng):
    X1 = rng.standard_normal((6, 3))
    coeffs = rng.standard_normal((3, 2))
    X2 = X1 @ coeffs
    algo = algorithmRegistry.get(category='svd', variant=variant, backendName='scipy')

    algo.decompose(matrixOperator(X1, backendName='scipy'), rank=3)
    U, S, Vt = algo.decompose(matrixOperator(X2, backendName='scipy'), rank=3)

    combined = np.hstack([X1, X2])
    np.testing.assert_allclose(U @ np.diag(S) @ Vt, combined)


@pytest.mark.parametrize('variant', ['incrementalFast', 'incrementalBrand', 'incrementalStandard'])
def test_incremental_svd_reconstructs_cumulative_matrix_after_third_batch(variant, rng):
    X1 = rng.standard_normal((6, 2))
    X2 = rng.standard_normal((6, 2))
    X3 = rng.standard_normal((6, 1))
    algo = algorithmRegistry.get(category='svd', variant=variant, backendName='scipy')

    algo.decompose(matrixOperator(X1, backendName='scipy'), rank=5)
    algo.decompose(matrixOperator(X2, backendName='scipy'), rank=5)
    U, S, Vt = algo.decompose(matrixOperator(X3, backendName='scipy'), rank=5)

    combined = np.hstack([X1, X2, X3])
    np.testing.assert_allclose(U @ np.diag(S) @ Vt, combined)
