"""SVD algorithm contract tests.

Covers:
  * economic/dual/incremental variants must accept a COMPOSED operator
    (sumOperator) -- they previously read ``xOperator.data`` directly and
    raised AttributeError on anything but a matrixOperator.
  * dualSVD must work on the torch backend -- it previously used
    ``argsort(evals)[::-1]`` which torch tensors do not support.
"""
import numpy as np
import pytest

from mor.operators import matrixOperator
from mor.algorithm.registry import algorithmRegistry


@pytest.fixture
def rng():
    return np.random.default_rng(1)


@pytest.fixture
def composed(rng):
    M1 = matrixOperator(rng.standard_normal((6, 4)), backendName='scipy')
    M2 = matrixOperator(rng.standard_normal((6, 4)), backendName='scipy')
    return M1 + M2, (M1.toBackendData() + M2.toBackendData())


@pytest.mark.parametrize('variant', ['economic', 'dual'])
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
    torch = pytest.importorskip('torch')
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
