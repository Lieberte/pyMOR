import torch
import numpy as np
import pytest

from mor.algorithm.registry import algorithmRegistry
from mor.operators import matrixOperator


@pytest.mark.parametrize('backendName', ['scipy', 'torch'])
@pytest.mark.parametrize('shape', [(24, 9), (9, 24)])
@pytest.mark.parametrize('complexData', [False, True])
@pytest.mark.parametrize('variant', ['dual', 'qrSVD'])
def test_ill_conditioned_svd_preserves_small_singular_values_and_orthogonality(backendName, shape, complexData, variant):
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(shape[0], 4)), rng.normal(size=(shape[1], 4))
    if complexData:
        a = a + 1j * rng.normal(size=a.shape)
        b = b + 1j * rng.normal(size=b.shape)
    u, v = np.linalg.qr(a)[0], np.linalg.qr(b)[0]
    spectrum = np.array([1., 1e-4, 1e-8, 1e-12])
    data = (u * spectrum) @ v.conj().T
    values = data
    if backendName == 'torch': values = torch.from_numpy(data)
    U, s, V = algorithmRegistry.get('svd', variant, backendName=backendName).decompose(matrixOperator(values, backendName), rank=4)
    if backendName == 'torch': U, s, V = U.numpy(), s.numpy(), V.numpy()
    np.testing.assert_allclose(s, spectrum, atol=1e-15, rtol=1e-4)
    np.testing.assert_allclose(U.conj().T @ U, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(V @ V.conj().T, np.eye(4), atol=1e-12)
    np.testing.assert_allclose((U * s) @ V, data, atol=1e-14)


@pytest.mark.parametrize('scale', [1e200, 1e-200])
def test_dual_svd_normalizes_before_forming_gram(scale):
    values = np.diag([scale, scale / 2])
    U, s, V = algorithmRegistry.get('svd', 'dual', backendName='scipy').decompose(matrixOperator(values, 'scipy'), rank=2)
    np.testing.assert_allclose(s / scale, [1., 0.5])
    np.testing.assert_allclose((U * (s / scale)) @ V, values / scale)
