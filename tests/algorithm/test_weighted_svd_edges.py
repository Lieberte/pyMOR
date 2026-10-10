import torch
import numpy as np
import pytest
from scipy.sparse import diags

from mor.algorithm.registry import algorithmRegistry
from mor.operators import matrixOperator
from mor.reductors.pod import podReductor


@pytest.mark.parametrize('backendName', ['scipy', 'torch'])
@pytest.mark.parametrize('diagonal', [False, True])
@pytest.mark.parametrize('complexData', [False, True])
def test_weighted_near_rank_deficiency_has_weight_orthonormal_basis(backendName, diagonal, complexData):
    rng = np.random.default_rng(8)
    q = np.linalg.qr(rng.normal(size=(12, 12)))[0]
    weight = np.diag(np.linspace(1., 3., 12)) if diagonal else q @ np.diag(np.linspace(1., 3., 12)) @ q.T
    left = rng.normal(size=(12, 4))
    right = rng.normal(size=(9, 4))
    if complexData:
        left = left + 1j * rng.normal(size=left.shape)
        right = right + 1j * rng.normal(size=right.shape)
        weight = weight.astype(complex)
    factor = np.linalg.cholesky(weight).conj().T
    left = np.linalg.solve(factor, np.linalg.qr(left)[0])
    spectrum = np.array([1., 1e-4, 1e-8, 1e-12])
    values = (left * spectrum) @ np.linalg.qr(right)[0].conj().T
    if backendName == 'torch':
        convert = torch.from_numpy
        valuesInput, weightInput = convert(values), convert(weight)
    else:
        valuesInput = values
        weightInput = diags(weight.diagonal()) if diagonal else weight
    U, s, V = algorithmRegistry.get('svd', 'dual', backendName=backendName).decompose(
        matrixOperator(valuesInput, backendName), rank=4, product=matrixOperator(weightInput, backendName))
    if backendName == 'torch': U, s, V = U.numpy(), s.numpy(), V.numpy()
    np.testing.assert_allclose(U.conj().T @ weight @ U, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(s, spectrum, rtol=1e-4, atol=1e-15)
    np.testing.assert_allclose((U * s) @ V, values, atol=1e-14)


@pytest.mark.parametrize('weight', [np.diag([1., -1.]), np.array([[1., 1.], [0., 1.]])])
def test_weighted_svd_rejects_nonpositive_or_nonhermitian_product(weight):
    with pytest.raises(ValueError, match='positive definite'):
        algorithmRegistry.get('svd', 'dual', backendName='scipy').decompose(
            matrixOperator(np.eye(2), 'scipy'), product=matrixOperator(weight, 'scipy'))


def test_sparse_weighted_pod_uses_diagonal_product_and_reconstructs():
    from scipy.sparse import csr_matrix
    x = csr_matrix(np.array([[1., 2., 3.], [0., 1., 2.], [0., 0., 0.]]))
    result = podReductor('scipy', product=matrixOperator(diags([1., 2., 3.]), 'scipy')).reduce(matrixOperator(x, 'scipy'), rank=2)
    assert result.reconstructionError < 1e-12


@pytest.mark.parametrize('backendName', ['scipy', 'torch', 'sparse'])
@pytest.mark.parametrize('scale', [1e-200, 1e200])
def test_weighted_pod_diagnostic_is_scale_invariant(backendName, scale):
    x = np.diag([1., 0.1, 0.01]) * scale
    w = np.diag([1., 2., 3.])
    if backendName == 'torch':
        x, w = torch.from_numpy(x), torch.from_numpy(w)
    elif backendName == 'sparse':
        from scipy.sparse import csr_matrix
        x, w = csr_matrix(x), diags(w.diagonal())
    backend = 'scipy' if backendName == 'sparse' else backendName
    result = podReductor(backend, product=matrixOperator(w, backend)).reduce(matrixOperator(x, backend), rank=1)
    assert result.reconstructionError == pytest.approx(np.sqrt(0.0203 / 1.0203), rel=1e-12)
