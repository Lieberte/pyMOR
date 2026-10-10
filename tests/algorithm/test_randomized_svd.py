import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mor.algorithm.registry import algorithmRegistry
from mor.operators import matrixOperator


@pytest.mark.parametrize('sparse', [False, True])
def test_randomized_svd_seeded_sparse_and_dense_reconstruction(sparse):
    import sklearn
    rng = np.random.default_rng(7)
    values = rng.normal(size=(40, 3)) @ rng.normal(size=(3, 20))
    operator = matrixOperator(csr_matrix(values) if sparse else values, 'scipy')
    algo = algorithmRegistry.get(category='svd', variant='randomized', backendName='scipy')
    first = algo.decompose(operator, rank=3, oversampling=100, randomState=7)
    second = algo.decompose(operator, rank=3, oversampling=100, randomState=7)
    np.testing.assert_allclose((first[0] * first[1]) @ first[2], values, atol=1e-11)
    for a, b in zip(first, second): np.testing.assert_array_equal(a, b)
    assert algo.decompose(operator, rank=0)[1].size == 0


@pytest.mark.parametrize('options', [{'rank': -1}, {'rank': 1.5}, {'rank': 2, 'iterations': -1}, {'rank': 2, 'oversampling': True}])
def test_randomized_svd_rejects_invalid_options(options):
    algo = algorithmRegistry.get(category='svd', variant='randomized', backendName='scipy')
    with pytest.raises(ValueError, match='non-negative integer'):
        algo.decompose(matrixOperator(np.eye(4), 'scipy'), **options)


def test_randomized_svd_power_iterations_improve_wide_matrix_accuracy():
    import sklearn
    rng = np.random.default_rng(4)
    u = np.linalg.qr(rng.normal(size=(32, 32)))[0]
    v = np.linalg.qr(rng.normal(size=(96, 32)))[0]
    spectrum = 1. / np.arange(1, 33) ** 1.2
    data = (u * spectrum) @ v.T
    algo = algorithmRegistry.get('svd', 'randomized', backendName='scipy')
    errors = []
    for iterations in [0, 4]:
        U, s, V = algo.decompose(matrixOperator(data, 'scipy'), rank=6, oversampling=4, iterations=iterations)
        errors.append(np.linalg.norm(data - (U * s) @ V))
    optimal = np.linalg.norm(spectrum[6:])
    assert errors[1] <= errors[0]
    assert errors[1] <= optimal * 1.01
