import torch
import numpy as np
import pytest

from mor.algorithm.registry import algorithmRegistry
from mor.operators import matrixOperator


@pytest.mark.parametrize('backendName', ['scipy', 'torch'])
@pytest.mark.parametrize('complexData', [False, True])
def test_incremental_core_matches_truncated_history_and_preserves_orthogonality(backendName, complexData):
    rng = np.random.default_rng(7)
    algo = algorithmRegistry.get(category='svd', variant='incrementalStandard', backendName=backendName)
    retained = np.empty((12, 0))
    for count in [5, 4, 3]:
        values = rng.normal(size=(12, count))
        if complexData: values = values + 1j * rng.normal(size=values.shape)
        target = np.hstack([retained, values])
        inputData = torch.from_numpy(values) if backendName == 'torch' else values
        U, S, Vh = algo.decompose(matrixOperator(inputData, backendName), rank=3)
        if backendName == 'torch': U, S, Vh = U.numpy(), S.numpy(), Vh.numpy()
        u, s, vh = np.linalg.svd(target, full_matrices=False)
        retained = (U * S) @ Vh
        np.testing.assert_allclose(retained, (u[:, :3] * s[:3]) @ vh[:3], atol=1e-11)
        np.testing.assert_allclose(U.conj().T @ U, np.eye(3), atol=1e-11)
        np.testing.assert_allclose(Vh @ Vh.conj().T, np.eye(3), atol=1e-11)
    with pytest.raises(ValueError, match='row count'):
        algo.decompose(matrixOperator(inputData[:5], backendName), rank=3)
    algo.reset()
    U, S, Vh = algo.decompose(matrixOperator(inputData[:5], backendName), rank=3)
    assert U.shape == (5, 3)


def test_incremental_update_avoids_full_history_svd(monkeypatch):
    rng = np.random.default_rng(7)
    algo = algorithmRegistry.get(category='svd', variant='incrementalStandard', backendName='scipy')
    algo.decompose(matrixOperator(rng.normal(size=(40, 100)), 'scipy'), rank=4)
    shapes = []
    original = algo.localBackend.decomposition.svdDense
    def svd(values, **kwargs):
        shapes.append(values.shape)
        return original(values, **kwargs)
    monkeypatch.setattr(algo.localBackend.decomposition, 'svdDense', svd)
    algo.decompose(matrixOperator(rng.normal(size=(40, 3)), 'scipy'), rank=4)
    assert (40, 103) not in shapes
    assert shapes == [(40, 3), (7, 7)]


@pytest.mark.parametrize('variant', ['incrementalStandard', 'incrementalBrand', 'incrementalFast'])
def test_zero_rank_stream_can_receive_new_information(variant):
    algo = algorithmRegistry.get(category='svd', variant=variant, backendName='scipy')
    algo.decompose(matrixOperator(np.ones((4, 2)), 'scipy'), rank=0)
    U, S, Vh = algo.decompose(matrixOperator(np.eye(4, 2), 'scipy'), rank=2)
    np.testing.assert_allclose((U * S) @ Vh, np.hstack([np.zeros((4, 2)), np.eye(4, 2)]), atol=1e-12)
