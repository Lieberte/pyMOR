import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mor.backends import backendRegistry


@pytest.mark.parametrize('scale', [1e-200, 1e200])
def test_default_dense_norm_preserves_extreme_finite_magnitudes(scale):
    values = np.ones((2, 3)) * scale
    result = backendRegistry.get('scipy').linalg.norm(values)
    assert result / scale == pytest.approx(np.sqrt(6.))


def test_dense_norm_orders_and_sparse_default_match_reference():
    values = np.array([[1., 2.], [3., 4.]])
    backend = backendRegistry.get('scipy')
    for order in [None, 'fro', 1, 2, np.inf]:
        assert backend.linalg.norm(values, ord=order) == pytest.approx(np.linalg.norm(values, ord=order))
    assert backend.linalg.norm(csr_matrix(values)) == pytest.approx(np.linalg.norm(values))
