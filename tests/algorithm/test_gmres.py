import numpy as np
import pytest
from scipy.sparse import diags

from mor.algorithm.registry import algorithmRegistry
from mor.operators import matrixOperator, lowRankOperator


@pytest.mark.parametrize('trans', [False, True])
@pytest.mark.parametrize('shift', [-1., -1. + 0.2j])
@pytest.mark.parametrize('inputs', [1, 3])
def test_gmres_composed_descriptor_matches_direct_without_materializing(trans, shift, inputs, monkeypatch):
    rng = np.random.default_rng(5)
    left = rng.normal(size=(20, 2)) * 0.01
    a = matrixOperator(diags(-np.linspace(1., 2., 20)), 'scipy') + lowRankOperator(left, backendName='scipy')
    e = matrixOperator(diags(np.linspace(1., 2., 20)), 'scipy')
    rhs = rng.normal(size=(20, inputs))
    full = a.toBackendData() + shift * e.toBackendData()
    full = full.T if trans else full
    expected = np.linalg.solve(full.toarray() if hasattr(full, 'toarray') else full, rhs)
    monkeypatch.setattr(a, 'toBackendData', lambda: pytest.fail('matrix-free solve materialized A'))
    solver = algorithmRegistry.get('linear', 'auto', backendName='scipy', matrixFree=True)
    actual = solver.solve(a, rhs, E=e, shift=shift, trans=trans)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
    assert actual.shape == rhs.shape and solver.lastConverged


def test_gmres_preserves_vector_rhs_and_exposes_failure():
    a = matrixOperator(np.diag([1., 2., 3.]), 'scipy')
    solver = algorithmRegistry.get('linear', 'gmres', backendName='scipy')
    np.testing.assert_allclose(solver.solve(a, np.ones(3)), [1., 0.5, 1 / 3])
    with pytest.raises(ValueError, match='converge'):
        solver.solve(a, np.ones(3), restart=1, maxIter=1, tol=1e-15)
    assert not solver.lastConverged


@pytest.mark.parametrize('options', [{'tol': np.nan}, {'maxIter': 0}, {'restart': True}])
def test_gmres_rejects_invalid_budget(options):
    solver = algorithmRegistry.get('linear', 'gmres', backendName='scipy')
    with pytest.raises(ValueError): solver.solve(matrixOperator(np.eye(2), 'scipy'), np.ones(2), **options)
