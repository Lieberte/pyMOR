import torch
import numpy as np
import pytest
from scipy import linalg
from scipy.sparse import csr_matrix

from mor.algorithm.registry import algorithmRegistry
from mor.operators import matrixOperator


@pytest.mark.parametrize('backendName,sparse', [('scipy', False), ('scipy', True), ('torch', False)])
@pytest.mark.parametrize('trans', [False, True])
def test_lradi_reuses_shifts_and_discards_factors_between_solves(backendName, trans, sparse, monkeypatch):
    import scipy.sparse.linalg
    a = np.array([[-1., 0.2], [0., -2.]])
    e = np.array([[1., 0.1], [0., 1.5]])
    b = np.array([[1.], [2.]])
    convert = (lambda value: torch.from_numpy(value)) if backendName == 'torch' else (csr_matrix if sparse else np.asarray)
    algo = algorithmRegistry.get('lyapunov', 'lradi', backendName=backendName, tol=1e-12,
                                 maxIter=100, shiftOptions={'maxShifts': 2})
    shifts = torch.tensor([-1., -2.], dtype=torch.float64) if backendName == 'torch' else np.array([-1., -2.])
    monkeypatch.setattr(algo, 'computeInitialShifts', lambda *args: shifts)
    monkeypatch.setattr(algo, 'updateShifts', lambda *args, **kwargs: shifts)
    module, name = (torch.linalg, 'lu_factor') if backendName == 'torch' else ((scipy.sparse.linalg, 'splu') if sparse else (linalg, 'lu_factor'))
    original, calls = getattr(module, name), []
    def factor(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, name, factor)
    for scale in [1., 2.]:
        result = algo.solve(matrixOperator(convert(a * scale), backendName),
                            matrixOperator(convert(e), backendName),
                            matrixOperator(torch.from_numpy(b.T if trans else b) if backendName == 'torch' else (b.T if trans else b), backendName), trans=trans)
        x = result.toBackendData()
        if backendName == 'torch': x = x.cpu().numpy()
        at, et = (a.T * scale, e.T) if trans else (a * scale, e)
        effectiveA, effectiveB = np.linalg.solve(et, at), np.linalg.solve(et, b)
        expected = linalg.solve_continuous_lyapunov(effectiveA, -effectiveB @ effectiveB.T)
        np.testing.assert_allclose(x, expected, atol=1e-9)
        assert algo.lastConverged
    assert len(calls) == 4


@pytest.mark.parametrize('backendName', ['scipy', 'torch'])
@pytest.mark.parametrize('trans', [False, True])
def test_factorized_complex_shift_preserves_plain_transpose_and_rhs_shape(backendName, trans):
    data = np.array([[-1., 0.2], [0., -2.]])
    rhs = np.ones((2, 1))
    convert = torch.from_numpy if backendName == 'torch' else np.asarray
    solver = algorithmRegistry.get('linear', 'shifted', backendName=backendName)
    solve = solver.factorize(matrixOperator(convert(data), backendName), shift=-1 + 0.5j, trans=trans)
    result = solve(convert(rhs))
    if backendName == 'torch': result = result.numpy()
    matrix = data + (-1 + 0.5j) * np.eye(2)
    np.testing.assert_allclose(result, np.linalg.solve(matrix.T if trans else matrix, rhs))
    assert result.shape == rhs.shape


@pytest.mark.parametrize('trans', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_torch_descriptor_factor_reused_and_matches_discrete_reference(trans, device, monkeypatch):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    a = np.array([[0.3, 0.1], [0., 0.4]])
    e = np.array([[1., 0.2], [0., 2.]])
    b = np.ones((2, 1))
    def op(value): return matrixOperator(torch.from_numpy(value).to(device), 'torch')
    smith = algorithmRegistry.get('lyapunov', 'lrsmith', backendName='torch', maxIter=80, tol=1e-12, checkStability=False)
    original, calls = torch.linalg.lu_factor, []
    def factor(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch.linalg, 'lu_factor', factor)
    for scale in [1., 2.]:
        result = smith.solve(op(a), op(e * scale), op(b.T if trans else b), trans=trans)
        et, at = (e.T * scale, a.T) if trans else (e * scale, a)
        aa, bb = np.linalg.solve(et, at), np.linalg.solve(et, b)
        expected = linalg.solve_discrete_lyapunov(aa, bb @ bb.T)
        np.testing.assert_allclose(result.toBackendData().cpu().numpy(), expected, atol=1e-10)
    assert len(calls) == 2


def test_lradi_can_disable_factor_cache_and_reject_invalid_budget(monkeypatch):
    from mor.algorithm.linear.shifted import shiftedLinear
    original, calls = shiftedLinear.factorize, []
    def factorize(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(shiftedLinear, 'factorize', factorize)
    algo = algorithmRegistry.get('lyapunov', 'lradi', backendName='scipy', maxIter=4, tol=0., factorCacheBytes=0)
    shifts = np.array([-1.])
    monkeypatch.setattr(algo, 'computeInitialShifts', lambda *args: shifts)
    monkeypatch.setattr(algo, 'updateShifts', lambda *args, **kwargs: shifts)
    a, b = matrixOperator(np.diag([-2., -3.]), 'scipy'), matrixOperator(np.ones((2, 1)), 'scipy')
    algo.solve(a, None, b)
    assert len(calls) == 4
    for budget in [-1, True, 1.5]:
        algo.options['factorCacheBytes'] = budget
        with pytest.raises(ValueError, match='factorCacheBytes'): algo.solve(a, None, b)


@pytest.mark.parametrize('sparse', [False, True])
def test_scipy_factorized_real_float32_matrix_accepts_complex_float64_rhs(sparse):
    data = np.array([[2., 0.1], [0., 3.]], dtype=np.float32)
    rhs = np.array([[1. + 2j], [3. - 1j]])
    solver = algorithmRegistry.get('linear', 'shifted', backendName='scipy')
    actual = solver.factorize(matrixOperator(csr_matrix(data) if sparse else data, 'scipy'))(rhs)
    np.testing.assert_allclose(actual, np.linalg.solve(data.astype(float), rhs), atol=1e-12)
