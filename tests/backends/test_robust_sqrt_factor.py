import warnings

import numpy as np
import pytest

from mor.backends.scipyBackend import scipyBackend


def assert_no_runtime_warning(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]
    return result


def test_scipy_rank_deficient_psd_truncation_is_silent():
    matrix = np.diag([4.0, 1.0, 0.0, 0.0])

    factor = assert_no_runtime_warning(lambda: scipyBackend.linalg.robustSqrtFactor(matrix))

    assert factor.shape == (4, 2)
    np.testing.assert_allclose(factor @ factor.T, matrix)


def test_scipy_roundoff_negative_eigenvalue_is_silent():
    eps = np.finfo(float).eps
    matrix = np.diag([1.0, -eps])

    factor = assert_no_runtime_warning(lambda: scipyBackend.linalg.robustSqrtFactor(matrix))

    np.testing.assert_allclose(factor @ factor.T, np.diag([1.0, 0.0]))


def test_scipy_significant_negative_eigenvalue_warns():
    matrix = np.diag([1.0, -1e-3])

    with pytest.warns(RuntimeWarning, match="not positive semidefinite"):
        factor = scipyBackend.linalg.robustSqrtFactor(matrix, name="Gramian")

    np.testing.assert_allclose(factor @ factor.T, np.diag([1.0, 0.0]))


def test_scipy_excessive_explicit_tolerance_warns():
    matrix = np.diag([1.0, 0.1])

    with pytest.warns(RuntimeWarning, match="positive spectral weight"):
        factor = scipyBackend.linalg.robustSqrtFactor(matrix, tol=0.2)

    np.testing.assert_allclose(factor @ factor.T, np.diag([1.0, 0.0]))


def test_scipy_negative_tolerance_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        scipyBackend.linalg.robustSqrtFactor(np.eye(2), tol=-1.0)


@pytest.mark.parametrize(
    ("diagonal", "tol", "warningMatch"),
    [
        ([4.0, 1.0, 0.0, 0.0], None, None),
        ([1.0, -np.finfo(float).eps], None, None),
        ([1.0, -1e-3], None, "not positive semidefinite"),
        ([1.0, 0.1], 0.2, "positive spectral weight"),
    ],
)
def test_torch_robust_sqrt_factor_diagnostics(diagonal, tol, warningMatch):
    torch = pytest.importorskip("torch")
    from mor.backends.torchBackend import torchBackend

    matrix = torch.diag(torch.tensor(diagonal, dtype=torch.float64))
    call = lambda: torchBackend.linalg.robustSqrtFactor(matrix, tol=tol)

    if warningMatch is None:
        factor = assert_no_runtime_warning(call)
    else:
        with pytest.warns(RuntimeWarning, match=warningMatch):
            factor = call()

    expected = torch.diag(torch.clamp(torch.diag(matrix), min=0))
    if tol is not None:
        expected = torch.diag(torch.where(torch.diag(matrix) > tol, torch.diag(matrix), 0.0))
    torch.testing.assert_close(factor @ factor.mT, expected)
