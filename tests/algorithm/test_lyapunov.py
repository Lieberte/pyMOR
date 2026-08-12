"""Lyapunov contract tests.

Covers:
  * lrsmith.solve must accept a ``trans`` keyword (the unified solver always
    passes one) -- it previously raised TypeError.
  * dense + discrete auto-dispatch must raise a clear NotImplementedError --
    it previously returned the unregistered name 'discreteHr', producing a
    confusing "Algorithm not registered" error deep in the registry.

Note: lrsmith cannot be *executed* end-to-end yet because it calls several
backend array methods (max/ndim/reshape/...) that do not exist -- a separate,
tracked gap. These tests therefore check the contract at the signature level.
"""
import inspect

import numpy as np
import pytest
from scipy.sparse import diags

from mor.algorithm.registry import algorithmRegistry
from mor.algorithm.auto.lyapunov import recommendLyapunovOptions, selectLyapunovAlgorithm
from mor.operators import lowRankOperator, matrixOperator
from scripts.standard_benchmark_cases import matlab_gallery_grcar


class _FakeA:
    isSparse = False


def test_lrsmith_solve_has_trans_parameter():
    smith = algorithmRegistry.get(category='lyapunov', variant='lrsmith', backendName='scipy')
    params = inspect.signature(smith.solve).parameters
    assert 'trans' in params, "lrsmith.solve must accept trans= (unified solver passes it)"
    assert params['trans'].default is False


def test_dense_discrete_dispatch_raises_not_implemented():
    """Dense discrete Lyapunov has no implementation; fail loudly instead of returning a phantom name."""
    with pytest.raises(NotImplementedError, match='discrete'):
        selectLyapunovAlgorithm(variant='auto', A=_FakeA(), backendName='scipy', isContinuous=False)


def test_explicit_variant_is_passed_through():
    """A non-auto variant must be honored (not routed into the discrete dead-end)."""
    assert selectLyapunovAlgorithm(variant='bartelsStewart', A=_FakeA(), backendName='scipy') == 'bartelsStewart'


def test_dense_continuous_still_dispatches_to_bartels_stewart():
    assert selectLyapunovAlgorithm(variant='auto', A=_FakeA(), backendName='scipy', isContinuous=True) == 'bartelsStewart'


def test_dense_continuous_torch_dispatches_to_sign():
    assert selectLyapunovAlgorithm(variant='auto', A=_FakeA(), backendName='torch', isContinuous=True) == 'sign'


def test_small_sparse_continuous_dispatches_to_dense_direct_solver():
    A = matrixOperator(diags([-1.0, -2.0, -3.0], format='csr'), backendName='scipy')

    assert selectLyapunovAlgorithm(variant='auto', A=A, backendName='scipy', isContinuous=True) == 'bartelsStewart'


def test_large_sparse_continuous_dispatches_to_lradi():
    A = matrixOperator(diags(-np.ones(1000), format='csr'), backendName='scipy')

    assert selectLyapunovAlgorithm(variant='auto', A=A, backendName='scipy', isContinuous=True) == 'lradi'


def test_moderate_high_density_sparse_continuous_dispatches_to_dense_direct_solver():
    A = matrixOperator(diags(-np.ones(300), format='csr') + np.ones((300, 300)) * 1e-6, backendName='scipy')

    assert selectLyapunovAlgorithm(variant='auto', A=A, backendName='scipy', isContinuous=True) == 'bartelsStewart'


def test_sparse_discrete_dispatches_to_lrsmith():
    A = matrixOperator(diags([0.2, 0.4, 0.6], format='csr'), backendName='scipy')

    assert selectLyapunovAlgorithm(variant='auto', A=A, backendName='scipy', isContinuous=False) == 'lrsmith'


def test_sparse_auto_dispatch_rejects_non_scipy_backend():
    A = matrixOperator(diags([-1.0, -2.0, -3.0], format='csr'), backendName='scipy')

    with pytest.raises(NotImplementedError, match='scipy'):
        selectLyapunovAlgorithm(variant='auto', A=A, backendName='torch', isContinuous=True)


def test_small_sparse_descriptor_dispatches_to_dense_direct_solver():
    A = matrixOperator(-np.eye(3), backendName='scipy')
    E = matrixOperator(diags([1.0, 1.0, 1.0], format='csr'), backendName='scipy')

    assert selectLyapunovAlgorithm(variant='auto', A=A, E=E, backendName='scipy', isContinuous=True) == 'bartelsStewart'


def test_bartels_stewart_promotes_unsigned_rhs_before_forming_q():
    A = matrixOperator(diags([-1.0, -2.0, -3.0], format='csr'), backendName='scipy')
    B = matrixOperator(np.array([[1, 0, 0]], dtype=np.uint8), backendName='scipy')
    algo = algorithmRegistry.get(category='lyapunov', variant='bartelsStewart', backendName='scipy')

    X = algo.solve(A, None, B, trans=True).toBackendData()
    qData = B.toBackendData().T.astype(float) @ B.toBackendData().astype(float)
    aData = A.toBackendData().toarray().T
    residual = aData @ X + X @ aData.T + qData

    assert np.linalg.norm(residual) / np.linalg.norm(qData) < 1e-12
    assert np.linalg.eigvalsh(X).min() >= 0


def test_bartels_stewart_generalized_residual_is_refined_in_original_coordinates():
    E = np.diag([1e-8, 3e-7, 2e-5, 1e-3])
    A = -np.diag([2.0, 3.0, 4.0, 5.0]) @ E
    B = np.ones((4, 1))
    aOperator = matrixOperator(A, backendName='scipy')
    eOperator = matrixOperator(E, backendName='scipy')
    bOperator = matrixOperator(B, backendName='scipy')
    algo = algorithmRegistry.get(category='lyapunov', variant='bartelsStewart', backendName='scipy')

    X = algo.solve(aOperator, eOperator, bOperator).toBackendData()
    qData = B @ B.T
    residual = A @ X @ E.T + E @ X @ A.T + qData

    assert np.linalg.norm(residual, 2) / np.linalg.norm(qData, 2) < 1e-10


def test_lradi_option_recommendation_increases_budget_for_large_sparse_case():
    A = matrixOperator(diags(-np.ones(10000), format='csr'), backendName='scipy')
    B = matrixOperator(np.ones((10000, 2)), backendName='scipy')

    options = recommendLyapunovOptions(variant='auto', A=A, B=B, backendName='scipy', isContinuous=True)

    assert options['maxIter'] >= 240
    assert options['compressionColumns'] >= 60
    assert options['shiftOptions']['maxShifts'] >= 12


def test_lradi_option_recommendation_increases_budget_for_many_inputs():
    A = matrixOperator(diags(-np.ones(1000), format='csr'), backendName='scipy')
    B = matrixOperator(np.ones((1000, 4)), backendName='scipy')

    options = recommendLyapunovOptions(variant='auto', A=A, B=B, backendName='scipy', isContinuous=True)

    assert options['maxIter'] >= 240
    assert options['compressionColumns'] >= 80


def test_lrsmith_option_recommendation_flags_near_unit_discrete_case():
    A = matrixOperator(diags(np.linspace(0.1, 0.995, 80), format='csr'), backendName='scipy')
    B = matrixOperator(np.ones((80, 1)), backendName='scipy')

    options = recommendLyapunovOptions(variant='auto', A=A, B=B, backendName='scipy', isContinuous=False)

    assert options['risk'] == 'near_unit_slow_convergence'
    assert options['stability'] == 'stable'
    assert options['spectralRadiusEstimate'] == pytest.approx(0.995)
    assert options['maxIter'] >= 600


def test_lrsmith_option_recommendation_flags_unstable_discrete_case():
    A = matrixOperator(diags([0.8, 1.01, 1.05], format='csr'), backendName='scipy')
    B = matrixOperator(np.ones((3, 1)), backendName='scipy')

    options = recommendLyapunovOptions(variant='auto', A=A, B=B, backendName='scipy', isContinuous=False)

    assert options['risk'] == 'reject_unstable_discrete_operator'
    assert options['stability'] == 'unstable'
    assert options['spectralRadiusEstimate'] == pytest.approx(1.05)


def test_lrsmith_option_recommendation_uses_large_case_residual_floor():
    A = matrixOperator(diags(0.2 * np.ones(1000), format='csr'), backendName='scipy')
    B = matrixOperator(np.ones((1000, 2)), backendName='scipy')

    options = recommendLyapunovOptions(variant='auto', A=A, B=B, backendName='scipy', isContinuous=False)

    assert options['risk'] == 'standard'
    assert options['acceptanceResidual'] >= 1e-7
    assert options['maxIter'] >= 220


def test_lradi_sparse_continuous_residual_and_stats():
    n = 6
    A = matrixOperator(diags(-np.linspace(1.0, 6.0, n), format='csr'), backendName='scipy')
    B = matrixOperator(np.ones((n, 1)), backendName='scipy')
    algo = algorithmRegistry.get(
        category='lyapunov',
        variant='lradi',
        backendName='scipy',
        tol=1e-10,
        maxIter=80,
        compressionColumns=8,
        shiftOptions={'maxShifts': 6},
    )

    Z = algo.solve(A, None, B)
    X = Z.toBackendData()
    aData = A.toBackendData()
    qData = B.toBackendData() @ B.toBackendData().T
    residual = aData @ X + X @ aData.T + qData
    relative_residual = np.linalg.norm(residual) / np.linalg.norm(qData)

    assert isinstance(Z, lowRankOperator)
    assert Z.left.shape[0] == n
    assert algo.lastIterations > 0
    assert len(algo.lastResidualHistory) == algo.lastIterations + 1
    assert relative_residual < 1e-8


def test_bartels_stewart_and_sign_dense_continuous_standard_case_residuals():
    G = matlab_gallery_grcar(8)
    eig_radius = max(abs(np.linalg.eigvals(G)))
    aData = G / (2.0 * eig_radius) - 1.5 * np.eye(8)
    A = matrixOperator(aData, backendName='scipy')
    B = matrixOperator(np.ones((8, 1)), backendName='scipy')
    qData = B.toBackendData() @ B.toBackendData().T

    for variant in ('bartelsStewart', 'sign'):
        algo = algorithmRegistry.get(category='lyapunov', variant=variant, backendName='scipy', maxIter=80)
        Z = algo.solve(A, None, B)
        xData = Z.toBackendData()
        residual = aData @ xData + xData @ aData.T + qData
        assert np.linalg.norm(residual) / np.linalg.norm(qData) < 1e-10


def test_sign_accepts_transpose_contract():
    A = matrixOperator(-np.eye(3), backendName='scipy')
    B = matrixOperator(np.ones((2, 3)), backendName='scipy')
    algo = algorithmRegistry.get(category='lyapunov', variant='sign', backendName='scipy', maxIter=2)

    Z = algo.solve(A, None, B, trans=True)

    assert isinstance(Z, lowRankOperator)
    assert Z.left.shape[0] == 3
