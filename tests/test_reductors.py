"""End-to-end regression smoke for the classical-MOR primary path (scipy).

Guards against the operator/SVD contract changes breaking POD or balanced
truncation. These were validated manually before; locking them in as tests.
"""
import numpy as np
import pytest

from mor.operators import matrixOperator
from mor.models.lti import ltiModel
from mor.reductors.pod import podReductor
from mor.reductors.balancedTruncation import balancedTruncationReductor
from mor.reductors.frequencyResponse import frequencyResponseReductor


@pytest.fixture
def rng():
    return np.random.default_rng(3)


def test_pod_end_to_end(rng):
    snap = matrixOperator(rng.standard_normal((10, 6)), backendName='scipy')
    result = podReductor(globalBackendName='scipy').reduce(snap, rank=3)
    assert result.order == 3
    ref = np.linalg.svd(snap.toBackendData(), compute_uv=False)
    np.testing.assert_allclose(np.sort(result.S)[::-1], np.sort(ref[:3])[::-1])
    assert result.solverInfo['podSolver'] == 'snapshotPodSolver'
    assert result.basis is result.V
    assert result.reconstructionError is not None
    assert 0 <= result.reconstructionError <= 1


def test_weighted_pod_passes_product_to_dual_svd(rng):
    snapshots = matrixOperator(rng.standard_normal((10, 6)), backendName='scipy')
    product = matrixOperator(np.diag(np.linspace(1.0, 2.0, 10)), backendName='scipy')

    result = podReductor(globalBackendName='scipy', product=product).reduce(snapshots, rank=3)

    weightedGram = result.V.toNumpy().T @ product.toNumpy() @ result.V.toNumpy()
    np.testing.assert_allclose(weightedGram, np.eye(3), atol=1e-10)
    assert result.solverInfo['svd'] == 'dualSVD'
    assert 0 <= result.reconstructionError <= 1


def test_centered_pod_preserves_mean_for_reconstruction():
    snapshotsData = np.array([[10.0, 11.0, 9.0], [20.0, 22.0, 18.0]])
    snapshots = matrixOperator(snapshotsData, backendName='scipy')
    reductor = podReductor(globalBackendName='scipy', centerSnapshots=True)

    result = reductor.reduce(snapshots, rank=1)
    centered = snapshotsData - snapshotsData.mean(axis=1, keepdims=True)
    coefficients = result.V.toNumpy().T @ centered
    reconstructed = reductor.reconstruct(result, coefficients)

    np.testing.assert_allclose(result.snapshotMean, snapshotsData.mean(axis=1))
    np.testing.assert_allclose(reconstructed.toNumpy(), snapshotsData, atol=1e-10)


def test_centered_weighted_pod_combines_mean_and_product(rng):
    snapshotsData = rng.standard_normal((8, 5)) + np.arange(8)[:, None]
    productData = np.diag(np.linspace(1.0, 2.0, 8))
    product = matrixOperator(productData, backendName='scipy')

    result = podReductor(globalBackendName='scipy', centerSnapshots=True, product=product).reduce(
        matrixOperator(snapshotsData, backendName='scipy'), rank=3
    )

    np.testing.assert_allclose(result.snapshotMean, snapshotsData.mean(axis=1))
    np.testing.assert_allclose(result.V.toNumpy().T @ productData @ result.V.toNumpy(), np.eye(3), atol=1e-10)
    assert result.solverInfo['svd'] == 'dualSVD'


def test_balanced_truncation_end_to_end(rng):
    n = 12
    A = -np.eye(n) + 0.1 * rng.standard_normal((n, n))
    A = (A + A.T) / 2  # symmetric -> real, stable spectrum for the Lyapunov solve
    lti = ltiModel(
        matrixOperator(A, backendName='scipy'),
        matrixOperator(rng.standard_normal((n, 3)), backendName='scipy'),
        matrixOperator(rng.standard_normal((2, n)), backendName='scipy'),
        backendName='scipy',
    )
    rom = balancedTruncationReductor(globalBackendName='scipy').reduce(lti, order=5)
    assert rom.order == 5
    assert len(rom.hsv) == 5
    assert all(h > 0 for h in rom.hsv)
    assert rom.solverInfo['lyapunov'] == 'bartelsStewart'


def test_balanced_truncation_accepts_composed_operators(rng):
    n = 8
    baseA = -np.eye(n) + 0.05 * rng.standard_normal((n, n))
    baseA = (baseA + baseA.T) / 2
    A0 = matrixOperator(baseA, backendName='scipy')
    B0 = matrixOperator(rng.standard_normal((n, n)), backendName='scipy')
    C0 = matrixOperator(rng.standard_normal((n, n)), backendName='scipy')
    lti = ltiModel(
        A0 + (A0 * 0.0),
        B0 + (B0 * 0.0),
        C0 + (C0 * 0.0),
        backendName='scipy',
    )

    rom = balancedTruncationReductor(globalBackendName='scipy').reduce(lti, order=3)

    assert rom.order == 3
    assert rom.Ar.shape == (3, 3)


def test_balanced_truncation_result_exposes_lti_model(rng):
    n = 6
    model = ltiModel(
        matrixOperator(-np.diag(np.linspace(1.0, 2.0, n)), backendName='scipy'),
        matrixOperator(rng.standard_normal((n, 2)), backendName='scipy'),
        matrixOperator(rng.standard_normal((3, n)), backendName='scipy'),
        backendName='scipy',
    )

    rom = balancedTruncationReductor(globalBackendName='scipy').reduce(model, order=2)

    assert isinstance(rom.model, ltiModel)
    assert rom.model.order == 2
    assert rom.model.inputDim == 2
    assert rom.model.outputDim == 3
    assert rom.model.isContinuous


def test_balanced_truncation_uses_model_time_domain_by_default(rng):
    n = 5
    model = ltiModel(
        matrixOperator(0.3 * np.eye(n), backendName='scipy'),
        matrixOperator(rng.standard_normal((n, 1)), backendName='scipy'),
        matrixOperator(rng.standard_normal((1, n)), backendName='scipy'),
        backendName='scipy',
        isContinuous=False,
    )

    with pytest.raises(NotImplementedError, match='Dense discrete'):
        balancedTruncationReductor(globalBackendName='scipy').reduce(model, order=2)


def test_sparse_discrete_balanced_truncation_uses_lrsmith(rng):
    from scipy.sparse import diags

    n = 6
    model = ltiModel(
        matrixOperator(diags(np.linspace(0.1, 0.6, n), format='csr'), backendName='scipy'),
        matrixOperator(rng.standard_normal((n, 1)), backendName='scipy'),
        matrixOperator(rng.standard_normal((1, n)), backendName='scipy'),
        backendName='scipy',
        isContinuous=False,
    )

    rom = balancedTruncationReductor(globalBackendName='scipy').reduce(model, order=2)

    assert not rom.isContinuous
    assert not rom.model.isContinuous
    assert rom.solverInfo['lyapunov'] == 'lrsmith'


def test_balanced_truncation_rejects_unstable_continuous_model():
    model = ltiModel(
        matrixOperator(np.diag([0.2, -1.0]), backendName='scipy'),
        matrixOperator(np.ones((2, 1)), backendName='scipy'),
        matrixOperator(np.ones((1, 2)), backendName='scipy'),
        backendName='scipy',
    )

    with pytest.raises(ValueError, match='stable continuous'):
        balancedTruncationReductor(globalBackendName='scipy').reduce(model, order=1)


@pytest.mark.parametrize('operator_factory', [
    lambda unstable, zero: unstable + zero,
    lambda unstable, zero: 1.0 * (unstable + zero),
])
def test_balanced_truncation_rejects_unstable_composed_operator(operator_factory):
    unstable = matrixOperator(np.diag([0.2, -1.0]), backendName='scipy')
    zero = matrixOperator(np.zeros((2, 2)), backendName='scipy')
    model = ltiModel(
        operator_factory(unstable, zero),
        matrixOperator(np.ones((2, 1)), backendName='scipy'),
        matrixOperator(np.ones((1, 2)), backendName='scipy'),
        backendName='scipy',
    )

    with pytest.raises(ValueError, match='stable continuous'):
        balancedTruncationReductor(globalBackendName='scipy').reduce(model, order=1)


def test_balanced_truncation_checks_large_composed_diagonal_without_materializing(monkeypatch):
    from scipy.sparse import diags

    order = 1_000
    unstable = matrixOperator(diags([-1.0] * (order - 1) + [0.2], format='csr'), backendName='scipy')
    A = unstable + 0.0 * unstable
    model = ltiModel(
        A,
        matrixOperator(np.ones((order, 1)), backendName='scipy'),
        matrixOperator(np.ones((1, order)), backendName='scipy'),
        backendName='scipy',
    )
    monkeypatch.setattr(A, 'toNumpy', lambda: (_ for _ in ()).throw(AssertionError('materialized A')))

    with pytest.raises(ValueError, match='stable continuous'):
        balancedTruncationReductor(globalBackendName='scipy')._validateStability(model, True)


def test_balanced_truncation_preserves_symmetric_unstable_modes():
    A = np.diag([0.2, -1.0, -2.0, -3.0])
    model = ltiModel(
        matrixOperator(A, backendName='scipy'),
        matrixOperator(np.ones((4, 1)), backendName='scipy'),
        matrixOperator(np.ones((1, 4)), backendName='scipy'),
        backendName='scipy',
    )

    rom = balancedTruncationReductor(
        globalBackendName='scipy', unstableHandling='preserve'
    ).reduce(model, order=3)

    eigenvalues = np.linalg.eigvals(rom.Ar.toNumpy())
    assert rom.order == 3
    assert rom.solverInfo['preservedUnstableModes'] == 1
    assert np.count_nonzero(eigenvalues.real >= 0) == 1
    assert eigenvalues.real.max() == pytest.approx(0.2)


def test_balanced_truncation_rejects_zero_hankel_singular_values():
    model = ltiModel(
        matrixOperator(-np.diag([1.0, 2.0]), backendName='scipy'),
        matrixOperator(np.array([[1.0], [0.0]]), backendName='scipy'),
        matrixOperator(np.array([[0.0, 1.0]]), backendName='scipy'),
        backendName='scipy',
    )

    with pytest.raises(ValueError, match='positive numerical Hankel'):
        balancedTruncationReductor(globalBackendName='scipy').reduce(model, order=1)


def test_descriptor_balanced_truncation_uses_e_weighted_cross_gramian(rng):
    n = 5
    model = ltiModel(
        matrixOperator(-np.diag(np.linspace(1.0, 2.0, n)), backendName='scipy'),
        matrixOperator(rng.standard_normal((n, 2)), backendName='scipy'),
        matrixOperator(rng.standard_normal((2, n)), backendName='scipy'),
        E=matrixOperator(np.diag(np.linspace(1.0, 3.0, n)), backendName='scipy'),
        backendName='scipy',
    )

    rom = balancedTruncationReductor(globalBackendName='scipy').reduce(model, order=2)

    np.testing.assert_allclose(rom.Er.toNumpy(), np.eye(2), atol=1e-10)
    assert rom.solverInfo['descriptorKind'] == 'regular'


def test_balanced_truncation_rejects_singular_descriptor():
    model = ltiModel(
        matrixOperator(-np.eye(3), backendName='scipy'),
        matrixOperator(np.ones((3, 1)), backendName='scipy'),
        matrixOperator(np.ones((1, 3)), backendName='scipy'),
        E=matrixOperator(np.diag([1.0, 1.0, 0.0]), backendName='scipy'),
        backendName='scipy',
    )

    with pytest.raises(NotImplementedError, match='spectral projectors'):
        balancedTruncationReductor(globalBackendName='scipy').reduce(model, order=2)


def test_balanced_truncation_rejects_singular_composed_descriptor():
    singular = matrixOperator(np.diag([1.0, 1.0, 0.0]), backendName='scipy')
    zero = matrixOperator(np.zeros((3, 3)), backendName='scipy')
    model = ltiModel(
        matrixOperator(-np.eye(3), backendName='scipy'),
        matrixOperator(np.ones((3, 1)), backendName='scipy'),
        matrixOperator(np.ones((1, 3)), backendName='scipy'),
        E=singular + zero,
        backendName='scipy',
    )

    with pytest.raises(NotImplementedError, match='spectral projectors'):
        balancedTruncationReductor(globalBackendName='scipy').reduce(model, order=2)


def test_frequency_response_reductor_handles_singular_descriptor():
    A = -np.eye(4)
    E = np.diag([1.0, 2.0, 0.0, 0.0])
    model = ltiModel(
        matrixOperator(A, backendName='scipy'),
        matrixOperator(np.ones((4, 1)), backendName='scipy'),
        matrixOperator(np.ones((1, 4)), backendName='scipy'),
        E=matrixOperator(E, backendName='scipy'),
        backendName='scipy',
    )
    frequencies = np.logspace(-2, 2, 12)

    rom = frequencyResponseReductor(
        globalBackendName='scipy', frequencies=frequencies
    ).reduce(model, order=3)

    full = np.array([
        model.C.toNumpy() @ np.linalg.solve(1j * frequency * E - A, model.B.toNumpy())
        for frequency in frequencies
    ])
    reduced = np.array([
        rom.Cr.toNumpy() @ np.linalg.solve(
            1j * frequency * rom.Er.toNumpy() - rom.Ar.toNumpy(), rom.Br.toNumpy()
        )
        for frequency in frequencies
    ])
    assert rom.order == 3
    assert rom.solverInfo['descriptorKind'] == 'singular'
    assert np.linalg.norm(reduced - full) / np.linalg.norm(full) < 1e-10


def test_frequency_response_reductor_samples_discrete_model_on_unit_circle():
    model = ltiModel(
        matrixOperator(np.diag([0.2, 0.6]), backendName='scipy'),
        matrixOperator(np.array([[1.0], [0.5]]), backendName='scipy'),
        matrixOperator(np.array([[0.3, 1.0]]), backendName='scipy'),
        backendName='scipy',
        isContinuous=False,
    )

    reduced = frequencyResponseReductor(
        globalBackendName='scipy', frequencies=[0.0]
    ).reduce(model, order=1).model

    assert reduced.timeDomain == 'discrete'
    np.testing.assert_allclose(reduced.frequencyResponse([0.0]), model.frequencyResponse([0.0]))


def test_frequency_response_reductor_preserves_complex_system_at_full_order():
    A = np.array([[-1.0 + 0.4j, 0.2], [-0.1j, -2.0 - 0.3j]])
    B = np.array([[1.0 + 0.5j], [0.3 - 0.2j]])
    C = np.array([[0.7 - 0.1j, 1.0 + 0.4j]])
    model = ltiModel(
        matrixOperator(A, backendName='scipy'),
        matrixOperator(B, backendName='scipy'),
        matrixOperator(C, backendName='scipy'),
        backendName='scipy',
    )

    reduced = frequencyResponseReductor(
        globalBackendName='scipy', frequencies=[0.2, 1.3]
    ).reduce(model, order=2).model

    assert np.iscomplexobj(reduced.A.toNumpy())
    np.testing.assert_allclose(
        reduced.frequencyResponse([0.0, 0.7, 2.0]),
        model.frequencyResponse([0.0, 0.7, 2.0]),
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.mark.parametrize('order', [True, 1.5, 0, -1])
def test_frequency_response_reductor_rejects_invalid_order(order):
    model = ltiModel(
        matrixOperator(-np.eye(2), backendName='scipy'),
        matrixOperator(np.ones((2, 1)), backendName='scipy'),
        matrixOperator(np.ones((1, 2)), backendName='scipy'),
        backendName='scipy',
    )

    with pytest.raises(ValueError, match='positive integer'):
        frequencyResponseReductor('scipy', frequencies=[1.0]).reduce(model, order=order)


@pytest.mark.parametrize('frequencies', [[], [np.nan], [np.inf]])
def test_frequency_response_reductor_rejects_invalid_frequencies(frequencies):
    model = ltiModel(
        matrixOperator(-np.eye(2), backendName='scipy'),
        matrixOperator(np.ones((2, 1)), backendName='scipy'),
        matrixOperator(np.ones((1, 2)), backendName='scipy'),
        backendName='scipy',
    )

    with pytest.raises(ValueError, match='frequencies'):
        frequencyResponseReductor('scipy', frequencies=frequencies).reduce(model, order=1)


def test_frequency_response_reductor_rejects_order_above_numerical_snapshot_rank():
    model = ltiModel(
        matrixOperator(-np.diag([1.0, 2.0, 3.0]), backendName='scipy'),
        matrixOperator(np.array([[1.0], [0.0], [0.0]]), backendName='scipy'),
        matrixOperator(np.array([[1.0, 0.0, 0.0]]), backendName='scipy'),
        backendName='scipy',
    )

    with pytest.raises(ValueError, match='numerical snapshot rank 1'):
        frequencyResponseReductor('scipy', frequencies=[0.0, 0.0]).reduce(model, order=2)


def test_frequency_response_reductor_rejects_incompatible_snapshot_spaces():
    model = ltiModel(
        matrixOperator(-np.diag([1.0, 2.0]), backendName='scipy'),
        matrixOperator(np.array([[1.0], [0.0]]), backendName='scipy'),
        matrixOperator(np.array([[0.0, 1.0]]), backendName='scipy'),
        backendName='scipy',
    )

    with pytest.raises(ValueError, match='Projected descriptor is singular'):
        frequencyResponseReductor('scipy', frequencies=[0.0]).reduce(model, order=1)


def test_frequency_response_reductor_keeps_sparse_training_operators_unmaterialized(monkeypatch):
    from scipy.sparse import diags

    A = matrixOperator(diags([-1.0, -2.0, -3.0], format='csr'), backendName='scipy')
    E = matrixOperator(diags([1.0, 1.5, 2.0], format='csr'), backendName='scipy')
    model = ltiModel(
        A,
        matrixOperator(np.ones((3, 1)), backendName='scipy'),
        matrixOperator(np.ones((1, 3)), backendName='scipy'),
        E=E,
        backendName='scipy',
    )
    monkeypatch.setattr(A, 'toNumpy', lambda: (_ for _ in ()).throw(AssertionError('materialized A')))
    monkeypatch.setattr(E, 'toNumpy', lambda: (_ for _ in ()).throw(AssertionError('materialized E')))

    reduced = frequencyResponseReductor(
        'scipy', frequencies=[0.1, 1.0, 10.0]
    ).reduce(model, order=2)

    assert reduced.order == 2
    assert reduced.solverInfo['rightSnapshotRank'] >= 2
    assert reduced.solverInfo['leftSnapshotRank'] >= 2


def test_balanced_truncation_rejects_unconverged_gramian():
    from scipy.sparse import diags

    n = 20
    model = ltiModel(
        matrixOperator(diags(np.linspace(0.8, 0.999, n), format='csr'), backendName='scipy'),
        matrixOperator(np.ones((n, 1)), backendName='scipy'),
        matrixOperator(np.ones((1, n)), backendName='scipy'),
        backendName='scipy',
        isContinuous=False,
    )

    with pytest.raises(RuntimeError, match='did not converge'):
        balancedTruncationReductor(globalBackendName='scipy', maxIter=2, tol=1e-14).reduce(model, order=2)


def test_balanced_truncation_max_error_uses_hankel_tail_bound(rng):
    n = 8
    model = ltiModel(
        matrixOperator(-np.diag(np.linspace(1.0, 3.0, n)), backendName='scipy'),
        matrixOperator(rng.standard_normal((n, 2)), backendName='scipy'),
        matrixOperator(rng.standard_normal((2, n)), backendName='scipy'),
        backendName='scipy',
    )
    full = balancedTruncationReductor(globalBackendName='scipy').reduce(model)
    target_order = 3
    max_error = 2.0 * sum(full.hsv[target_order:])

    rom = balancedTruncationReductor(globalBackendName='scipy').reduce(model, maxError=max_error * (1.0 + 1e-12))

    assert rom.order <= target_order
    assert rom.solverInfo['hankelErrorBound'] <= max_error * (1.0 + 1e-10)
