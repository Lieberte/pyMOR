import numpy as np
import pytest

from mor.operators import matrixOperator
from mor.reductors import podDmdReductor, podDmdResult, podReductor, reductorRegistry


def trajectory(operator, initialState, steps):
    states = [initialState]
    for _ in range(steps - 1):
        states.append(operator @ states[-1])
    return np.column_stack(states)


def embeddedSnapshots(stateDimension=20, steps=16):
    rng = np.random.default_rng(41)
    basis, _ = np.linalg.qr(rng.standard_normal((stateDimension, 2)))
    dynamics = np.array([[0.9, -0.15], [0.15, 0.9]])
    reduced = trajectory(dynamics, np.array([1.0, 0.2]), steps)
    return basis @ reduced, dynamics


def test_pod_dmd_reconstructs_low_rank_full_state_dynamics():
    snapshots, dynamics = embeddedSnapshots()

    result = podDmdReductor('scipy').reduce(
        matrixOperator(snapshots, 'scipy'), podRank=2, dmdRank=2
    )

    assert isinstance(result, podDmdResult)
    assert result.podRank == 2
    assert result.dmdRank == 2
    assert result.modes.shape == (20, 2)
    np.testing.assert_allclose(result.reconstruct(), snapshots, atol=1e-11)
    np.testing.assert_allclose(
        np.sort_complex(result.eigenvalues),
        np.sort_complex(np.linalg.eigvals(dynamics)),
        atol=1e-11,
    )
    assert result.diagnostics['fullReconstructionError'] < 1e-11
    assert result.diagnostics['fullOneStepResidual'] < 1e-11


def test_pod_dmd_centering_restores_snapshot_mean():
    rng = np.random.default_rng(42)
    basis, _ = np.linalg.qr(rng.standard_normal((10, 2)))
    mean = np.linspace(5.0, 10.0, 10)
    angles = np.arange(8) * (2 * np.pi / 8)
    reduced = np.vstack([np.cos(angles), np.sin(angles)])
    snapshots = mean[:, None] + basis @ reduced

    result = podDmdReductor('scipy', centerSnapshots=True).reduce(
        matrixOperator(snapshots, 'scipy'), podRank=2, dmdRank=2
    )

    np.testing.assert_allclose(result.snapshotMean, mean, atol=1e-12)
    np.testing.assert_allclose(result.reconstruct(), snapshots, atol=1e-10)
    np.testing.assert_allclose(result.lift(result.project(snapshots)), snapshots, atol=1e-12)


def test_weighted_pod_dmd_uses_product_coordinates():
    snapshots, _ = embeddedSnapshots(stateDimension=8)
    productData = np.diag(np.linspace(1.0, 2.0, 8))
    product = matrixOperator(productData, 'scipy')

    result = podDmdReductor('scipy', product=product).reduce(
        matrixOperator(snapshots, 'scipy'), podRank=2, dmdRank=2
    )

    basis = result.basis.toNumpy()
    np.testing.assert_allclose(basis.conj().T @ productData @ basis, np.eye(2), atol=1e-10)
    np.testing.assert_allclose(result.lift(result.project(snapshots)), snapshots, atol=1e-10)
    np.testing.assert_allclose(result.reconstruct(), snapshots, atol=1e-10)


def test_pod_dmd_explicit_pairs_fit_pod_on_both_snapshot_sets():
    rng = np.random.default_rng(43)
    basis, _ = np.linalg.qr(rng.standard_normal((12, 2)))
    dynamics = np.array([[0.8, 0.1], [0.0, 0.95]])
    reducedX = rng.standard_normal((2, 8))
    xData = basis @ reducedX
    yData = basis @ (dynamics @ reducedX)

    result = podDmdReductor('scipy').reduce(
        matrixOperator(xData, 'scipy'),
        shiftedSnapshots=matrixOperator(yData, 'scipy'),
        podRank=2,
        dmdRank=2,
    )

    assert result.trainingSteps is None
    assert result.diagnostics['fullOneStepResidual'] < 1e-11
    with pytest.raises(ValueError, match='prediction steps'):
        result.reconstruct()


def test_pod_dmd_reuses_existing_pod_result():
    snapshots, _ = embeddedSnapshots(stateDimension=10)
    operator = matrixOperator(snapshots, 'scipy')
    pod = podReductor('scipy').reduce(operator, rank=2)

    result = podDmdReductor('scipy').reduce(operator, podBasis=pod, dmdRank=2)

    assert result.pod is pod
    np.testing.assert_allclose(result.reconstruct(), snapshots, atol=1e-11)


def test_pod_dmd_keeps_pod_and_dmd_ranks_independent():
    rng = np.random.default_rng(44)
    basis, _ = np.linalg.qr(rng.standard_normal((15, 3)))
    reduced = np.vstack([
        trajectory(np.diag([0.9, 0.8]), np.array([1.0, 0.3]), 14),
        1e-3 * rng.standard_normal(14),
    ])
    snapshots = basis @ reduced

    result = podDmdReductor('scipy').reduce(
        matrixOperator(snapshots, 'scipy'), podRank=3, dmdRank=2
    )

    assert result.podRank == 3
    assert result.dmdRank == 2
    assert result.diagnostics['podRank'] == 3
    assert result.diagnostics['dmdRank'] == 2


def test_pod_dmd_forwards_dmd_variant_options():
    snapshots, _ = embeddedSnapshots(stateDimension=10)

    result = podDmdReductor(
        'scipy', dmdOptions={'amplitudeMethod': 'global'}
    ).reduce(matrixOperator(snapshots, 'scipy'), podRank=2, dmdRank=2)

    assert result.diagnostics['dmdVariant'] == 'exact'
    assert result.reducedDmd.diagnostics['amplitudeMethod'] == 'global'


def test_pod_dmd_rejects_reserved_nested_dmd_options():
    snapshots, _ = embeddedSnapshots(stateDimension=10)
    with pytest.raises(ValueError, match='reserved dmdOptions: rank'):
        podDmdReductor('scipy', dmdOptions={'rank': 1}).reduce(
            matrixOperator(snapshots, 'scipy'), podRank=2, dmdRank=2
        )


def test_pod_dmd_accepts_composed_snapshot_operator():
    snapshots, _ = embeddedSnapshots(stateDimension=10)
    base = matrixOperator(snapshots, 'scipy')

    result = podDmdReductor('scipy').reduce(
        base + 0.0 * base, podRank=2, dmdRank=2
    )

    np.testing.assert_allclose(result.reconstruct(), snapshots, atol=1e-11)


def test_pod_dmd_torch_preserves_backend_dtype_and_device():
    torch = pytest.importorskip('torch')
    snapshots, _ = embeddedSnapshots(stateDimension=8)
    tensor = torch.tensor(snapshots, dtype=torch.float32)

    result = podDmdReductor('torch').reduce(
        matrixOperator(tensor, 'torch'), podRank=2, dmdRank=2
    )
    reconstructed = result.reconstruct()

    assert reconstructed.dtype == torch.float32
    assert reconstructed.device == tensor.device
    torch.testing.assert_close(reconstructed, tensor, atol=2e-5, rtol=2e-5)


def test_pod_dmd_is_registered():
    assert 'podDmd' in reductorRegistry.list()


@pytest.mark.parametrize(('name', 'kwargs'), [
    ('podRank', {'podRank': 0}),
    ('dmdRank', {'dmdRank': True}),
])
def test_pod_dmd_rejects_invalid_ranks(name, kwargs):
    snapshots, _ = embeddedSnapshots(stateDimension=5)
    with pytest.raises(ValueError, match=name):
        podDmdReductor('scipy').reduce(matrixOperator(snapshots, 'scipy'), **kwargs)


def test_pod_dmd_rejects_controlled_variant_until_contract_exists():
    snapshots, _ = embeddedSnapshots(stateDimension=5)
    controls = matrixOperator(np.ones((1, snapshots.shape[1] - 1)), 'scipy')
    with pytest.raises(NotImplementedError, match='POD-DMDc'):
        podDmdReductor('scipy').reduce(
            matrixOperator(snapshots, 'scipy'), controlSnapshots=controls
        )
