import torch
import numpy as np
import pytest

from mor.algorithm.dmd import (
    dmdc,
    forwardBackwardDMD,
    hankelDMD,
    optimizedDMD,
    tlsDMD,
)
from mor.operators import matrixOperator


def trajectory(operator, initialState, steps):
    states = [initialState]
    for _ in range(steps - 1):
        states.append(operator @ states[-1])
    return np.column_stack(states)


def test_tls_dmd_runs_on_noisy_snapshots():
    rng = np.random.default_rng(8)
    dynamics = np.array([[0.92, -0.15], [0.15, 0.92]])
    snapshots = trajectory(dynamics, np.array([1.0, 0.2]), 30)
    snapshots += 1e-3 * rng.standard_normal(snapshots.shape)

    result = tlsDMD('scipy').fit(matrixOperator(snapshots, 'scipy'), rank=2)

    assert result.rank == 2
    assert result.diagnostics['variant'] == 'tls'
    assert result.diagnostics['tlsProjectionResidual'] >= 0
    assert result.reconstruct().shape == snapshots.shape
    assert result.diagnostics['oneStepResidual'] > 0


def test_forward_backward_dmd_recovers_exact_spectrum():
    dynamics = np.diag([0.8, 0.95])
    snapshots = trajectory(dynamics, np.array([1.0, 2.0]), 10)

    result = forwardBackwardDMD('scipy').fit(matrixOperator(snapshots, 'scipy'))

    np.testing.assert_allclose(np.sort(result.eigenvalues.real), [0.8, 0.95], atol=1e-10)
    assert result.diagnostics['variant'] == 'forwardBackward'


def test_dmdc_identifies_controlled_system_and_simulates():
    rng = np.random.default_rng(9)
    stateOperator = np.array([[0.85, 0.1], [0.0, 0.9]])
    controlOperator = np.array([[1.0], [0.4]])
    controls = rng.standard_normal((1, 20))
    states = [np.array([0.2, -0.3])]
    for control in controls.T:
        states.append(stateOperator @ states[-1] + controlOperator @ control)
    snapshots = np.column_stack(states)

    result = dmdc('scipy').fit(
        matrixOperator(snapshots, 'scipy'),
        matrixOperator(controls, 'scipy'),
        rank=3,
        stateRank=2,
    )

    simulated = result.simulate(snapshots[:, 0], controls)
    np.testing.assert_allclose(simulated, snapshots, atol=1e-11)
    assert result.diagnostics['oneStepResidual'] < 1e-12
    assert result.controlOperator.shape == (2, 1)
    with pytest.raises(ValueError, match='requires controls'):
        result.reconstruct()


def test_hankel_dmd_reconstructs_delayed_scalar_sequence():
    snapshots = np.array([[0.9 ** index for index in range(12)]])

    result = hankelDMD('scipy').fit(
        matrixOperator(snapshots, 'scipy'), delay=3, rank=1
    )

    np.testing.assert_allclose(result.reconstruct(), snapshots, atol=1e-11)
    assert result.augmentedModes.shape == (3, 1)
    assert result.diagnostics['variant'] == 'hankel'


def test_optimized_dmd_improves_or_preserves_global_fit():
    rng = np.random.default_rng(10)
    dynamics = np.array([[0.93, -0.12], [0.12, 0.93]])
    snapshots = trajectory(dynamics, np.array([1.0, 0.2]), 20)
    snapshots += 5e-4 * rng.standard_normal(snapshots.shape)

    result = optimizedDMD('scipy').fit(
        matrixOperator(snapshots, 'scipy'), rank=2, maxNfev=100
    )

    assert result.diagnostics['reconstructionError'] <= (
        result.diagnostics['initialReconstructionError'] + 1e-12
    )
    assert result.diagnostics['optimizerEvaluations'] > 0
    assert result.diagnostics['variant'] == 'optimized'


@pytest.mark.parametrize('algorithm', [tlsDMD, forwardBackwardDMD, hankelDMD, optimizedDMD])
def test_advanced_variants_reject_unverified_torch_paths(algorithm):
    snapshots = matrixOperator(torch.eye(3, dtype=torch.float64), 'torch')

    with pytest.raises(NotImplementedError, match='scipy backend'):
        algorithm('torch').fit(snapshots)


def test_forward_backward_rejects_singular_reverse_mapping():
    with pytest.raises(ValueError, match='singular'):
        forwardBackwardDMD('scipy').fit(matrixOperator(np.eye(2), 'scipy'),
                                       matrixOperator(np.diag([0.8, 0.]), 'scipy'), rank=2)


def test_hankel_delay_at_last_valid_window_reconstructs():
    snapshots = np.array([[0.9 ** index for index in range(12)]])
    result = hankelDMD('scipy').fit(matrixOperator(snapshots, 'scipy'), delay=11, rank=1)
    np.testing.assert_allclose(result.reconstruct(), snapshots, atol=1e-10)
    with pytest.raises(ValueError, match='more snapshots'):
        hankelDMD('scipy').fit(matrixOperator(snapshots, 'scipy'), delay=12)


def test_optimized_dmd_reports_exhausted_budget():
    rng = np.random.default_rng(5)
    snapshots = trajectory(np.array([[0.93, -0.12], [0.12, 0.93]]), np.array([1., 0.2]), 20)
    snapshots += rng.normal(size=snapshots.shape) * 0.01
    result = optimizedDMD('scipy').fit(matrixOperator(snapshots, 'scipy'), rank=2, maxNfev=1)
    assert not result.diagnostics['optimizerSuccess']
    assert result.diagnostics['optimizerEvaluations'] == 1


def test_dmd_long_unstable_prediction_rejects_nonfinite_result():
    from mor.algorithm.dmd.exact import exactDMD
    snapshots = np.array([[1.1 ** index for index in range(10)]])
    result = exactDMD('scipy').fit(matrixOperator(snapshots, 'scipy'), rank=1)
    with np.errstate(over='ignore', invalid='ignore'):
        with pytest.raises(ValueError, match='prediction overflowed'):
            result.predict(steps=np.array([10000]))


def test_dmdc_reports_insufficient_excitation_and_can_require_identifiability():
    dynamics = np.diag([0.8, 0.95])
    states = matrixOperator(trajectory(dynamics, np.ones(2), 20), 'scipy')
    controls = matrixOperator(np.zeros((1, 19)), 'scipy')
    result = dmdc('scipy').fit(states, controls)
    assert not result.diagnostics['identifiable']
    assert result.diagnostics['controlRank'] == 0
    with pytest.raises(ValueError, match='not identifiable'):
        dmdc('scipy').fit(states, controls, requireIdentifiable=True)
    assert result.simulate(np.ones(2), np.empty((1, 0)), includeInitial=False).shape == (2, 0)
    with pytest.raises(ValueError, match='finite'):
        result.simulate(np.ones(2), np.array([[np.nan]]))


def test_hankel_memory_budget_is_checked_before_augmentation(monkeypatch):
    states = matrixOperator(np.ones((10, 100)), 'scipy')
    monkeypatch.setattr(np, 'vstack', lambda *args: pytest.fail('augmentation allocated before budget check'))
    with pytest.raises(MemoryError, match='requires 204000 bytes'):
        hankelDMD('scipy').fit(states, delay=50, maxAugmentedBytes=1000)
    with pytest.raises(ValueError, match='maxAugmentedBytes'):
        hankelDMD('scipy').fit(states, delay=50, maxAugmentedBytes=-1)
