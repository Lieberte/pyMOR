import numpy as np
import pytest

from mor.algorithm.dmd import exactDMD
from mor.operators import matrixOperator


def trajectory(operator, initialState, steps):
    snapshots = [initialState]
    for _ in range(steps - 1):
        snapshots.append(operator @ snapshots[-1])
    return np.column_stack(snapshots)


def test_exact_dmd_recovers_diagonal_dynamics():
    dynamics = np.diag([0.9, 1.05])
    snapshots = trajectory(dynamics, np.array([1.0, 2.0]), 8)

    result = exactDMD('scipy').fit(matrixOperator(snapshots, 'scipy'))

    np.testing.assert_allclose(np.sort(result.eigenvalues.real), [0.9, 1.05], atol=1e-12)
    np.testing.assert_allclose(result.reconstruct(), snapshots, atol=1e-11)
    np.testing.assert_allclose(result.predict(steps=[8])[:, 0], dynamics @ snapshots[:, -1], atol=1e-11)
    assert result.rank == 2
    assert result.diagnostics['variant'] == 'exact'


def test_exact_dmd_preserves_oscillatory_complex_modes():
    angle = 0.3
    dynamics = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    snapshots = trajectory(dynamics, np.array([1.0, 0.2]), 10)

    result = exactDMD('scipy').fit(matrixOperator(snapshots, 'scipy'), dt=0.1)

    expected = np.exp(np.array([-1j * angle, 1j * angle]))
    np.testing.assert_allclose(np.sort_complex(result.eigenvalues), np.sort_complex(expected), atol=1e-12)
    assert np.iscomplexobj(result.modes.toNumpy())
    np.testing.assert_allclose(result.reconstruct(), snapshots, atol=1e-11)
    np.testing.assert_allclose(
        np.sort_complex(result.continuousEigenvalues),
        np.sort_complex(np.log(expected) / 0.1),
        atol=1e-11,
    )


def test_exact_dmd_accepts_explicit_snapshot_pairs():
    dynamics = np.array([[0.8, 0.1], [0.0, 0.95]])
    xData = np.array([[1.0, 0.5, -0.2], [0.3, 1.2, 0.7]])
    yData = dynamics @ xData

    result = exactDMD('scipy').fit(
        matrixOperator(xData, 'scipy'), matrixOperator(yData, 'scipy')
    )

    np.testing.assert_allclose(np.sort(result.eigenvalues.real), [0.8, 0.95], atol=1e-12)
    with pytest.raises(ValueError, match='prediction steps'):
        result.reconstruct()


def test_exact_dmd_rank_truncation():
    dynamics = np.diag([0.9, 0.7])
    snapshots = trajectory(dynamics, np.array([1.0, 1.0]), 7)

    result = exactDMD('scipy').fit(matrixOperator(snapshots, 'scipy'), rank=1)

    assert result.rank == 1
    assert result.modes.shape == (2, 1)
    assert result.reconstruct().shape == snapshots.shape


def test_exact_dmd_filters_numerically_zero_singular_values():
    snapshots = np.array([
        [1.0, 0.9, 0.81, 0.729],
        [2.0, 1.8, 1.62, 1.458],
        [0.0, 0.0, 0.0, 0.0],
    ])

    result = exactDMD('scipy').fit(matrixOperator(snapshots, 'scipy'), rank=3)

    assert result.rank == 1
    assert result.diagnostics['requestedRank'] == 3
    assert result.diagnostics['numericalRank'] == 1
    assert np.isfinite(result.diagnostics['conditionNumber'])


def test_projected_and_exact_modes_share_reduced_spectrum():
    dynamics = np.array([[0.85, 0.1], [0.0, 0.95]])
    snapshots = trajectory(dynamics, np.array([1.0, 0.4]), 8)
    operator = matrixOperator(snapshots, 'scipy')

    exact = exactDMD('scipy').fit(operator, mode='exact')
    projected = exactDMD('scipy').fit(operator, mode='projected')

    np.testing.assert_allclose(
        np.sort_complex(exact.eigenvalues), np.sort_complex(projected.eigenvalues), atol=1e-12
    )
    assert projected.diagnostics['mode'] == 'projected'


def test_global_amplitude_fit_does_not_increase_training_error():
    rng = np.random.default_rng(4)
    dynamics = np.array([[0.9, -0.2], [0.2, 0.9]])
    snapshots = trajectory(dynamics, np.array([1.0, 0.2]), 20)
    snapshots += 2e-3 * rng.standard_normal(snapshots.shape)
    operator = matrixOperator(snapshots, 'scipy')

    first = exactDMD('scipy').fit(operator, amplitudeMethod='first')
    globalFit = exactDMD('scipy').fit(operator, amplitudeMethod='global')

    assert globalFit.diagnostics['reconstructionError'] <= first.diagnostics['reconstructionError']
    assert globalFit.diagnostics['amplitudeMethod'] == 'global'


def test_exact_dmd_reports_numerical_diagnostics():
    dynamics = np.diag([0.8, 0.95])
    snapshots = trajectory(dynamics, np.array([1.0, 2.0]), 8)

    result = exactDMD('scipy').fit(matrixOperator(snapshots, 'scipy'))

    assert result.diagnostics['snapshotResidual'] < 1e-12
    assert result.diagnostics['oneStepResidual'] < 1e-12
    assert result.diagnostics['reconstructionError'] < 1e-12


@pytest.mark.parametrize(('option', 'value', 'message'), [
    ('mode', 'unknown', 'mode'),
    ('amplitudeMethod', 'unknown', 'amplitudeMethod'),
])
def test_exact_dmd_rejects_unknown_options(option, value, message):
    kwargs = {option: value}
    with pytest.raises(ValueError, match=message):
        exactDMD('scipy').fit(matrixOperator(np.eye(2), 'scipy'), **kwargs)


@pytest.mark.parametrize('dt', [0, -1, np.inf, np.nan, True])
def test_exact_dmd_rejects_invalid_dt(dt):
    with pytest.raises(ValueError, match='dt'):
        exactDMD('scipy').fit(matrixOperator(np.eye(2), 'scipy'), dt=dt)


def test_exact_dmd_rejects_nonfinite_snapshots():
    with pytest.raises(ValueError, match='finite'):
        exactDMD('scipy').fit(matrixOperator(np.array([[1.0, np.nan]]), 'scipy'))


def test_exact_dmd_rejects_zero_rank_and_zero_snapshots():
    snapshots = matrixOperator(np.zeros((3, 4)), 'scipy')
    with pytest.raises(ValueError, match='positive numerical snapshot rank'):
        exactDMD('scipy').fit(snapshots, rank=0)
    with pytest.raises(ValueError, match='positive numerical snapshot rank'):
        exactDMD('scipy').fit(snapshots)


def test_exact_dmd_rejects_mismatched_pairs():
    with pytest.raises(ValueError, match='identical'):
        exactDMD('scipy').fit(
            matrixOperator(np.ones((2, 3)), 'scipy'),
            matrixOperator(np.ones((2, 2)), 'scipy'),
        )


def test_exact_dmd_accepts_composed_snapshot_operator():
    dynamics = np.diag([0.8, 0.95])
    snapshots = trajectory(dynamics, np.array([1.0, 2.0]), 7)
    base = matrixOperator(snapshots, 'scipy')

    result = exactDMD('scipy').fit(base + 0.0 * base)

    np.testing.assert_allclose(result.reconstruct(), snapshots, atol=1e-11)


def test_exact_dmd_torch_preserves_device_and_complex_dtype():
    torch = pytest.importorskip('torch')
    angle = 0.2
    dynamics = torch.tensor(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=torch.float64,
    )
    states = [torch.tensor([1.0, 0.3], dtype=torch.float64)]
    for _ in range(7):
        states.append(dynamics @ states[-1])
    snapshots = torch.stack(states, dim=1)

    result = exactDMD('torch').fit(matrixOperator(snapshots, 'torch'))
    reconstructed = result.reconstruct()

    assert result.modes.toBackendData().device == snapshots.device
    assert result.eigenvalues.device == snapshots.device
    assert result.eigenvalues.dtype == torch.complex128
    assert reconstructed.device == snapshots.device
    assert reconstructed.dtype == snapshots.dtype
    torch.testing.assert_close(reconstructed, snapshots)


def test_torch_svd_preserves_complex_snapshot_information():
    torch = pytest.importorskip('torch')
    snapshots = torch.tensor(
        [[1.0 + 0.2j, 0.8 + 0.1j, 0.6 - 0.1j], [0.3j, 0.2 + 0.4j, -0.1 + 0.2j]],
        dtype=torch.complex128,
    )

    result = exactDMD('torch').fit(matrixOperator(snapshots, 'torch'))

    assert result.modes.toBackendData().dtype == torch.complex128
    assert result.singularValues.dtype == torch.float64
