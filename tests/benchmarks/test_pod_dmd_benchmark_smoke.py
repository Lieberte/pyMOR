import time

import numpy as np
import pytest

from mor.algorithm.dmd import exactDMD
from mor.operators import matrixOperator
from mor.reductors import podDmdReductor


def embeddedTrajectory(stateDimension, latentDimension, steps, seed):
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((stateDimension, latentDimension)))
    eigenvalues = np.linspace(0.75, 0.95, latentDimension)
    reduced = [rng.standard_normal(latentDimension)]
    for _ in range(steps - 1):
        reduced.append(eigenvalues * reduced[-1])
    return basis @ np.column_stack(reduced), eigenvalues


@pytest.mark.benchmark
@pytest.mark.parametrize('stateDimension', [80, 160])
def test_pod_dmd_benchmark_collects_layered_errors(stateDimension):
    snapshots, expectedEigenvalues = embeddedTrajectory(
        stateDimension, latentDimension=4, steps=24, seed=50 + stateDimension
    )
    operator = matrixOperator(snapshots, 'scipy')

    directStart = time.perf_counter()
    direct = exactDMD('scipy').fit(operator, rank=4)
    directElapsed = (time.perf_counter() - directStart) * 1000.0
    podStart = time.perf_counter()
    combined = podDmdReductor('scipy').reduce(
        operator, podRank=4, dmdRank=4
    )
    podElapsed = (time.perf_counter() - podStart) * 1000.0
    metrics = {
        'case': f'embedded_diagonal_n{stateDimension}',
        'backend': 'scipy',
        'operator_form': 'matrixOperator',
        'data_regime': 'high_dimensional_low_rank_snapshots',
        'state_dimension': stateDimension,
        'pod_rank': combined.podRank,
        'dmd_rank': combined.dmdRank,
        'coordinate_reduction_ratio': combined.podRank / stateDimension,
        'pod_projection_error': combined.diagnostics['podProjectionError'],
        'reduced_dmd_error': combined.diagnostics['reducedDmdReconstructionError'],
        'full_reconstruction_error': combined.diagnostics['fullReconstructionError'],
        'direct_dmd_error': direct.diagnostics['reconstructionError'],
        'pod_dmd_elapsed_ms': podElapsed,
        'direct_dmd_elapsed_ms': directElapsed,
    }

    np.testing.assert_allclose(
        np.sort(combined.eigenvalues.real), expectedEigenvalues, atol=1e-10
    )
    assert metrics['pod_rank'] == metrics['dmd_rank'] == 4
    assert metrics['coordinate_reduction_ratio'] < 0.1
    assert metrics['pod_projection_error'] < 1e-12
    assert metrics['reduced_dmd_error'] < 1e-11
    assert metrics['full_reconstruction_error'] < 1e-11
    assert metrics['direct_dmd_error'] < 1e-11
    assert metrics['pod_dmd_elapsed_ms'] >= 0
    assert metrics['direct_dmd_elapsed_ms'] >= 0
