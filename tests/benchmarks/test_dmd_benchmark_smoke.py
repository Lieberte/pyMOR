import time

import numpy as np
import pytest

from mor.algorithm.auto.dmd import selectDMDVariant
from mor.algorithm.registry import algorithmRegistry
from mor.operators import matrixOperator


def trajectory(operator, initialState, steps):
    states = [initialState]
    for _ in range(steps - 1):
        states.append(operator @ states[-1])
    return np.column_stack(states)


def eigenvalue_error(actual, expected):
    unmatched = list(actual)
    errors = []
    for value in expected:
        index = int(np.argmin(np.abs(np.asarray(unmatched) - value)))
        errors.append(abs(unmatched.pop(index) - value))
    return float(max(errors, default=0.0))


@pytest.mark.benchmark
@pytest.mark.parametrize(('case', 'dynamics'), [
    ('diagonal_decay', np.diag([0.8, 0.95])),
    ('conjugate_oscillation', np.array([[0.92, -0.2], [0.2, 0.92]])),
    ('nonnormal', np.array([[0.85, 0.4], [0.0, 0.9]])),
])
def test_dmd_benchmark_smoke_collects_accuracy_and_runtime(case, dynamics):
    snapshots = trajectory(dynamics, np.array([1.0, 0.3]), 30)
    operator = matrixOperator(snapshots, 'scipy')
    variant = selectDMDVariant('scipy')
    algorithm = algorithmRegistry.get('dmd', variant, backendName='scipy')

    start = time.perf_counter()
    result = algorithm.fit(operator, rank=2)
    elapsedMs = (time.perf_counter() - start) * 1000.0
    metrics = {
        'case': case,
        'backend': 'scipy',
        'operator_form': 'matrixOperator',
        'data_regime': case,
        'auto_variant': variant,
        'elapsed_ms': elapsedMs,
        'rank': result.rank,
        'eigenvalue_error': eigenvalue_error(result.eigenvalues, np.linalg.eigvals(dynamics)),
        'one_step_error': result.diagnostics['oneStepResidual'],
        'reconstruction_error': result.diagnostics['reconstructionError'],
    }

    assert metrics['auto_variant'] == 'exact'
    assert metrics['elapsed_ms'] >= 0
    assert metrics['rank'] == 2
    assert metrics['eigenvalue_error'] < 1e-10
    assert metrics['one_step_error'] < 1e-10
    assert metrics['reconstruction_error'] < 1e-10


@pytest.mark.benchmark
def test_dmd_noise_benchmark_compares_robust_variants_without_runtime_gate():
    rng = np.random.default_rng(12)
    dynamics = np.array([[0.92, -0.15], [0.15, 0.92]])
    clean = trajectory(dynamics, np.array([1.0, 0.3]), 40)
    noisy = clean + 2e-3 * rng.standard_normal(clean.shape)
    operator = matrixOperator(noisy, 'scipy')
    results = []
    for variant in ('exact', 'tls', 'forwardBackward'):
        algorithm = algorithmRegistry.get('dmd', variant, backendName='scipy')
        start = time.perf_counter()
        result = algorithm.fit(operator, rank=2)
        results.append({
            'variant': variant,
            'elapsed_ms': (time.perf_counter() - start) * 1000.0,
            'rank': result.rank,
            'eigenvalue_error': eigenvalue_error(
                result.eigenvalues, np.linalg.eigvals(dynamics)
            ),
            'one_step_error': result.diagnostics['oneStepResidual'],
        })

    assert {row['variant'] for row in results} == {'exact', 'tls', 'forwardBackward'}
    for row in results:
        assert row['elapsed_ms'] >= 0
        assert row['rank'] == 2
        assert np.isfinite(row['eigenvalue_error'])
        assert np.isfinite(row['one_step_error'])
