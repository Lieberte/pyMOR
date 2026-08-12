import time

import numpy as np
import pytest
from scipy import linalg
from scipy.sparse import diags

from mor.algorithm.auto.lyapunov import selectLyapunovAlgorithm
from mor.algorithm.auto.svd import selectSVDVariant
from mor.algorithm.registry import algorithmRegistry
from mor.operators import matrixOperator
from tests.benchmarks.standard_cases import (
    STANDARD_BENCHMARK_SOURCES,
    extreme_scaled_diagonal,
    ill_conditioned_diagonal,
    matlab_gallery_grcar,
    matlab_gallery_hilbert,
    matrix_depot_poisson_2d,
    singular_diagonal,
)
try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


def _to_numpy(data):
    if torch is not None and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _reconstruct(U, S, Vt):
    return _to_numpy(U) @ np.diag(_to_numpy(S)) @ _to_numpy(Vt)


def _svd_metric(variant, operator, target, rank, backend_name='scipy'):
    start = time.perf_counter()
    algo = algorithmRegistry.get(category='svd', variant=variant, backendName=backend_name)
    U, S, Vt = algo.decompose(operator, rank=rank)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    approx = _reconstruct(U, S, Vt)
    return {
        'variant': variant,
        'elapsed_ms': elapsed_ms,
        'rank': len(S),
        'relative_error': float(np.linalg.norm(approx - target) / np.linalg.norm(target)),
    }


@pytest.mark.benchmark
@pytest.mark.parametrize("n_rows", [12, 18, 24])
def test_svd_benchmark_smoke_collects_operator_backend_metrics(n_rows):
    dense = matlab_gallery_grcar(n_rows)[:, :6]
    operator = matrixOperator(dense, backendName='scipy')
    auto_variant = selectSVDVariant(backendName='scipy', xOperator=operator)

    metrics = {
        'case': f'matlab_gallery_grcar_tall_skinny_svd_n{n_rows}',
        'source': 'matlab_gallery',
        'backend': 'scipy',
        'operator_form': 'matrixOperator',
        'data_regime': 'standard_nonnormal_tall_skinny',
        'auto_variant': auto_variant,
        'results': [
            _svd_metric('economic', operator, dense, rank=6),
            _svd_metric('qrSVD', operator, dense, rank=6),
        ],
    }

    assert metrics['auto_variant'] == 'qrSVD'
    assert metrics['source'] in STANDARD_BENCHMARK_SOURCES
    assert {result['variant'] for result in metrics['results']} == {'economic', 'qrSVD'}
    for result in metrics['results']:
        assert result['elapsed_ms'] >= 0.0
        assert result['rank'] == 6
        assert result['relative_error'] < 1e-12


def _relative_lyapunov_residual(a_data, x_data, q_data):
    residual = a_data @ x_data + x_data @ a_data.T + q_data
    return float(np.linalg.norm(residual) / np.linalg.norm(q_data))


def _stable_grcar_continuous(n):
    G = matlab_gallery_grcar(n)
    eig_radius = max(abs(np.linalg.eigvals(G)))
    return G / (2.0 * eig_radius) - 1.5 * np.eye(n)


@pytest.mark.benchmark
@pytest.mark.parametrize("m", [3, 4, 5])
def test_lradi_benchmark_smoke_collects_sparse_metrics_against_dense_reference(m):
    a_sparse = -matrix_depot_poisson_2d(m)
    n = a_sparse.shape[0]
    b_data = np.ones((n, 1))
    A = matrixOperator(a_sparse, backendName='scipy')
    B = matrixOperator(b_data, backendName='scipy')
    q_data = b_data @ b_data.T
    auto_variant = selectLyapunovAlgorithm(
        variant='auto',
        A=A,
        backendName='scipy',
        isContinuous=True,
    )

    lradi = algorithmRegistry.get(
        category='lyapunov',
        variant='lradi',
        backendName='scipy',
        tol=1e-10,
        maxIter=120,
        compressionColumns=12,
        shiftOptions={'maxShifts': 8},
    )
    start = time.perf_counter()
    Z = lradi.solve(A, None, B)
    lradi_elapsed_ms = (time.perf_counter() - start) * 1000.0
    x_lradi = Z.toBackendData()

    start = time.perf_counter()
    x_ref = linalg.solve_continuous_lyapunov(a_sparse.toarray(), -q_data)
    dense_elapsed_ms = (time.perf_counter() - start) * 1000.0

    metrics = {
        'case': f'matrix_depot_poisson_2d_lradi_m{m}',
        'source': 'julia_matrix_depot',
        'backend': 'scipy',
        'operator_form': 'sparse_matrixOperator',
        'data_regime': 'standard_sparse_poisson_lr_solution',
        'auto_variant': auto_variant,
        'results': [
            {
                'variant': 'lradi',
                'elapsed_ms': lradi_elapsed_ms,
                'rank': Z.left.shape[1],
                'iterations': lradi.lastIterations,
                'relative_residual': _relative_lyapunov_residual(a_sparse, x_lradi, q_data),
                'relative_reference_error': float(
                    np.linalg.norm(x_lradi - x_ref) / np.linalg.norm(x_ref)
                ),
            },
            {
                'variant': 'dense_reference',
                'elapsed_ms': dense_elapsed_ms,
                'rank': np.linalg.matrix_rank(x_ref),
                'iterations': 1,
                'relative_residual': _relative_lyapunov_residual(a_sparse.toarray(), x_ref, q_data),
                'relative_reference_error': 0.0,
            },
        ],
    }

    assert metrics['auto_variant'] == 'bartelsStewart'
    assert metrics['source'] in STANDARD_BENCHMARK_SOURCES
    lradi_metric = metrics['results'][0]
    dense_metric = metrics['results'][1]
    assert lradi_metric['elapsed_ms'] >= 0.0
    assert dense_metric['elapsed_ms'] >= 0.0
    assert lradi_metric['rank'] > 0
    assert lradi_metric['iterations'] > 0
    assert lradi_metric['relative_residual'] < 1e-8
    assert lradi_metric['relative_reference_error'] < 1e-8
    assert dense_metric['relative_residual'] < 1e-12


@pytest.mark.benchmark
@pytest.mark.parametrize("n", [6, 8, 10])
def test_dense_lyapunov_benchmark_smoke_compares_bartels_stewart_and_sign(n):
    a_data = _stable_grcar_continuous(n)
    b_data = np.ones((n, 1))
    A = matrixOperator(a_data, backendName='scipy')
    B = matrixOperator(b_data, backendName='scipy')
    q_data = b_data @ b_data.T
    auto_variant = selectLyapunovAlgorithm(
        variant='auto',
        A=A,
        backendName='scipy',
        isContinuous=True,
    )
    results = []
    for variant in ('bartelsStewart', 'sign'):
        algo = algorithmRegistry.get(category='lyapunov', variant=variant, backendName='scipy', maxIter=80)
        start = time.perf_counter()
        Z = algo.solve(A, None, B)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        x_data = Z.toBackendData()
        results.append(
            {
                'variant': variant,
                'elapsed_ms': elapsed_ms,
                'rank': getattr(Z, 'left', x_data).shape[1],
                'relative_residual': _relative_lyapunov_residual(a_data, x_data, q_data),
            }
        )

    metrics = {
        'case': f'matlab_gallery_grcar_dense_continuous_lyapunov_n{n}',
        'source': 'matlab_gallery',
        'backend': 'scipy',
        'operator_form': 'matrixOperator',
        'data_regime': 'standard_dense_nonnormal_stable',
        'auto_variant': auto_variant,
        'results': results,
    }

    assert metrics['auto_variant'] == 'bartelsStewart'
    assert {result['variant'] for result in metrics['results']} == {'bartelsStewart', 'sign'}
    for result in metrics['results']:
        assert result['elapsed_ms'] >= 0.0
        assert result['relative_residual'] < 1e-10


@pytest.mark.benchmark
@pytest.mark.parametrize("n", [6, 8, 10])
def test_lrsmith_benchmark_smoke_compares_sparse_discrete_to_dense_reference(n):
    g_data = matlab_gallery_grcar(n)
    eig_radius = max(abs(np.linalg.eigvals(g_data)))
    a_dense = g_data / (2.5 * eig_radius)
    a_sparse = diags(a_dense.diagonal(), format='csr')
    b_data = np.ones((n, 1))
    A = matrixOperator(a_sparse, backendName='scipy')
    B = matrixOperator(b_data, backendName='scipy')
    q_data = b_data @ b_data.T
    auto_variant = selectLyapunovAlgorithm(
        variant='auto',
        A=A,
        backendName='scipy',
        isContinuous=False,
    )

    smith = algorithmRegistry.get(
        category='lyapunov',
        variant='lrsmith',
        backendName='scipy',
        maxIter=120,
        tol=1e-12,
    )
    start = time.perf_counter()
    Z = smith.solve(A, None, B)
    smith_elapsed_ms = (time.perf_counter() - start) * 1000.0
    x_smith = Z.toBackendData()

    start = time.perf_counter()
    x_ref = linalg.solve_discrete_lyapunov(a_sparse.toarray(), q_data)
    dense_elapsed_ms = (time.perf_counter() - start) * 1000.0
    residual = a_sparse @ x_smith @ a_sparse.T - x_smith + q_data

    metrics = {
        'case': f'matlab_gallery_grcar_scaled_sparse_discrete_lrsmith_n{n}',
        'source': 'matlab_gallery',
        'backend': 'scipy',
        'operator_form': 'sparse_matrixOperator',
        'data_regime': 'standard_sparse_discrete_stable',
        'auto_variant': auto_variant,
        'results': [
            {
                'variant': 'lrsmith',
                'elapsed_ms': smith_elapsed_ms,
                'rank': Z.left.shape[1],
                'iterations': smith.lastIterations,
                'relative_residual': float(np.linalg.norm(residual) / np.linalg.norm(q_data)),
                'relative_reference_error': float(np.linalg.norm(x_smith - x_ref) / np.linalg.norm(x_ref)),
            },
            {
                'variant': 'dense_reference',
                'elapsed_ms': dense_elapsed_ms,
                'rank': np.linalg.matrix_rank(x_ref),
                'iterations': 1,
                'relative_residual': 0.0,
                'relative_reference_error': 0.0,
            },
        ],
    }

    assert metrics['auto_variant'] == 'lrsmith'
    assert metrics['source'] in STANDARD_BENCHMARK_SOURCES
    smith_metric = metrics['results'][0]
    assert smith_metric['elapsed_ms'] >= 0.0
    assert smith_metric['rank'] > 0
    assert smith_metric['iterations'] > 0
    assert smith_metric['relative_residual'] < 1e-8
    assert smith_metric['relative_reference_error'] < 1e-8


@pytest.mark.benchmark
def test_standard_case_edge_benchmark_smoke_records_ill_conditioned_dense_case():
    A = matlab_gallery_hilbert(8)
    operator = matrixOperator(A, backendName='scipy')
    metric = _svd_metric('economic', operator, A, rank=8)

    assert 'matlab_gallery' in STANDARD_BENCHMARK_SOURCES
    assert np.linalg.cond(A) > 1e8
    assert metric['rank'] == 8
    assert metric['relative_error'] < 1e-10


@pytest.mark.benchmark
@pytest.mark.parametrize("n", [10, 100])
def test_pathological_diagonal_cases_cover_ill_conditioned_and_singular_spectra(n):
    ill = ill_conditioned_diagonal(n)
    singular = singular_diagonal(n, zero_columns=max(1, n // 10))

    ill_metric = _svd_metric('economic', matrixOperator(ill, backendName='scipy'), ill, rank=n)
    singular_metric = _svd_metric('economic', matrixOperator(singular, backendName='scipy'), singular, rank=n)

    assert np.linalg.cond(ill) > 1e6
    assert np.linalg.matrix_rank(singular) < n
    assert ill_metric['relative_error'] < 1e-10
    assert singular_metric['relative_error'] < 1e-10


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("case_name", "dense"),
    [
        ("ill_conditioned_n100_kappa1e24", ill_conditioned_diagonal(100, min_exp=-24.0)),
        ("singular_n100_zero50", singular_diagonal(100, zero_columns=50)),
        ("extreme_scaled_n100", extreme_scaled_diagonal(100)),
    ],
)
def test_special_diagonal_svd_dispatch_records_auto_vs_fastest(case_name, dense):
    operator = matrixOperator(dense, backendName='scipy')
    auto_variant = selectSVDVariant(backendName='scipy', xOperator=operator, rank=dense.shape[1])
    metrics = [_svd_metric(variant, operator, dense, rank=dense.shape[1]) for variant in ('static', 'economic', 'dual', 'qrSVD')]
    auto_metric = next(metric for metric in metrics if metric['variant'] == auto_variant)
    fastest_metric = min(metrics, key=lambda metric: metric['elapsed_ms'])

    dispatch_record = {
        'case': case_name,
        'auto_variant': auto_variant,
        'fastest_variant': fastest_metric['variant'],
        'auto_runtime_ratio': auto_metric['elapsed_ms'] / max(fastest_metric['elapsed_ms'], 1e-12),
    }

    assert dispatch_record['auto_variant'] in {'economic', 'dual', 'static', 'qrSVD'}
    assert auto_metric['relative_error'] < 1e-8
    assert fastest_metric['relative_error'] < 1e-8


@pytest.mark.benchmark
@pytest.mark.parametrize("m", [7, 10])
def test_composed_operator_svd_dispatch_records_auto_vs_fastest(m):
    c1 = matrix_depot_poisson_2d(m).toarray()
    c2 = matlab_gallery_grcar(m * m)
    operator = matrixOperator(c1, backendName='scipy') + matrixOperator(c2, backendName='scipy')
    target = c1 + c2
    rank = target.shape[1]
    auto_variant = selectSVDVariant(backendName='scipy', xOperator=operator, rank=rank)
    metrics = [_svd_metric(variant, operator, target, rank=rank) for variant in ('static', 'economic', 'dual', 'qrSVD')]
    auto_metric = next(metric for metric in metrics if metric['variant'] == auto_variant)
    fastest_metric = min(metrics, key=lambda metric: metric['elapsed_ms'])

    dispatch_record = {
        'case': f'poisson_plus_grcar_n{m*m}',
        'auto_variant': auto_variant,
        'fastest_variant': fastest_metric['variant'],
        'auto_runtime_ratio': auto_metric['elapsed_ms'] / max(fastest_metric['elapsed_ms'], 1e-12),
    }

    assert dispatch_record['auto_variant'] in {'economic', 'dual', 'static', 'qrSVD'}
    assert auto_metric['relative_error'] < 1e-8
    assert fastest_metric['relative_error'] < 1e-8


@pytest.mark.benchmark
def test_torch_cpu_gpu_smoke_match_on_supported_svd_case():
    if torch is None or not torch.cuda.is_available():
        pytest.skip("CUDA torch is not available")

    data_cpu = torch.randn((100, 10), dtype=torch.float64, device='cpu')
    data_gpu = data_cpu.to('cuda')
    cpu_metric = _svd_metric('economic', matrixOperator(data_cpu, backendName='torch'), data_cpu.cpu().numpy(), rank=10, backend_name='torch')
    gpu_metric = _svd_metric('economic', matrixOperator(data_gpu, backendName='torch'), data_gpu.cpu().numpy(), rank=10, backend_name='torch')

    assert abs(cpu_metric['relative_error'] - gpu_metric['relative_error']) < 1e-10


@pytest.mark.benchmark
def test_lyapunov_dispatch_smoke_covers_top_level_auto_branches():
    dense_A = matrixOperator(_stable_grcar_continuous(6), backendName='scipy')
    dense_B = matrixOperator(np.ones((6, 1)), backendName='scipy')
    sparse_A = matrixOperator(-matrix_depot_poisson_2d(3), backendName='scipy')
    sparse_B = matrixOperator(np.ones((9, 1)), backendName='scipy')
    discrete_A = matrixOperator(diags([0.2, 0.3, 0.4], format='csr'), backendName='scipy')

    assert selectLyapunovAlgorithm(variant='auto', A=dense_A, B=dense_B, backendName='scipy', isContinuous=True) == 'bartelsStewart'
    assert selectLyapunovAlgorithm(variant='auto', A=sparse_A, B=sparse_B, backendName='scipy', isContinuous=True) == 'bartelsStewart'
    assert selectLyapunovAlgorithm(variant='auto', A=discrete_A, B=matrixOperator(np.ones((3, 1)), backendName='scipy'), backendName='scipy', isContinuous=False) == 'lrsmith'
    with pytest.raises(NotImplementedError, match='discrete'):
        selectLyapunovAlgorithm(variant='auto', A=dense_A, B=dense_B, backendName='scipy', isContinuous=False)
