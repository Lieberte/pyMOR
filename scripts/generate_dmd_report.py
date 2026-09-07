from __future__ import annotations

import csv
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import mor  # noqa: F401
from mor.algorithm.auto.dmd import selectDMDVariant
from mor.algorithm.registry import algorithmRegistry
from mor.operators import matrixOperator
from report_run_utils import reset_report_dir, write_case_manifest
from report_svg_utils import write_bar_chart_svg, write_success_matrix_svg


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / 'reports' / 'dmd'
JSON_PATH = REPORT_DIR / 'dmd_results.json'
CSV_PATH = REPORT_DIR / 'dmd_results.csv'
MD_PATH = REPORT_DIR / 'DMD_REPORT.md'
ERROR_SVG_PATH = REPORT_DIR / 'dmd_eigenvalue_error.svg'
RUNTIME_SVG_PATH = REPORT_DIR / 'dmd_runtime.svg'
SUCCESS_SVG_PATH = REPORT_DIR / 'dmd_success_matrix.svg'


@dataclass
class DMDCase:
    name: str
    family: str
    dynamics: np.ndarray
    initialState: np.ndarray
    snapshots: int
    noise: float = 0.0


def trajectory(operator, initialState, steps):
    states = [initialState]
    for _ in range(steps - 1):
        states.append(operator @ states[-1])
    return np.column_stack(states)


def build_cases():
    return [
        DMDCase('diagonal_decay', 'decay', np.diag([0.8, 0.95]), np.array([1.0, 0.3]), 30),
        DMDCase(
            'conjugate_oscillation', 'oscillation',
            np.array([[0.92, -0.2], [0.2, 0.92]]), np.array([1.0, 0.3]), 30,
        ),
        DMDCase(
            'nonnormal', 'nonnormal', np.array([[0.85, 0.4], [0.0, 0.9]]),
            np.array([1.0, 0.3]), 30,
        ),
        DMDCase(
            'noisy_oscillation', 'noisy', np.array([[0.92, -0.15], [0.15, 0.92]]),
            np.array([1.0, 0.3]), 40, noise=2e-3,
        ),
        DMDCase(
            'rank_deficient', 'rank_deficient', np.diag([0.9, 0.7]),
            np.array([1.0, 0.0]), 20,
        ),
    ]


def eigenvalue_error(actual, expected, count):
    unmatched = list(actual)
    errors = []
    for value in expected[:count]:
        index = int(np.argmin(np.abs(np.asarray(unmatched) - value)))
        errors.append(abs(unmatched.pop(index) - value))
    return float(max(errors, default=0.0))


def evaluate(case):
    clean = trajectory(case.dynamics, case.initialState, case.snapshots)
    rng = np.random.default_rng(31)
    snapshots = clean + case.noise * rng.standard_normal(clean.shape)
    operator = matrixOperator(snapshots, 'scipy')
    variants = ['exact', 'tls', 'forwardBackward'] if case.noise else ['exact']
    if case.family == 'rank_deficient':
        variants = ['exact']
    autoVariant = selectDMDVariant('scipy', noiseAware=case.noise > 0)
    results = []
    expected = np.linalg.eigvals(case.dynamics)
    for variant in variants:
        start = time.perf_counter()
        try:
            result = algorithmRegistry.get('dmd', variant, backendName='scipy').fit(
                operator, rank=2
            )
            elapsedMs = (time.perf_counter() - start) * 1000.0
            rows = result.reconstruct(realOutput=False)
            results.append({
                'variant': variant,
                'success': True,
                'elapsed_ms': round(elapsedMs, 6),
                'rank': result.rank,
                'eigenvalue_error': eigenvalue_error(result.eigenvalues, expected, result.rank),
                'one_step_error': result.diagnostics['oneStepResidual'],
                'reconstruction_error': float(np.linalg.norm(rows - clean) / np.linalg.norm(clean)),
                'failure_reason': '',
            })
        except Exception as error:
            results.append({
                'variant': variant,
                'success': False,
                'elapsed_ms': round((time.perf_counter() - start) * 1000.0, 6),
                'rank': 0,
                'eigenvalue_error': float('inf'),
                'one_step_error': float('inf'),
                'reconstruction_error': float('inf'),
                'failure_reason': f'{type(error).__name__}: {error}',
            })
    return {
        'case': case.name,
        'family': case.family,
        'backend': 'scipy',
        'operator_form': 'matrixOperator',
        'data_regime': case.family,
        'shape': list(snapshots.shape),
        'noise_standard_deviation': case.noise,
        'auto_variant': autoVariant,
        'results': results,
    }


def flatten(cases):
    return [
        {
            'case': case['case'],
            'family': case['family'],
            'backend': case['backend'],
            'operator_form': case['operator_form'],
            'data_regime': case['data_regime'],
            'shape': 'x'.join(str(value) for value in case['shape']),
            'noise_standard_deviation': case['noise_standard_deviation'],
            'auto_variant': case['auto_variant'],
            **result,
        }
        for case in cases for result in case['results']
    ]


def write_outputs(payload, rows):
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    with CSV_PATH.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    successful = [row for row in rows if row['success']]
    write_bar_chart_svg(
        successful, ERROR_SVG_PATH, 'DMD Eigenvalue Error', 'eigenvalue_error',
        'Lower is better; noise-aware variants are reported separately.', 'case',
        log10=True, color='#256d7b',
    )
    write_bar_chart_svg(
        successful, RUNTIME_SVG_PATH, 'DMD Runtime', 'elapsed_ms',
        'Local characterization time in milliseconds.', 'case', color='#7b5d73',
    )
    cases = [case['case'] for case in payload['cases']]
    variants = sorted({row['variant'] for row in rows})
    successMap = {(row['variant'], row['case']): row['success'] for row in rows}
    write_success_matrix_svg(variants, cases, successMap, SUCCESS_SVG_PATH, 'DMD Success Matrix')
    summary = [
        '# DMD Report', '', f"Generated: `{payload['generated_at']}`", '', '## Scope', '',
        'Exact, TLS, and forward-backward DMD are characterized on deterministic, noisy,',
        'nonnormal, oscillatory, and rank-deficient snapshot sequences.', '', '## Results', '',
    ]
    for case in payload['cases']:
        for result in case['results']:
            summary.append(
                f"- `{case['case']}` / `{result['variant']}`: rank={result['rank']}, "
                f"eigenvalue error={result['eigenvalue_error']:.3e}, "
                f"reconstruction error={result['reconstruction_error']:.3e}, "
                f"elapsed={result['elapsed_ms']:.3f} ms"
            )
    summary.extend([
        '', '## Dispatch', '',
        'Exact DMD remains the default. TLS-DMD is selected only when `noiseAware=True`;',
        'controlled, delayed, bidirectional, and optimized variants likewise require explicit intent.',
        '', 'Irregular training times are intentionally rejected; uniform sampling uses a positive `dt`.',
        '', '## Artifacts', '',
        f'- `{JSON_PATH.name}`', f'- `{CSV_PATH.name}`', f'- `{ERROR_SVG_PATH.name}`',
        f'- `{RUNTIME_SVG_PATH.name}`', f'- `{SUCCESS_SVG_PATH.name}`', '',
    ])
    MD_PATH.write_text('\n'.join(summary), encoding='utf-8-sig')


def main():
    reset_report_dir(REPORT_DIR)
    cases = [evaluate(case) for case in build_cases()]
    rows = flatten(cases)
    payload = {
        'phase': 'dmd',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'environment': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'numpy': np.__version__,
        },
        'cases': cases,
    }
    write_outputs(payload, rows)
    write_case_manifest(REPORT_DIR, 'dmd', [
        {
            'case': case.name,
            'family': case.family,
            'source_name': 'local-generated linear dynamics',
            'state_dimension': case.dynamics.shape[0],
            'snapshots': case.snapshots,
            'noise_standard_deviation': case.noise,
        }
        for case in build_cases()
    ])
    print(f'Wrote {REPORT_DIR}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
