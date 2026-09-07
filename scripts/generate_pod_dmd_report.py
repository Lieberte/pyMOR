from __future__ import annotations

import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import mor  # noqa: F401
from mor.algorithm.dmd import exactDMD
from mor.operators import matrixOperator
from mor.reductors import podDmdReductor
from report_run_utils import reset_report_dir, write_case_manifest
from report_svg_utils import write_bar_chart_svg


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / 'reports' / 'pod_dmd'
JSON_PATH = REPORT_DIR / 'pod_dmd_results.json'
CSV_PATH = REPORT_DIR / 'pod_dmd_results.csv'
MD_PATH = REPORT_DIR / 'POD_DMD_REPORT.md'
ERROR_SVG_PATH = REPORT_DIR / 'pod_dmd_reconstruction_error.svg'
RUNTIME_SVG_PATH = REPORT_DIR / 'pod_dmd_runtime.svg'


def embeddedTrajectory(stateDimension, latentDimension, steps, seed):
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((stateDimension, latentDimension)))
    eigenvalues = np.linspace(0.72, 0.96, latentDimension)
    reduced = [rng.standard_normal(latentDimension)]
    for _ in range(steps - 1):
        reduced.append(eigenvalues * reduced[-1])
    return basis @ np.column_stack(reduced), eigenvalues


def evaluate(stateDimension, podRank, dmdRank, seed):
    snapshots, expectedEigenvalues = embeddedTrajectory(
        stateDimension, latentDimension=podRank, steps=32, seed=seed
    )
    operator = matrixOperator(snapshots, 'scipy')
    start = time.perf_counter()
    direct = exactDMD('scipy').fit(operator, rank=dmdRank)
    directElapsed = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    combined = podDmdReductor('scipy').reduce(
        operator, podRank=podRank, dmdRank=dmdRank
    )
    podDmdElapsed = (time.perf_counter() - start) * 1000.0
    selectedExpected = expectedEigenvalues[-dmdRank:]
    eigenvalueError = float(max(
        abs(actual - expected)
        for actual, expected in zip(
            np.sort(combined.eigenvalues.real), np.sort(selectedExpected)
        )
    ))
    acceptanceError = 1e-10 if dmdRank == podRank else 5e-3
    return {
        'case': f'embedded_n{stateDimension}_pod{podRank}_dmd{dmdRank}',
        'backend': 'scipy',
        'operator_form': 'matrixOperator',
        'data_regime': 'high_dimensional_low_rank_snapshots',
        'state_dimension': stateDimension,
        'snapshot_count': snapshots.shape[1],
        'pod_rank': combined.podRank,
        'dmd_rank': combined.dmdRank,
        'coordinate_reduction_ratio': combined.podRank / stateDimension,
        'pod_projection_error': combined.diagnostics['podProjectionError'],
        'reduced_dmd_reconstruction_error': combined.diagnostics['reducedDmdReconstructionError'],
        'full_reconstruction_error': combined.diagnostics['fullReconstructionError'],
        'direct_dmd_reconstruction_error': direct.diagnostics['reconstructionError'],
        'eigenvalue_error': eigenvalueError,
        'pod_dmd_elapsed_ms': round(podDmdElapsed, 6),
        'direct_dmd_elapsed_ms': round(directElapsed, 6),
        'acceptance_error': acceptanceError,
        'success': bool(combined.diagnostics['fullReconstructionError'] < acceptanceError),
    }


def writeOutputs(payload, rows):
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    with CSV_PATH.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_bar_chart_svg(
        rows, ERROR_SVG_PATH, 'POD-DMD Full Reconstruction Error',
        'full_reconstruction_error', 'Lower is better; logarithmic scale.',
        'case', log10=True, color='#256d7b',
    )
    write_bar_chart_svg(
        rows, RUNTIME_SVG_PATH, 'POD-DMD Runtime', 'pod_dmd_elapsed_ms',
        'Local end-to-end POD projection and DMD fit time.', 'case', color='#7b5d73',
    )
    lines = [
        '# POD-DMD Report', '', f"Generated: `{payload['generated_at']}`", '',
        '## Results', '',
    ]
    for row in rows:
        lines.append(
            f"- `{row['case']}`: POD rank={row['pod_rank']}, DMD rank={row['dmd_rank']}, "
            f"coordinate ratio={row['coordinate_reduction_ratio']:.3e}, "
            f"POD error={row['pod_projection_error']:.3e}, "
            f"reduced DMD error={row['reduced_dmd_reconstruction_error']:.3e}, "
            f"full error={row['full_reconstruction_error']:.3e}, "
            f"acceptance={row['acceptance_error']:.1e}, "
            f"elapsed={row['pod_dmd_elapsed_ms']:.3f} ms"
        )
    lines.extend([
        '', '## Contract', '',
        'POD and DMD ranks are selected independently. DMD is trained only in POD',
        'coordinates, while modes and predictions are lifted to the original state space.',
        'Centered and weighted POD use the same projection contract as the standalone POD reductor.',
        '', 'Controlled data is intentionally rejected until a dedicated POD-DMDc result is added.',
        '', '## Artifacts', '', f'- `{JSON_PATH.name}`', f'- `{CSV_PATH.name}`',
        f'- `{ERROR_SVG_PATH.name}`', f'- `{RUNTIME_SVG_PATH.name}`', '',
    ])
    MD_PATH.write_text('\n'.join(lines), encoding='utf-8-sig')


def main():
    reset_report_dir(REPORT_DIR)
    configurations = [(80, 4, 4, 61), (200, 4, 4, 62), (500, 6, 4, 63)]
    rows = [evaluate(*configuration) for configuration in configurations]
    payload = {
        'phase': 'pod_dmd',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'environment': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'numpy': np.__version__,
        },
        'results': rows,
    }
    writeOutputs(payload, rows)
    write_case_manifest(REPORT_DIR, 'pod_dmd', [
        {
            'case': row['case'],
            'source_name': 'local-generated embedded linear dynamics',
            'state_dimension': row['state_dimension'],
            'snapshot_count': row['snapshot_count'],
            'pod_rank': row['pod_rank'],
            'dmd_rank': row['dmd_rank'],
        }
        for row in rows
    ])
    print(f'Wrote {REPORT_DIR}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
