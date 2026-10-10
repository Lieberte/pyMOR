"""Independent numerical characterization and native process/CUDA memory evidence."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import diags
from threadpoolctl import threadpool_limits

from mor.operators import matrixOperator


def processMemory():
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('peak', 'rss', 'peakPaged', 'paged',
                                                    'peakNonpaged', 'nonpaged', 'pagefile', 'peakPagefile')]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {'peakRssBytes': counters.peak, 'currentRssBytes': counters.rss}
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'peakRssBytes': int(peak * (1 if sys.platform == 'darwin' else 1024))}


def lyapunovSign(device):
    import torch
    from mor.algorithm.lyapunov.sign import sign
    rows = []
    for spread in (2, 6, 12):
        rates = np.geomspace(10 ** (-spread / 2), 10 ** (spread / 2), 64)
        a, b = -np.diag(rates), np.random.default_rng(7).normal(size=(64, 4))
        aa, bb = torch.tensor(a, device=device), torch.tensor(b, device=device)
        solver = sign('torch', tol=1e-12)
        x = solver.solve(matrixOperator(aa, 'torch'), None, matrixOperator(bb, 'torch')).toNumpy()
        residual = float(np.linalg.norm(a @ x + x @ a.T + b @ b.T) / np.linalg.norm(b @ b.T))
        assert residual < 1e-6, residual
        rows.append({'spread': spread, 'relativeResidual': residual, 'iterations': solver.lastIterations,
                     'fallback': solver.lastFallback, 'signResidual': solver.lastRelativeResidual})
    return {'device': device, 'cases': rows}


def multiInputLradi():
    from mor.algorithm.lyapunov.lradi import lradi
    rows = []
    n = 512
    rates = np.geomspace(1, 100, n)
    for inputs in (1, 8, 32):
        b = np.random.default_rng(7).normal(size=(n, inputs))
        solver = lradi('scipy', maxIter=120, tol=1e-10, compressionColumns=128)
        result = solver.solve(matrixOperator(diags(-rates), 'scipy'), None, matrixOperator(b, 'scipy'))
        x = result.toNumpy()
        q = b @ b.T
        residual = float(np.linalg.norm(-rates[:, None] * x - x * rates[None, :] + q) / np.linalg.norm(q))
        assert solver.lastConverged and residual < 1e-8, (inputs, residual)
        rows.append({'inputs': inputs, 'n': n, 'factorColumns': result.left.shape[1],
                     'iterations': solver.lastIterations, 'relativeResidual': residual})
    return {'cases': rows}


def frequencySweep():
    from mor.models.lti import ltiModel
    from mor.reductors.frequencyResponse import frequencyResponseReductor
    rows = []
    for continuous in (True, False):
        n = 128
        rates = np.geomspace(1e-2, 1e3, n) if continuous else np.linspace(0.1, 0.999, n)
        a = -rates if continuous else rates
        rng = np.random.default_rng(7)
        b, c = rng.normal(size=(n, 2)), rng.normal(size=(2, n))
        model = ltiModel(matrixOperator(diags(a), 'scipy'), matrixOperator(b, 'scipy'),
                         matrixOperator(c, 'scipy'), backendName='scipy', isContinuous=continuous)
        train = np.geomspace(1e-3, 1e4, 24) if continuous else np.linspace(0, np.pi, 24)
        evaluation = np.geomspace(1.1e-3, 9.9e3, 193) if continuous else np.linspace(0.001, np.pi - 0.001, 193)
        points = 1j * evaluation if continuous else np.exp(1j * evaluation)
        expected = np.stack([c @ (b / (point - a)[:, None]) for point in points])
        full = model.frequencyResponse(evaluation)
        np.testing.assert_allclose(full, expected, atol=1e-10, rtol=1e-10)
        for order in (8, 24):
            rom = frequencyResponseReductor('scipy', frequencies=train).reduce(model, order=order)
            actual = rom.model.frequencyResponse(evaluation)
            relative = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
            assert np.isfinite(relative)
            rows.append({'continuous': continuous, 'n': n, 'order': order, 'trainingCount': len(train),
                         'independentEvaluationCount': len(evaluation), 'relativeFrequencyError': relative})
    return {'cases': rows}


def dmdNoise():
    from mor.algorithm.dmd import exactDMD, tlsDMD, forwardBackwardDMD, dmdc
    dynamics = np.array([[0.99, -0.08], [0.08, 0.99]])
    states = [np.array([1., 0.2])]
    for _ in range(79):
        states.append(dynamics @ states[-1])
    clean = np.column_stack(states)
    spectrum = np.linalg.eigvals(dynamics)
    rows = []
    for structure in ('white', 'correlated'):
        for level in (0., 1e-3, 1e-2, 0.1):
            errors = {name: [] for name in ('exact', 'tls', 'forwardBackward')}
            for seed in range(8):
                noise = np.random.default_rng(seed).normal(size=clean.shape)
                if structure == 'correlated':
                    for t in range(1, noise.shape[1]):
                        noise[:, t] = 0.8 * noise[:, t - 1] + 0.6 * noise[:, t]
                snapshots = matrixOperator(clean + level * noise, 'scipy')
                for name, cls in [('exact', exactDMD), ('tls', tlsDMD), ('forwardBackward', forwardBackwardDMD)]:
                    result = cls('scipy').fit(snapshots, rank=2)
                    values = result.eigenvalues
                    error = min(np.linalg.norm(values - spectrum), np.linalg.norm(values[::-1] - spectrum))
                    assert np.isfinite(error)
                    errors[name].append(float(error))
            rows.extend({'noise': level, 'structure': structure, 'method': name, 'seeds': 8,
                         'spectrumErrorMean': float(np.mean(values)), 'spectrumErrorMax': max(values)}
                        for name, values in errors.items())
    excitation = []
    for active in (False, True):
        rng = np.random.default_rng(7)
        controls = rng.normal(size=(1, 79)) if active else np.zeros((1, 79))
        states = [np.ones(2)]
        for control in controls.T:
            states.append(dynamics @ states[-1] + np.array([[1.], [0.4]]) @ control)
        snapshots = np.column_stack(states)
        for noise in (0., 0.01):
            result = dmdc('scipy').fit(matrixOperator(snapshots + noise * rng.normal(size=snapshots.shape), 'scipy'),
                                       matrixOperator(controls, 'scipy'))
            assert result.diagnostics['identifiable'] == active
            excitation.append({'excited': active, 'noise': noise, **result.diagnostics})
    reverse = []
    for epsilon in (1e-2, 1e-8, 0.):
        try:
            result = forwardBackwardDMD('scipy').fit(matrixOperator(np.eye(2), 'scipy'),
                matrixOperator(np.diag([0.8, epsilon]), 'scipy'), rank=2)
            reverse.append({'smallEigenvalue': epsilon, 'status': 'finite', 'residual': result.diagnostics['oneStepResidual']})
        except ValueError as error:
            reverse.append({'smallEigenvalue': epsilon, 'status': 'rejected', 'reason': str(error)})
    return {'noiseCases': rows, 'excitationCases': excitation, 'reverseCases': reverse}


def hankelMemory(delay):
    from mor.algorithm.dmd import hankelDMD
    t = np.arange(256)
    signal = np.cos(t * 0.1) + 0.3 * np.sin(t * 0.2)
    data = np.arange(1, 65)[:, None] * signal
    result = hankelDMD('scipy').fit(matrixOperator(data, 'scipy'), delay=delay, rank=4,
                                    maxAugmentedBytes=64 * 1024 ** 2)
    error = float(np.linalg.norm(result.reconstruct() - data) / np.linalg.norm(data))
    # Short delay has too few scalar temporal dimensions to recover four modes.
    assert np.isfinite(error)
    return {'n': 64, 'steps': 256, 'delay': delay, 'reconstructionError': error,
            'augmentedBytes': result.diagnostics['augmentedBytes']}


def backendAssembly(name):
    from mor.operators import assembleLinearTetraThermalOperators, applyDirichletConditions
    backendName = 'scipy' if name == 'assembly_scipy' else 'torch'
    device = 'cuda' if name.endswith('cuda') else 'cpu'
    width = 8
    nodes = np.indices((width + 1,) * 3).reshape(3, -1).T.astype(float)
    ids = np.arange(len(nodes)).reshape((width + 1,) * 3)
    offsets = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
               (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    corners = np.stack([ids[i:i + width, j:j + width, k:k + width].reshape(-1)
                        for i, j, k in offsets], axis=1)
    pattern = np.array([[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
                        [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]])
    elements = corners[:, pattern].reshape(-1, 4)
    def synchronize():
        if device == 'cuda':
            import torch
            torch.cuda.synchronize()
    synchronize()
    start = time.perf_counter()
    result = assembleLinearTetraThermalOperators(nodes, elements, conductivity=2., density=3.,
        specificHeat=4., heatSource=6., assemblyBatchSize=256, backendName=backendName,
        **({'deviceName': device} if backendName == 'torch' else {}))
    synchronize()
    assemblyMs = (time.perf_counter() - start) * 1000
    assert result.K.isSparse and result.M.isSparse
    # Conservation and constant-field nullspace are independent element checks.
    mass, load = result.M.toNumpy(), result.Q.toNumpy()
    stiffness = result.K.toNumpy()
    np.testing.assert_allclose(mass.sum(), width ** 3 * 12.)
    np.testing.assert_allclose(load.sum(), width ** 3 * 6.)
    np.testing.assert_allclose(stiffness @ np.ones(len(nodes)), 0., atol=1e-13)
    fixed = np.flatnonzero(nodes[:, 0] == 0)
    synchronize()
    start = time.perf_counter()
    constrained, shifted = applyDirichletConditions(result.K, result.Q, fixed, 300.)
    synchronize()
    dirichletMs = (time.perf_counter() - start) * 1000
    data = result.K.toBackendData()
    return {'backend': backendName, 'device': device, 'nodes': len(nodes), 'elements': len(elements),
            'nnz': int(data.nnz) if backendName == 'scipy' else data._nnz(),
            'format': 'csr' if backendName == 'scipy' else str(data.layout),
            'assemblyMs': assemblyMs, 'dirichletMs': dirichletMs,
            'constantFieldResidual': float(np.linalg.norm(stiffness @ np.ones(len(nodes)))),
            'dirichletSparse': constrained.isSparse, 'loadSum': float(shifted.toNumpy().sum()),
            'elementKernelBackend': result.metadata['elementKernelBackend']}


def worker(name):
    torchWorker = name.startswith('sign_') or name.startswith('assembly_torch_')
    cudaWorker = name.endswith('_cuda')
    if torchWorker:
        import torch
        torch.set_num_threads(1)
    if cudaWorker:
        if not torch.cuda.is_available():
            return {'status': 'unavailable', 'reason': 'CUDA unavailable'}
        torch.ones(1, device='cuda')
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    before = processMemory()
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        if name.startswith('sign_'):
            result = lyapunovSign(name.split('_')[1])
        elif name == 'lradi':
            result = multiInputLradi()
        elif name == 'frequency':
            result = frequencySweep()
        elif name == 'dmd':
            result = dmdNoise()
        elif name.startswith('hankel_'):
            result = hankelMemory(int(name.split('_')[1]))
        elif name.startswith('assembly_'):
            result = backendAssembly(name)
        else:
            raise ValueError(f'unknown scenario: {name}')
    if cudaWorker:
        torch.cuda.synchronize()
        result['cudaMemory'] = {'peakAllocatedBytes': torch.cuda.max_memory_allocated(),
                                'peakReservedBytes': torch.cuda.max_memory_reserved()}
    return {'status': 'passed', 'elapsedMs': (time.perf_counter() - start) * 1000,
            'memoryBefore': before, 'memoryAfter': processMemory(), **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/validation'))
    parser.add_argument('--worker', choices=['sign_cpu', 'sign_cuda', 'lradi', 'frequency', 'dmd',
                                             'hankel_2', 'hankel_32', 'hankel_128',
                                             'assembly_scipy', 'assembly_torch_cpu', 'assembly_torch_cuda'])
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker), allow_nan=False))
        return
    rows = []
    for name in parser._option_string_actions['--worker'].choices:
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', name],
                                    capture_output=True, text=True, timeout=180)
        if completed.returncode:
            raise RuntimeError(f'{name} characterization failed:\n{completed.stderr}')
        row = {'scenario': name, **json.loads(completed.stdout)}
        rows.append(row)
        print(name, row['status'], flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                              'numpy': np.__version__, 'threads': 1},
               'memoryMetric': 'Isolated child process peak RSS includes imports; CUDA allocated/reserved excludes driver memory.',
               'cases': rows}
    (args.output / 'validation.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    summary = [{'scenario': row['scenario'], 'status': row['status'], 'elapsedMs': row.get('elapsedMs'),
                'peakRssBytes': row.get('memoryAfter', {}).get('peakRssBytes'),
                'cudaPeakAllocatedBytes': row.get('cudaMemory', {}).get('peakAllocatedBytes')} for row in rows]
    with (args.output / 'validation.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = ['# 数值与原生内存表征', '', payload['memoryMetric'], '',
             '| 场景 | 状态 | ms | 进程峰值 MiB |', '| --- | --- | ---: | ---: |']
    for row in summary:
        lines.append(f"| {row['scenario']} | {row['status']} | {row['elapsedMs'] or 0:.2f} | {(row['peakRssBytes'] or 0) / 1024**2:.2f} |")
    lines.extend(['', '误差、噪声统计和回退详情见 validation.json。有限的降阶误差不表示达到应用精度目标。'])
    (args.output / 'VALIDATION.md').write_text('\n'.join(lines) + '\n', encoding='utf-8-sig')
    tex = ['\\documentclass{article}', '\\begin{document}', '\\begin{tabular}{lrr}', 'Scenario & ms & RSS MiB \\\\']
    for row in summary:
        tex.append(f"{row['scenario'].replace('_', ' ')} & {row['elapsedMs'] or 0:.2f} & {(row['peakRssBytes'] or 0)/1024**2:.2f} \\\\")
    tex.extend(['\\end{tabular}', '\\end{document}'])
    (args.output / 'validation.tex').write_text('\n'.join(tex), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar([row['scenario'] for row in summary], [(row['peakRssBytes'] or 0) / 1024**2 for row in summary])
    axis.set_ylabel('Isolated process peak RSS (MiB)')
    axis.tick_params(axis='x', rotation=35)
    figure.tight_layout()
    figure.savefig(args.output / 'native_memory.svg')
    plt.close(figure)


if __name__ == '__main__':
    main()
