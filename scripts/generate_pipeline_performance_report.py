from __future__ import annotations

import argparse
import ast
import gc
import csv
import importlib
import json
import platform
import statistics
import sys
import tempfile
import time
import tracemalloc
import types
from pathlib import Path

import numpy as np

from mor.operators import matrixOperator


def loadBaseline(module, root):
    if root is None:
        return None
    path = root / Path(*module.__name__.split('.')).with_suffix('.py')
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    tree.body = [node for node in tree.body if not isinstance(node, ast.Expr)]
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            node.decorator_list = [decorator for decorator in node.decorator_list
                                   if not (isinstance(decorator, ast.Call)
                                           and isinstance(decorator.func, ast.Name)
                                           and decorator.func.id == 'registerAlgorithm')]
    baseline = types.ModuleType(module.__name__ + '_baseline')
    baseline.__package__ = module.__package__
    sys.modules[baseline.__name__] = baseline
    exec(compile(tree, str(path), 'exec'), baseline.__dict__)
    return baseline


def measure(operation, repeats):
    operation()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        times.append((time.perf_counter() - start) * 1000)
    tracemalloc.start()
    try:
        result = operation()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return result, {'medianMs': statistics.median(times), 'peakAllocationBytes': peak}


def compareMeasurements(baseline, optimized, repeats):
    baseline()
    optimized()
    samples = [[], []]
    operations = [baseline, optimized]
    for repeat in range(repeats):
        for index in ([0, 1] if repeat % 2 == 0 else [1, 0]):
            start = time.perf_counter()
            operations[index]()
            samples[index].append((time.perf_counter() - start) * 1000)
    results = []
    for operation, times in zip(operations, samples):
        gc.collect()
        tracemalloc.start()
        try:
            result = operation()
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        results.append((result, {'medianMs': statistics.median(times), 'peakAllocationBytes': peak}))
    return results


def benchmarkNodes(baselineRoot, repeats, scratch):
    geometry = importlib.import_module('mor.nn.data.motorFemGeometry')
    pod = importlib.import_module('mor.reductors.pod')
    dual = importlib.import_module('mor.algorithm.decompose.svd.dualSVD')
    storage = importlib.import_module('mor.workflow.storage')
    workflow = importlib.import_module('mor.workflow.autoencoder')
    trainer = importlib.import_module('mor.nn.trainers.autoEncoderTrainerTorch')
    validation = importlib.import_module('mor.nn.validation.reconstructionMetrics')
    scipyBackend = importlib.import_module('mor.backends.scipyBackend')
    lradi = importlib.import_module('mor.algorithm.lyapunov.lradi')
    smith = importlib.import_module('mor.algorithm.lyapunov.lrsmith')
    incremental = importlib.import_module('mor.algorithm.decompose.svd.incrementalStandardSVD')
    modules = [geometry, pod, dual, storage, workflow, trainer, validation, scipyBackend, lradi, smith, incremental]
    previous = {module.__name__: loadBaseline(module, baselineRoot) for module in modules}
    rng = np.random.default_rng(7)
    cases = []
    weights = rng.uniform(0.1, 2., 160000)
    groups = np.arange(weights.size) % 512
    cases.append(('group_sampling', geometry,
                  lambda m: m.sampleByGroupWithConservedWeight(weights, groups, 16000, 7),
                  lambda a, b: (np.testing.assert_array_equal(a[0], b[0]),
                                np.testing.assert_allclose(a[1], b[1], rtol=1e-13))))
    coordinates = np.arange(30000, dtype=float)[:, None] * np.array([[2., 0., 0.]])
    nodes = (coordinates[:, None, :] + np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])).reshape(-1, 3)
    faces = np.arange(len(nodes)).reshape(-1, 3)
    cases.append(('boundary_geometry', geometry,
                  lambda m: m.boundaryNodeGeometry(nodes, faces, np.zeros(3)),
                  lambda a, b: [np.testing.assert_allclose(x, y, atol=1e-12) for x, y in zip(a, b)]))
    cancelledNodes = nodes[:4500].copy()
    cancelledNodes[:, 2] = 0
    originalFaces = np.arange(len(cancelledNodes)).reshape(-1, 3)
    cancelledFaces = np.vstack((originalFaces, originalFaces[:, ::-1]))
    cases.append(('boundary_cancelled_normals', geometry,
                  lambda m: m.boundaryNodeGeometry(cancelledNodes, cancelledFaces, np.zeros(3)),
                  lambda a, b: [np.testing.assert_allclose(x, y, atol=1e-12) for x, y in zip(a, b)]))
    data = rng.normal(size=(2048, 512))
    basis = np.linalg.qr(rng.normal(size=(2048, 16)))[0]
    snapshots = matrixOperator(data, 'scipy')

    def diagnostic(m):
        reductor = m.podReductor('scipy')
        reductor.updateOptions(snapshots)
        if m is not pod:
            reductor.localBackend = previous[scipyBackend.__name__].scipyBackend()
        return reductor._reconstructionError(snapshots, basis, None, None)

    cases.append(('pod_diagnostic', pod, diagnostic,
                  lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-12)))
    wide = matrixOperator(rng.normal(size=(48, 512)), 'scipy')

    def wideSvd(m):
        U, S, Vt = m.dualSVD('scipy').decompose(wide, rank=12)
        return S, (U * S) @ Vt

    cases.append(('wide_dual_svd', dual, wideSvd,
                  lambda a, b: [np.testing.assert_allclose(x, y, atol=1e-10) for x, y in zip(a, b)]))
    from scipy.sparse import diags
    shiftOperator = matrixOperator(diags(-np.linspace(1., 10., 8192), format='csr'), 'scipy')
    factors = [rng.normal(size=(8192, 4)) for _ in range(32)]
    shiftRhs = factors[-1][:, :2]

    def updateShifts(m):
        return m.lradi('scipy').updateShifts(shiftOperator, None, shiftRhs, factors,
                                           np.array([-1.]), m.shiftComputationOptions())

    cases.append(('lradi_shift_update', lradi, updateShifts,
                  lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-12)))
    streamBasis = np.linalg.qr(rng.normal(size=(256, 12)))[0]
    stream = [streamBasis @ rng.normal(size=(12, 32)) for _ in range(12)]

    def streamSvd(m):
        algorithm = m.incrementalStandardSVD('scipy')
        for batch in stream:
            U, S, Vh = algorithm.decompose(matrixOperator(batch, 'scipy'), rank=12)
        return (U * S) @ Vh

    cases.append(('incremental_svd', incremental, streamSvd,
                  lambda a, b: np.testing.assert_allclose(a, b, atol=1e-10)))
    descriptor = matrixOperator(np.diag(np.linspace(1., 2., 96)), 'scipy')
    discreteA = matrixOperator(np.diag(np.linspace(0.1, 0.6, 96)) @ descriptor.toBackendData(), 'scipy')
    discreteB = matrixOperator(np.ones((96, 1)), 'scipy')

    def descriptorSmith(m):
        algorithm = m.lrsmith('scipy', maxIter=80, tol=1e-12, checkStability=False)
        result = algorithm.solve(discreteA, descriptor, discreteB)
        assert algorithm.lastConverged
        return result.toBackendData()

    cases.append(('lrsmith_descriptor', smith, descriptorSmith,
                  lambda a, b: np.testing.assert_allclose(a, b, atol=1e-10)))
    array = rng.normal(size=(2048, 1024))

    def saveArray(m):
        result = m.atomicArray(scratch / 'array.npy', array)
        return result['sha256']

    cases.append(('array_storage', storage, saveArray, lambda a, b: np.testing.assert_equal(a, b)))
    import torch
    from mor.nn.data.snapshotDataModule import snapshotDataModule
    from mor.nn.losses.mseReconstructionLoss import mseReconstructionLoss
    from mor.nn.models.autoEncoderTorch import autoEncoderTorch
    torch.manual_seed(7)
    model = autoEncoderTorch(64, 8, [16]).eval()
    trainingInputs = torch.from_numpy(rng.normal(size=(1024, 64)).astype(np.float32))
    dataModule = snapshotDataModule(trainingInputs, trainingInputs, batchSize=16, shuffle=False)

    def validateEpoch(m):
        instance = m.autoEncoderTrainerTorch(mseReconstructionLoss())
        with torch.no_grad():
            return instance._runEpoch(model, dataModule.getValidationLoader(), dataModule, 'cpu', torch.float32)[0]

    cases.append(('trainer_evaluation', trainer, validateEpoch,
                  lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-12)))
    cases.append(('reconstruction_metrics', validation,
                  lambda m: m.reconstructionMetrics(deviceName='cpu', lossFunction=mseReconstructionLoss()).evaluate(model, dataModule),
                  lambda a, b: [np.testing.assert_allclose(a[key], b[key], rtol=1e-6, atol=1e-12) for key in a]))
    if torch.cuda.is_available():
        cudaModel = autoEncoderTorch(64, 8, [16]).to('cuda').eval()
        cudaModel.load_state_dict(model.state_dict())
        cudaData = snapshotDataModule(trainingInputs, trainingInputs, batchSize=16,
                                     shuffle=False, pinMemory=True)

        def cudaEpoch(m):
            torch.cuda.synchronize()
            with torch.no_grad():
                result = m.autoEncoderTrainerTorch(mseReconstructionLoss())._runEpoch(
                    cudaModel, cudaData.getValidationLoader(), cudaData, 'cuda', torch.float32)[0]
            torch.cuda.synchronize()
            return result

        def cudaMetrics(m):
            torch.cuda.synchronize()
            result = m.reconstructionMetrics(deviceName='cuda', lossFunction=mseReconstructionLoss()).evaluate(cudaModel, cudaData)
            torch.cuda.synchronize()
            return result

        cases.append(('trainer_evaluation_cuda', trainer, cudaEpoch,
                      lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-7)))
        cases.append(('reconstruction_metrics_cuda', validation, cudaMetrics,
                      lambda a, b: [np.testing.assert_allclose(a[key], b[key], rtol=1e-6, atol=1e-12) for key in a]))
    trainingData = rng.normal(size=(4, 16))
    parameters = dict(latentDim=2, hiddenDim=8, epochs=400, batchSize=16, seed=7,
                      learningRate=0.001, verbose=False)

    def train(m):
        writes = []
        original = m.atomicJson

        def record(path, payload):
            if path.name == 'training-history.json':
                writes.append(len(payload['loss']))
            return original(path, payload)

        m.atomicJson = record
        try:
            _, summary = m.reduceAutoencoder({'parameters': parameters, 'taskId': 'benchmark',
                                             'nodeId': 'benchmark', 'inputs': {'snapshots': {}}},
                                            trainingData, scratch, lambda: None,
                                            lambda stage: None, lambda message: None,
                                            lambda metrics: None)
        finally:
            m.atomicJson = original
        return {'loss': summary['finalLoss'], 'historyWrites': len(writes),
                'historyEntriesSerialized': sum(writes)}

    cases.append(('workflow_training', workflow, train,
                  lambda a, b: np.testing.assert_allclose(a['loss'], b['loss'], rtol=1e-12)))
    rows = []
    for name, module, operation, compare in cases:
        baseline = previous[module.__name__]
        if baseline is not None:
            (reference, baselineMetrics), (optimized, optimizedMetrics) = compareMeasurements(
                lambda: operation(baseline), lambda: operation(module), repeats)
        else:
            optimized, optimizedMetrics = measure(lambda: operation(module), repeats)
        row = {'node': name, 'backend': 'torch_cuda' if name.endswith('_cuda') else
               ('torch_cpu' if name in {'trainer_evaluation', 'reconstruction_metrics', 'workflow_training'} else 'scipy'),
               'optimized': optimizedMetrics, 'validation': 'not_compared' if baseline is None else 'passed'}
        if baseline is not None:
            compare(reference, optimized)
            row['baseline'] = baselineMetrics
            row['speedup'] = baselineMetrics['medianMs'] / optimizedMetrics['medianMs']
            row['allocationRatio'] = baselineMetrics['peakAllocationBytes'] / max(optimizedMetrics['peakAllocationBytes'], 1)
            if name == 'workflow_training':
                row['baselineHistory'] = reference
                row['optimizedHistory'] = optimized
        rows.append(row)
        print(name, json.dumps(row, ensure_ascii=False), flush=True)
    return rows


def writeCharts(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    comparisons = [row for row in rows if 'baseline' in row]
    if not comparisons:
        return []
    names = [row['node'] for row in comparisons]
    paths = []
    for field, filename, label in [('speedup', 'speedup.svg', 'Time before / after'),
                                   ('allocationRatio', 'allocation.svg', 'Peak allocation before / after')]:
        values = [row[field] for row in comparisons]
        fig, ax = plt.subplots(figsize=(10, max(4, len(comparisons) * 0.4)))
        ax.barh(names, values, color='#256d7b')
        ax.axvline(1., color='#777777', linestyle='--', linewidth=1)
        ax.set_xscale('log')
        ax.set_xlim(min(0.8, min(values) * 0.7), max(values) * 2)
        ax.set_xlabel(label + ' (log scale; >1 is better)')
        ax.invert_yaxis()
        for index, value in enumerate(values):
            ax.text(value * 1.05, index, f'{value:.2f}x', va='center', fontsize=9)
        fig.tight_layout()
        fig.savefig(output / filename)
        plt.close(fig)
        paths.append(filename)
    return paths


def writeReport(payload, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = payload['cases']
    (output / 'performance.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    flat = [{'node': row['node'], 'baselineMs': row.get('baseline', {}).get('medianMs'),
             'optimizedMs': row['optimized']['medianMs'], 'speedup': row.get('speedup'),
             'baselineAllocationBytes': row.get('baseline', {}).get('peakAllocationBytes'),
             'optimizedAllocationBytes': row['optimized']['peakAllocationBytes']} for row in rows]
    with (output / 'performance.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = ['# Pipeline node performance', '',
             'Local medians; each optimized output is compared with its baseline when provided.', '',
             f"Environment: {json.dumps(payload['environment'])}. Repeats: {payload['repeats']}.", '',
             '| Node | Before ms | After ms | Speedup | Peak allocation before/after MiB |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        before = row.get('baseline', {})
        beforeMs = f"{before['medianMs']:.3f}" if before else 'n/a'
        speedup = f"{row['speedup']:.2f}x" if before else 'n/a'
        beforeMemory = f"{before['peakAllocationBytes'] / 1024**2:.2f}" if before else 'n/a'
        lines.append(f"| {row['node']} | {beforeMs} | {row['optimized']['medianMs']:.3f} | {speedup} | "
                     f"{beforeMemory} / {row['optimized']['peakAllocationBytes'] / 1024**2:.2f} |")
    for filename in writeCharts(rows, output):
        lines.extend(['', f'![{filename}]({filename})'])
    lines.extend(['', payload['memoryMetric'], '',
                  'CUDA cases synchronize before and after each operation. Other cases use CPU. '
                  'These measurements do not establish whole-application speedups.'])
    (output / 'PERFORMANCE.md').write_text('\n'.join(lines) + '\n', encoding='utf-8-sig')


def main():
    parser = argparse.ArgumentParser(description='Measure representative pipeline nodes and validate outputs.')
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--output', type=Path, default=Path('reports/pipeline_performance'))
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    from threadpoolctl import threadpool_limits
    limit = threadpool_limits(limits=1)
    blasThreads = 1
    import torch
    previousThreads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with limit, tempfile.TemporaryDirectory(prefix='pymor-node-benchmark-') as temporary:
            rows = benchmarkNodes(args.baseline_root, args.repeats, Path(temporary))
    finally:
        torch.set_num_threads(previousThreads)
    payload = {'baselineRoot': str(args.baseline_root.resolve()) if args.baseline_root else None,
               'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                               'numpy': np.__version__, 'torch': torch.__version__,
                               'cudaAvailable': torch.cuda.is_available(), 'blasThreads': blasThreads,
                               'torchThreads': 1},
               'repeats': args.repeats, 'cases': rows,
               'memoryMetric': 'tracemalloc peak; excludes native Torch allocations and process RSS'}
    writeReport(payload, args.output)
    print('Report:', args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
