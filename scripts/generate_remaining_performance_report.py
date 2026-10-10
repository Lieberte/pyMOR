from __future__ import annotations

import argparse
import importlib
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import diags
from threadpoolctl import threadpool_limits

from mor.operators import matrixOperator
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_pipeline_performance_report import loadBaseline, compareMeasurements, writeReport


def structuredMesh(width):
    grid = np.indices((width + 1,) * 3).reshape(3, -1).T.astype(float)
    ids = np.arange(len(grid)).reshape((width + 1,) * 3)
    offsets = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
               (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    corners = np.stack([ids[i:i + width, j:j + width, k:k + width].reshape(-1)
                        for i, j, k in offsets], axis=1)
    pattern = np.array([[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
                        [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]])
    return grid, corners[:, pattern].reshape(-1, 4)


def benchmarkCases(root, repeats, width=20):
    modules = [importlib.import_module(name) for name in
               ['mor.algorithm.lyapunov.lradi', 'mor.algorithm.lyapunov.lrsmith', 'mor.operators.thermal']]
    previous = {module.__name__: loadBaseline(module, root) for module in modules}
    adi, smith, thermal = modules
    a = matrixOperator(diags([np.ones(511) * 0.1, -np.linspace(1., 5., 512), np.ones(511) * 0.1],
                            [-1, 0, 1], format='csr'), 'scipy')
    b = matrixOperator(np.ones((512, 2)), 'scipy')
    def adiRun(module):
        solver = module.lradi('scipy', maxIter=30, tol=1e-30, compressionColumns=100)
        solver.computeInitialShifts = lambda *args: np.array([-1., -2.])
        solver.updateShifts = lambda *args, **kwargs: np.array([-1., -2.])
        result = solver.solve(a, None, b)
        return result.toBackendData()
    cases = [('lradi_repeated_shifts', adi, adiRun,
              {'n': 512, 'inputs': 2, 'shifts': [-1., -2.], 'iterations': 30, 'backend': 'scipy'})]
    descriptor = torch.diag(torch.linspace(1., 2., 128, dtype=torch.float64))
    aa = torch.diag(torch.linspace(0.1, 0.65, 128, dtype=torch.float64)) @ descriptor
    bb = torch.ones((128, 2), dtype=torch.float64)
    for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
        def smithRun(module, device=device):
            if device == 'cuda': torch.cuda.synchronize()
            result = module.lrsmith('torch', maxIter=50, tol=1e-12, checkStability=False).solve(
                matrixOperator(aa.to(device), 'torch'), matrixOperator(descriptor.to(device), 'torch'),
                matrixOperator(bb.to(device), 'torch')).toBackendData()
            if device == 'cuda': torch.cuda.synchronize()
            return result.cpu().numpy()
        cases.append(('torch_descriptor_' + device, smith, smithRun,
                      {'n': 128, 'inputs': 2, 'device': device, 'dtype': 'float64'}))
    nodes, elements = structuredMesh(width)
    for batch in [256, 50000]:
        def thermalRun(module, batch=batch):
            result = module.assembleLinearTetraThermalOperators(nodes, elements, conductivity=2.,
                density=3., specificHeat=4., heatSource=6., assemblyBatchSize=batch)
            return result
        cases.append(('fem_batch_' + str(batch), thermal, thermalRun,
                      {'nodes': len(nodes), 'elements': len(elements), 'batchSize': batch}))
    rows = []
    for name, module, operation, scenario in cases:
        (reference, before), (result, after) = compareMeasurements(
            lambda: operation(previous[module.__name__]), lambda: operation(module), repeats)
        if name.startswith('fem'):
            errors = []
            for field in ['conductivity', 'capacity']:
                ref, current = getattr(reference, field).toBackendData(), getattr(result, field).toBackendData()
                difference = current - ref
                errors.append(float(np.linalg.norm(difference.data) / np.linalg.norm(ref.data)))
            np.testing.assert_allclose(result.load.toNumpy(), reference.load.toNumpy())
            error = max(errors)
        else:
            error = float(np.linalg.norm(result - reference) / np.linalg.norm(reference))
        assert error < 1e-10, (name, error)
        row = {'node': name, 'scenario': scenario, 'relativeDifference': error, 'passed': True,
               'baseline': before, 'optimized': after, 'speedup': before['medianMs'] / after['medianMs'],
               'allocationRatio': before['peakAllocationBytes'] / max(after['peakAllocationBytes'], 1)}
        rows.append(row)
        print(name, row, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description='Compare remaining performance stages with saved baseline sources.')
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('reports/remaining_performance'))
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats <= 0: parser.error('repeats must be positive')
    oldThreads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            rows = benchmarkCases(args.baseline_root, args.repeats)
    finally:
        torch.set_num_threads(oldThreads)
    writeReport({'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                 'torch': torch.__version__, 'platform': platform.platform(),
                                 'blasThreads': 1, 'torchThreads': 1},
                 'baselineRoot': str(args.baseline_root), 'repeats': args.repeats, 'cases': rows,
                 'memoryMetric': 'tracemalloc peak; excludes native Torch allocations and process RSS'}, args.output)


if __name__ == '__main__':
    main()
