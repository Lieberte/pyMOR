"""Exercise the reduction API independently of the upstream data pipeline.

Example (from repository root):
    .venv/Scripts/python.exe scripts/check_reduction_methods.py --rounds 3
    .venv/Scripts/python.exe scripts/check_reduction_methods.py --motor-data

The optional local motor case selects up to 64 state coordinates from the
numeric inputs array. It never loads object arrays and does not normalize the
data. These are training reconstruction checks, not generalization claims.
"""

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mor.shonta import executeWorkflow, loadAutoencoderModel, loadPodModel
from mor.workflow.storage import atomicArray, atomicJson


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--motor-data', action='store_true')
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()
    if not 1 <= options.rounds <= 20:
        parser.error('--rounds must be between 1 and 20')
    repository = Path(__file__).resolve().parents[1]
    motorSource = repository / 'applicationTests' / 'shonTAmotor' / 'aeResult.npz'
    if options.motor_data and not motorSource.is_file():
        parser.error('--motor-data requires applicationTests/shonTAmotor/aeResult.npz; '
                     'regenerate the local motor data with applicationTests/shonTAmotor/nn.py first')
    output = (options.output or repository / 'reports' / f'reduction-methods-{uuid4().hex[:12]}').resolve()
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(7)
    basis, _ = np.linalg.qr(rng.normal(size=(6, 2)))
    datasets = {'seeded-rank2': basis @ rng.uniform(-1, 1, (2, 32))}
    sources = {'seeded-rank2': {'kind': 'synthetic', 'seed': 7, 'trueRank': 2}}
    if options.motor_data:
        source = motorSource
        with np.load(source, allow_pickle=False) as archive:
            inputs = archive['inputs']
            states = np.linspace(0, inputs.shape[1] - 1, min(64, inputs.shape[1]), dtype=int)
            datasets['motor-subsample'] = inputs[:, states].T.copy()
        sources['motor-subsample'] = {'kind': 'repository-case', 'path': str(source), 'array': 'inputs',
                                      'selectedStateIndices': states.tolist(), 'preprocessing': 'none'}
    report = {'metricScope': 'training reconstruction only', 'rounds': options.rounds, 'datasets': []}
    allPassed = True
    for name, data in datasets.items():
        source = output / f'{name}.npy'
        atomicArray(source, data)
        datasetReport = {'name': name, 'shape': list(data.shape), 'source': sources[name], 'runs': []}
        baseline = {}
        for roundIndex in range(options.rounds):
            for method, parameters, loader in (
                ('pod', {'rank': 2, 'centerSnapshots': True, 'verbose': False}, loadPodModel),
                ('autoencoder', {'latentDim': 2, 'hiddenDim': 16, 'epochs': 80,
                                 'batchSize': 32, 'learningRate': 0.01, 'seed': 7, 'verbose': False},
                 loadAutoencoderModel),
            ):
                taskId = uuid4().hex
                request = {'apiVersion': 1, 'taskId': taskId, 'caseId': name, 'nodeId': 'reduction-only',
                           'method': method, 'parameters': parameters, 'outputDirectory': str(output / taskId),
                           'inputs': {'snapshots': {'path': str(source), 'format': 'npy',
                                                    'axes': ['state', 'snapshot']}}}
                events = []
                result = executeWorkflow(request, events.append)
                run = {'round': roundIndex + 1, 'method': method, 'result': result,
                       'eventCount': len(events), 'passed': result['state'] == 'succeeded'}
                if run['passed']:
                    model = loader(result['modelReference'])
                    coefficients = model.project(data)
                    reconstructed = model.reconstruct(coefficients)
                    relativeError = float(np.linalg.norm(reconstructed - data) / max(np.linalg.norm(data), 1e-30))
                    run['reloadReconstructionError'] = relativeError
                    run['coefficientShape'] = list(coefficients.shape)
                    run['passed'] = bool(np.all(np.isfinite(reconstructed)) and reconstructed.shape == data.shape)
                    if method == 'autoencoder':
                        run['passed'] &= result['summary']['finalLoss'] < result['summary']['initialLoss']
                    if method in baseline:
                        run['repeatMaxDifference'] = float(np.max(np.abs(reconstructed - baseline[method])))
                        run['passed'] &= run['repeatMaxDifference'] == 0
                    else:
                        baseline[method] = reconstructed
                allPassed &= run['passed']
                datasetReport['runs'].append(run)
        report['datasets'].append(datasetReport)
    report['passed'] = allPassed
    atomicJson(output / 'report.json', report)
    print(json.dumps({'passed': allPassed, 'report': str(output / 'report.json'),
                      'runs': options.rounds * len(datasets) * 2}))
    return 0 if allPassed else 1


if __name__ == '__main__':
    raise SystemExit(main())
