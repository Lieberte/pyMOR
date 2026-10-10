from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pod import prepareSnapshots
from .storage import atomicArray, atomicJson, fileDigest, readJsonObject, validateCommittedTask
from .validation import MAX_INPUT_BYTES, MAX_HEADER_BYTES, MAX_ESTIMATED_PEAK_BYTES


def errorMetrics(reference, prediction):
    difference = np.abs(prediction - reference)
    if not np.all(np.isfinite(difference)):
        raise ValueError('model prediction produced non-finite errors')
    norm = np.linalg.norm(reference)
    return {'MAE': float(np.mean(difference)), 'MSE': float(np.mean(difference ** 2)),
            'RMSE': float(np.sqrt(np.mean(difference ** 2))), 'maxError': float(np.max(difference)),
            'relativeError': float(np.linalg.norm(difference) / norm) if norm else None}


def prepareSystem(request, directory, checkpoint):
    system = request['inputs']['system']
    matrices, frozen = {}, {}
    totalBytes = 0
    for key, reference in system['matrices'].items():
        target = directory / key
        target.mkdir()
        matrix, immutable = prepareSnapshots({**request, 'inputs': {'snapshots': reference}}, target, checkpoint)
        matrices[key] = matrix
        immutable['frozenFile'] = key + '/snapshots.npy'
        frozen[key] = immutable
        totalBytes += matrix.nbytes
        if totalBytes * 20 > MAX_ESTIMATED_PEAK_BYTES:
            raise ValueError('system matrices exceed the estimated peak-memory limit')
    n, m, p = matrices['A'].shape[0], matrices['B'].shape[1], matrices['C'].shape[0]
    if (n * n + p * m) * 8 > MAX_INPUT_BYTES:
        raise ValueError('default descriptor and feedthrough matrices exceed the input-memory limit')
    matrices.setdefault('D', np.zeros((p, m)))
    matrices.setdefault('E', np.eye(n))
    shapes = {'A': (n, n), 'B': (n, m), 'C': (p, n), 'D': (p, m), 'E': (n, n)}
    for key, shape in shapes.items():
        if matrices[key].shape != shape:
            raise ValueError(f'{key} dimensions must be {shape}')
    if request['parameters']['order'] is not None and request['parameters']['order'] > n:
        raise ValueError('BT order cannot exceed the full state dimension')
    if np.linalg.matrix_rank(matrices['E']) != n:
        raise ValueError('LTI descriptor E must be nonsingular')
    from scipy.linalg import eigvals
    poles = eigvals(matrices['A'], matrices['E'])
    stable = np.all(poles.real < 0) if system['isContinuous'] else np.all(np.abs(poles) < 1)
    if not np.all(np.isfinite(poles)) or not stable:
        raise ValueError('Balanced truncation requires a stable LTI system')
    request['inputs']['system'] = {**system, 'matrices': frozen}
    return matrices


def _persist(request, directory, arrays, summary, checkpoint, **metadata):
    descriptors = {}
    for key, value in arrays.items():
        checkpoint()
        array = np.asarray(value)
        if array.dtype.kind not in 'fc' or not np.all(np.isfinite(array)):
            raise ValueError(f'invalid model array: {key}')
        descriptors[key] = atomicArray(directory / f'{key}.npy', array)
    if 'metrics' in summary:
        atomicJson(directory / 'metrics.json', {'scope': summary['metricScope'], **summary['metrics']})
    manifest = {'apiVersion': 1, 'modelVersion': 1, 'kind': request['method'],
                'taskId': request['taskId'], 'nodeId': request['nodeId'], 'backend': 'scipy', 'device': 'cpu',
                'arrays': descriptors, 'parameters': request['parameters'], 'inputs': request['inputs'],
                'summary': summary, **metadata}
    return manifest, summary


def reduceDynamics(request, data, directory, checkpoint, setStage, log):
    from mor.operators import matrixOperator
    from mor.reductors import dmdReductor, podDmdReductor

    sequence = request['inputs']['sequence']
    pairs = np.asarray(sequence.get('pairs', [[i, i + 1] for i in range(data.shape[1] - 1)]), dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0 or np.max(pairs) >= data.shape[1]:
        raise ValueError('time-series pairs must reference existing training snapshots')
    x, y = data[:, pairs[:, 0]], data[:, pairs[:, 1]]
    parameters = request['parameters']
    for key in ('rank', 'podRank', 'dmdRank'):
        rank = parameters.get(key)
        limit = min(x.shape) if key != 'podRank' else min(x.shape[0], x.shape[1] * 2)
        if rank is not None and rank > limit:
            raise ValueError(f'{key} exceeds the training-pair dimensions')
    setStage('reducing')
    checkpoint()
    operator = lambda array: matrixOperator(array, backendName='scipy')
    if request['method'] == 'dmd':
        result = dmdReductor('scipy').reduce(operator(x), shiftedSnapshots=operator(y),
                                            rank=parameters['rank'], tol=parameters['tol'], dt=sequence['dt'])
    else:
        result = podDmdReductor('scipy', centerSnapshots=parameters['centerSnapshots']).reduce(
            operator(x), shiftedSnapshots=operator(y), podRank=parameters['podRank'], podTol=parameters['podTol'],
            dmdRank=parameters['dmdRank'], dmdTol=parameters['dmdTol'], dt=sequence['dt'])
    checkpoint()
    modes = np.asarray(result.modes.toBackendData())
    mean = np.zeros(data.shape[0])
    if request['method'] == 'pod-dmd' and result.snapshotMean is not None:
        mean = np.asarray(result.snapshotMean).reshape(-1)
    arrays = {'modes': modes, 'eigenvalues': np.asarray(result.eigenvalues), 'amplitudes': np.asarray(result.amplitudes),
              'reducedOperator': np.asarray(result.reducedOperator.toBackendData()),
              'singularValues': np.asarray(result.singularValues), 'mean': mean}
    if request['method'] == 'pod-dmd':
        arrays['podBasis'] = np.asarray(result.basis.toBackendData())
    prediction = modes @ (arrays['eigenvalues'][:, None] * (np.linalg.pinv(modes) @ (x - mean[:, None]))) + mean[:, None]
    metrics = errorMetrics(y, prediction)
    summary = {'method': request['method'], 'actualRank': int(result.rank), 'shape': list(data.shape),
               'trainingPairs': len(pairs), 'dt': sequence['dt'], 'metricScope': 'training-one-step-prediction',
               'metrics': metrics, 'reconstructionError': None}
    if request['method'] == 'pod-dmd':
        summary['podRank'] = int(result.podRank)
    setStage('saving')
    return _persist(request, directory, arrays, summary, checkpoint, dt=sequence['dt'])


def reduceBalanced(request, matrices, directory, checkpoint, setStage, log):
    from mor.models.lti import ltiModel
    from mor.operators import matrixOperator
    from mor.reductors.balancedTruncation import balancedTruncationReductor
    from scipy.sparse import csc_matrix

    system = request['inputs']['system']
    setStage('reducing')
    checkpoint()
    operators = {key: matrixOperator(value, backendName='scipy') for key, value in matrices.items()}
    if not system['isContinuous']:
        operators['A'] = matrixOperator(csc_matrix(np.linalg.solve(matrices['E'], matrices['A'])), backendName='scipy')
        operators['B'] = matrixOperator(np.linalg.solve(matrices['E'], matrices['B']), backendName='scipy')
        operators['E'] = None
    model = ltiModel(**operators,
                     backendName='scipy', isContinuous=system['isContinuous'])
    parameters = request['parameters']
    result = balancedTruncationReductor('scipy').reduce(model, order=parameters['order'], maxError=parameters['maxError'])
    checkpoint()
    arrays = {name: np.asarray(getattr(result, name + 'r').toBackendData()) for name in ('A', 'B', 'C', 'D', 'E')}
    arrays['hsv'] = np.asarray(result.hsv, dtype=np.float64)
    summary = {'method': 'bt', 'actualRank': int(result.order), 'fullOrder': matrices['A'].shape[0],
               'isContinuous': system['isContinuous'], 'reconstructionError': None,
               'hankelSingularValues': arrays['hsv'].tolist(), 'metricScope': 'no-response-validation-data'}
    setStage('saving')
    return _persist(request, directory, arrays, summary, checkpoint,
                    isContinuous=system['isContinuous'], dt=system.get('dt'))


@dataclass(frozen=True)
class DynamicsModel:
    arrays: dict
    manifest: dict

    def project(self, states):
        from .pod import PodModel
        data = PodModel._matrix(states, self.arrays['modes'].shape[0], 'states')
        return np.linalg.pinv(self.arrays['modes']) @ (data - self.arrays['mean'][:, None])

    def reconstruct(self, coordinates):
        data = np.asarray(coordinates)
        if data.ndim != 2 or data.shape[0] != len(self.arrays['eigenvalues']) or not np.all(np.isfinite(data)):
            raise ValueError('invalid modal coordinates')
        return self.arrays['modes'] @ data + self.arrays['mean'][:, None]

    def step(self, states):
        return self.reconstruct(self.arrays['eigenvalues'][:, None] * self.project(states))


@dataclass(frozen=True)
class BalancedModel:
    arrays: dict
    manifest: dict

    def step(self, states, inputs, dt=None):
        from scipy.linalg import expm
        from .pod import PodModel
        a = np.linalg.solve(self.arrays['E'], self.arrays['A'])
        b = np.linalg.solve(self.arrays['E'], self.arrays['B'])
        x = PodModel._matrix(states, a.shape[0], 'states')
        u = PodModel._matrix(inputs, b.shape[1], 'inputs')
        if x.shape[1] != u.shape[1]:
            raise ValueError('state and input sample counts must match')
        if self.manifest['isContinuous']:
            if type(dt) not in (int, float) or not np.isfinite(dt) or dt <= 0:
                raise ValueError('continuous systems require a positive dt for zero-order-hold stepping')
            n, m = b.shape
            augmented = np.zeros((n + m, n + m))
            augmented[:n, :n], augmented[:n, n:] = a, b
            transition = expm(augmented * dt)
            a, b = transition[:n, :n], transition[:n, n:]
        elif dt is not None and dt != self.manifest.get('dt'):
            raise ValueError('dt must match the discrete model sampling interval')
        return a @ x + b @ u

    def output(self, states, inputs):
        from .pod import PodModel
        x = PodModel._matrix(states, self.arrays['C'].shape[1], 'states')
        u = PodModel._matrix(inputs, self.arrays['D'].shape[1], 'inputs')
        if x.shape[1] != u.shape[1]:
            raise ValueError('state and input sample counts must match')
        return self.arrays['C'] @ x + self.arrays['D'] @ u


def loadDynamicsModel(manifestPath):
    path = Path(manifestPath).resolve()
    if path.name != 'manifest.json' or path.stat().st_size > 1024 * 1024:
        raise ValueError('expected a bounded manifest.json')
    manifest = readJsonObject(path)
    kind = manifest.get('kind')
    if (type(manifest.get('apiVersion')) is not int or manifest['apiVersion'] != 1 or
            type(manifest.get('modelVersion')) is not int or manifest['modelVersion'] != 1 or
            kind not in ('dmd', 'pod-dmd', 'bt')):
        raise ValueError('unsupported model kind or version')
    validateCommittedTask(manifest, path.parent)
    expected = {'A', 'B', 'C', 'D', 'E', 'hsv'} if kind == 'bt' else {
        'modes', 'eigenvalues', 'amplitudes', 'reducedOperator', 'singularValues', 'mean'}
    if kind == 'pod-dmd':
        expected.add('podBasis')
    descriptors = manifest.get('arrays')
    if not isinstance(descriptors, dict) or set(descriptors) != expected:
        raise ValueError('model array set is incomplete')
    arrays = {}
    totalBytes = 0
    for key in expected:
        descriptor = descriptors[key]
        arrayPath = (path.parent / f'{key}.npy').resolve()
        if not isinstance(descriptor, dict) or descriptor.get('file') != key + '.npy' or arrayPath.parent != path.parent:
            raise ValueError('invalid model array path')
        if arrayPath.stat().st_size > MAX_INPUT_BYTES + MAX_HEADER_BYTES or fileDigest(arrayPath) != descriptor.get('sha256'):
            raise ValueError('model array checksum or size mismatch')
        data = np.load(arrayPath, allow_pickle=False, max_header_size=MAX_HEADER_BYTES)
        totalBytes += data.nbytes
        if (totalBytes > 3 * MAX_INPUT_BYTES or data.dtype.kind not in 'fc' or
                list(data.shape) != descriptor.get('shape') or str(data.dtype) != descriptor.get('dtype') or not np.all(np.isfinite(data))):
            raise ValueError('invalid model array contents')
        data.setflags(write=False)
        arrays[key] = data
    rank = manifest.get('summary', {}).get('actualRank')
    if type(rank) is not int or rank < 1:
        raise ValueError('invalid model rank')
    if kind == 'bt':
        a, b, c, d, e = (arrays[key] for key in ('A', 'B', 'C', 'D', 'E'))
        if (a.shape != (rank, rank) or e.shape != a.shape or b.ndim != 2 or c.ndim != 2 or b.shape[0] != rank or
                c.shape[1] != rank or d.shape != (c.shape[0], b.shape[1]) or type(manifest.get('isContinuous')) is not bool):
            raise ValueError('inconsistent system model dimensions')
        if np.linalg.matrix_rank(e) != rank:
            raise ValueError('singular reduced descriptor')
        if not manifest['isContinuous'] and (type(manifest.get('dt')) not in (int, float) or not np.isfinite(manifest['dt']) or manifest['dt'] <= 0):
            raise ValueError('invalid discrete sampling interval')
        return BalancedModel(arrays, manifest)
    modes = arrays['modes']
    if (modes.ndim != 2 or modes.shape[1] != rank or arrays['mean'].shape != (modes.shape[0],) or
            arrays['eigenvalues'].shape != (rank,) or arrays['amplitudes'].shape != (rank,) or arrays['reducedOperator'].shape != (rank, rank)):
        raise ValueError('inconsistent DMD model dimensions')
    if type(manifest.get('dt')) not in (int, float) or not np.isfinite(manifest['dt']) or manifest['dt'] <= 0:
        raise ValueError('invalid DMD sampling interval')
    return DynamicsModel(arrays, manifest)
