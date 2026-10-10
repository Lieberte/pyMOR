import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .storage import atomicArray, fileDigest, readJsonObject, validateCommittedTask
from .validation import MAX_ESTIMATED_PEAK_BYTES, MAX_HEADER_BYTES, MAX_INPUT_BYTES


def prepareSnapshots(request, directory, checkpoint):
    import numpy as np

    source = Path(request['inputs']['snapshots']['path'])
    target = directory / 'snapshots.npy'
    digest = hashlib.sha256()
    copied = 0
    with source.open('rb') as incoming, target.open('xb') as outgoing:
        before = os.fstat(incoming.fileno())
        while True:
            checkpoint()
            block = incoming.read(1024 * 1024)
            if not block:
                break
            copied += len(block)
            if copied > MAX_INPUT_BYTES + MAX_HEADER_BYTES:
                raise ValueError('snapshot file exceeds the configured input size limit')
            digest.update(block)
            outgoing.write(block)
        after = os.fstat(incoming.fileno())
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns or copied != before.st_size:
        raise ValueError('snapshot file changed while creating the immutable task input')
    reference = request['inputs']['snapshots']
    if 'sha256' in reference and reference['sha256'].lower() != digest.hexdigest():
        raise ValueError('snapshot content does not match the requested SHA-256 digest')
    checkpoint()
    mapped = np.load(target, mmap_mode='r', allow_pickle=False, max_header_size=MAX_HEADER_BYTES)
    try:
        if not isinstance(mapped, np.ndarray) or mapped.ndim != 2 or min(mapped.shape) < 1:
            raise ValueError('snapshots must be a nonempty 2D array with axes [state, snapshot]')
        if mapped.dtype.kind not in 'fiu' or mapped.dtype.itemsize > 8:
            raise ValueError('snapshots must contain real numeric values convertible to float64')
        if 'shape' in reference and reference['shape'] != list(mapped.shape):
            raise ValueError('snapshot shape does not match the input reference')
        if 'dtype' in reference and np.dtype(reference['dtype']) != mapped.dtype:
            raise ValueError('snapshot dtype does not match the input reference')
        sourceDtype = str(mapped.dtype)
        floatBytes = int(mapped.size) * 8
        estimatedPeak = 10 * floatBytes
        if request['method'] == 'pod':
            estimatedPeak += 3 * min(mapped.shape) ** 2 * 8
        if floatBytes > MAX_INPUT_BYTES or estimatedPeak > MAX_ESTIMATED_PEAK_BYTES:
            raise ValueError('snapshot shape exceeds the float64 input or estimated peak-memory limit')
        if request['method'] == 'pod':
            rank = request['parameters']['rank']
            if rank is not None and rank > min(mapped.shape):
                raise ValueError('rank must not exceed min(state count, snapshot count)')
            if request['parameters']['centerSnapshots'] and mapped.shape[1] < 2:
                raise ValueError('centered POD requires at least two snapshots')
        data = np.array(mapped, dtype=np.float64, order='C', copy=True)
    finally:
        if hasattr(mapped, '_mmap') and mapped._mmap is not None:
            mapped._mmap.close()
        elif hasattr(mapped, 'close'):
            mapped.close()
    if not np.all(np.isfinite(data)):
        raise ValueError('snapshots must contain only finite values')
    checkpoint()
    frozenInput = {
        **request['inputs']['snapshots'],
        'frozenFile': target.name,
        'sha256': digest.hexdigest(),
        'sourceBytes': copied,
        'sourceMtimeNs': before.st_mtime_ns,
        'shape': list(data.shape),
        'dtype': sourceDtype,
        'computationDtype': str(data.dtype),
        'estimatedPeakBytes': estimatedPeak,
    }
    return data, frozenInput


def reducePod(request, data, directory, checkpoint, setStage, log):
    import platform

    import mor
    import numpy as np
    import scipy

    from mor.operators import matrixOperator
    from mor.reductors import podReductor

    parameters = request['parameters']
    setStage('reducing')
    log('Computing CPU POD; cancellation is checked when the decomposition returns.')
    checkpoint()
    result = podReductor('scipy', centerSnapshots=parameters['centerSnapshots']).reduce(
        matrixOperator(data, backendName='scipy'), rank=parameters['rank'], tol=parameters['tol'],
    )
    checkpoint()
    basis = np.asarray(result.basis.toBackendData(), dtype=np.float64)
    singularValues = np.asarray(result.S, dtype=np.float64).reshape(-1)
    mean = np.zeros(data.shape[0], dtype=np.float64) if result.snapshotMean is None else np.asarray(
        result.snapshotMean, dtype=np.float64,
    ).reshape(-1)
    if (basis.ndim != 2 or basis.shape != (data.shape[0], result.order) or
            singularValues.shape != (result.order,) or mean.shape != (data.shape[0],)):
        raise ValueError('POD returned inconsistent model dimensions')
    if not all(np.all(np.isfinite(array)) for array in (basis, singularValues, mean)):
        raise ValueError('POD returned non-finite model arrays')
    error = float(result.reconstructionError) if result.reconstructionError is not None else None
    if error is not None and not np.isfinite(error):
        raise ValueError('POD returned a non-finite reconstruction error')
    setStage('saving')
    arrays = {}
    for key, array, filename in [('basis', basis, 'basis.npy'), ('mean', mean, 'mean.npy'),
                                 ('singularValues', singularValues, 'singular-values.npy')]:
        checkpoint()
        arrays[key] = atomicArray(directory / filename, array)
    checkpoint()
    summary = {
        'method': 'pod', 'requestedRank': parameters['rank'], 'actualRank': int(result.order),
        'discardedEnergyTolerance': parameters['tol'], 'shape': list(data.shape),
        'centerSnapshots': parameters['centerSnapshots'], 'reconstructionError': error,
        'solverInfo': result.solverInfo,
    }
    from .dynamics import errorMetrics
    from .storage import atomicJson
    metrics = errorMetrics(data, basis @ (basis.T @ (data - mean[:, None])) + mean[:, None])
    summary.update(metrics=metrics, metricScope='training-reconstruction')
    atomicJson(directory / 'metrics.json', {'scope': 'training-reconstruction', **metrics})
    manifest = {
        'apiVersion': 1, 'modelVersion': 1, 'kind': 'pod', 'taskId': request['taskId'],
        'nodeId': request['nodeId'], 'backend': 'scipy', 'device': 'cpu', 'weighted': False,
        'arrays': arrays, 'summary': summary,
        'input': request['inputs']['snapshots'],
        'parameters': parameters,
        'preprocessing': {'centerSnapshots': parameters['centerSnapshots'], 'axes': ['state', 'snapshot'],
                          'dtype': 'float64'},
        'versions': {'python': platform.python_version(), 'mor': mor.__version__,
                     'numpy': np.__version__, 'scipy': scipy.__version__},
    }
    log(f'POD produced a basis with {result.order} columns.')
    return manifest, summary


@dataclass(frozen=True)
class PodModel:
    basis: Any
    mean: Any
    singularValues: Any
    manifest: dict

    def project(self, snapshots):
        data = self._matrix(snapshots, self.basis.shape[0], 'snapshots')
        return self.basis.T @ (data - self.mean[:, None])

    def reconstruct(self, coefficients):
        data = self._matrix(coefficients, self.basis.shape[1], 'coefficients')
        return self.basis @ data + self.mean[:, None]

    @staticmethod
    def _matrix(value, rows, name):
        import numpy as np

        raw = np.asarray(value)
        if raw.dtype.kind not in 'fiu' or raw.ndim != 2 or raw.shape[0] != rows:
            raise ValueError(f'{name} must be a real 2D matrix with {rows} rows')
        data = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(data)):
            raise ValueError(f'{name} must contain only finite values')
        return data


def loadPodModel(manifestPath):
    import numpy as np

    path = Path(manifestPath).resolve()
    if path.name != 'manifest.json' or path.stat().st_size > 1024 * 1024:
        raise ValueError('expected a bounded manifest.json model file')
    manifest = readJsonObject(path)
    if type(manifest.get('apiVersion')) is not int or manifest['apiVersion'] != 1 or type(manifest.get('modelVersion')) is not int or manifest['modelVersion'] != 1 or manifest.get('kind') != 'pod':
        raise ValueError('unsupported POD model manifest version or kind')
    if not all(isinstance(manifest.get(key), dict) for key in ('arrays', 'summary', 'preprocessing')):
        raise ValueError('POD model manifest requires arrays, summary, and preprocessing objects')
    validateCommittedTask(manifest, path.parent)
    if manifest.get('weighted') is not False or manifest.get('device') != 'cpu':
        raise ValueError('only real CPU unweighted POD models are supported')
    preprocessing = manifest['preprocessing']
    if (type(preprocessing.get('centerSnapshots')) is not bool or preprocessing.get('axes') != ['state', 'snapshot'] or
            preprocessing.get('dtype') != 'float64'):
        raise ValueError('unsupported or incomplete POD preprocessing metadata')
    if manifest['summary'].get('centerSnapshots') is not preprocessing['centerSnapshots']:
        raise ValueError('POD model centering metadata is inconsistent')
    arrays = {}
    totalBytes = 0
    for key, filename in [('basis', 'basis.npy'), ('mean', 'mean.npy'), ('singularValues', 'singular-values.npy')]:
        descriptor = manifest['arrays'].get(key)
        if not isinstance(descriptor, dict) or descriptor.get('file') != filename:
            raise ValueError('model array path is invalid')
        arrayPath = (path.parent / filename).resolve()
        if arrayPath.parent != path.parent or arrayPath.stat().st_size > MAX_INPUT_BYTES + MAX_HEADER_BYTES:
            raise ValueError('model array exceeds the permitted directory or size')
        if fileDigest(arrayPath) != descriptor.get('sha256'):
            raise ValueError(f'model array checksum mismatch: {key}')
        mapped = np.load(arrayPath, mmap_mode='r', allow_pickle=False, max_header_size=MAX_HEADER_BYTES)
        try:
            if not isinstance(mapped, np.ndarray) or mapped.dtype.kind != 'f' or mapped.dtype.itemsize != 8:
                raise ValueError('model arrays must be float64')
            totalBytes += int(mapped.nbytes)
            if totalBytes > 3 * MAX_INPUT_BYTES:
                raise ValueError('model arrays exceed the permitted memory limit')
            if list(mapped.shape) != descriptor.get('shape') or str(mapped.dtype) != descriptor.get('dtype'):
                raise ValueError('model array metadata does not match the stored array')
            arrays[key] = np.array(mapped, copy=True)
        finally:
            if hasattr(mapped, '_mmap') and mapped._mmap is not None:
                mapped._mmap.close()
            elif hasattr(mapped, 'close'):
                mapped.close()
        if not np.all(np.isfinite(arrays[key])):
            raise ValueError('model arrays must contain only finite values')
    basis, mean, values = arrays['basis'], arrays['mean'], arrays['singularValues']
    if (basis.ndim != 2 or basis.shape[0] < 1 or mean.shape != (basis.shape[0],) or
            values.shape != (basis.shape[1],) or np.any(values < 0)):
        raise ValueError('POD model array dimensions or singular values are invalid')
    if type(manifest['summary'].get('actualRank')) is not int or manifest['summary']['actualRank'] != basis.shape[1]:
        raise ValueError('POD model rank metadata is inconsistent')
    shape = manifest['summary'].get('shape')
    if not isinstance(shape, list) or len(shape) != 2 or any(type(size) is not int or size < 1 for size in shape) or shape[0] != basis.shape[0]:
        raise ValueError('POD model state dimension metadata is inconsistent')
    if manifest['preprocessing'].get('centerSnapshots') is False and np.any(mean != 0):
        raise ValueError('uncentered POD models must have a zero mean')
    for array in arrays.values():
        array.setflags(write=False)
    return PodModel(basis=basis, mean=mean, singularValues=values, manifest=manifest)
