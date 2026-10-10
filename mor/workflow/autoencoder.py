"""CPU autoencoder reduction on already prepared state-by-snapshot matrices.

The adapter adds task cancellation, metrics and portable arrays around the
repository's autoEncoderTorch model. It deliberately performs no data split,
normalization or centering; those belong to the upstream data pipeline.
"""

from pathlib import Path

from .storage import atomicArray, atomicJson, fileDigest, jsonCopy, readJsonObject, validateCommittedTask
from .validation import MAX_ESTIMATED_PEAK_BYTES, MAX_HEADER_BYTES, MAX_INPUT_BYTES

MAX_MODEL_PARAMETERS = 8_000_000


def _architecture(stateDim, latentDim, hiddenDim):
    if any(type(value) is not int or value < 1 for value in (stateDim, latentDim, hiddenDim)):
        raise ValueError('autoencoder dimensions must be positive integers')
    if latentDim > stateDim:
        raise ValueError('latentDim must not exceed the state count')
    if latentDim > 4096 or hiddenDim > 4096:
        raise ValueError('autoencoder hidden and latent dimensions exceed the supported limit')
    count = 2 * hiddenDim * (stateDim + latentDim) + 2 * hiddenDim + stateDim + latentDim
    if count > MAX_MODEL_PARAMETERS:
        raise ValueError('autoencoder architecture exceeds the model parameter limit')
    return count


def _newModel(stateDim, latentDim, hiddenDim, seed=0):
    import torch

    from mor.nn.models.autoEncoderTorch import autoEncoderTorch

    # nn.Linear initialization consumes the global CPU generator. Restore it so
    # a workflow task does not alter unrelated clients of the embedded runtime.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return autoEncoderTorch(stateDim, latentDim, [hiddenDim]).to(device='cpu', dtype=torch.float32)


def reduceAutoencoder(request, data, directory, checkpoint, setStage, log, emitMetrics):
    import platform

    import mor
    import numpy as np
    import torch

    parameters = request['parameters']
    stateDim, sampleCount = data.shape
    parameterCount = _architecture(stateDim, parameters['latentDim'], parameters['hiddenDim'])
    if sampleCount < 2:
        raise ValueError('autoencoder training requires at least two snapshots')
    batchSize = min(parameters['batchSize'], sampleCount)
    estimatedPeak = (int(data.nbytes) * 4 + parameterCount * 32 +
                     batchSize * (stateDim + 4 * parameters['hiddenDim'] + parameters['latentDim']) * 32)
    if estimatedPeak > MAX_ESTIMATED_PEAK_BYTES:
        raise ValueError('autoencoder training exceeds the estimated peak-memory limit')
    if np.max(np.abs(data)) > np.finfo(np.float32).max:
        raise ValueError('autoencoder snapshots exceed the finite float32 range')
    checkpoint()
    tensor = torch.from_numpy(np.array(data.T, dtype=np.float32, order='C', copy=True))
    model = _newModel(stateDim, parameters['latentDim'], parameters['hiddenDim'], parameters['seed'])
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learningRate'])
    generator = torch.Generator(device='cpu').manual_seed(parameters['seed'])

    def evaluateLoss():
        total = 0.0
        absoluteTotal = 0.0
        model.eval()
        with torch.no_grad():
            for start in range(0, sampleCount, batchSize):
                checkpoint()
                batch = tensor[start:start + batchSize]
                # Use float64 for reported losses and reconstruction diagnostics.
                difference = model(batch).to(torch.float64) - batch.to(torch.float64)
                total += float(torch.sum(difference.square()).item())
                absoluteTotal += float(torch.sum(difference.abs()).item())
        return total / tensor.numel(), absoluteTotal / tensor.numel()

    setStage('training')
    initialLoss, initialMae = evaluateLoss()
    if not np.isfinite(initialLoss):
        raise ValueError('autoencoder produced a non-finite initial loss; rescale training snapshots')
    log(f'Training CPU autoencoder with {parameterCount} parameters for {parameters["epochs"]} epochs.')
    history = []
    maeHistory = []
    def saveHistory():
        atomicJson(directory / 'training-history.json', {'initialLoss': initialLoss, 'initialMAE': initialMae,
            'loss': history, 'MSE': history, 'MAE': maeHistory, 'scope': 'training', 'complete': False})

    historyInterval = max(1, (parameters['epochs'] + 99) // 100)
    try:
        for epoch in range(1, parameters['epochs'] + 1):
            checkpoint()
            model.train()
            indices = torch.randperm(sampleCount, generator=generator)
            for start in range(0, sampleCount, batchSize):
                checkpoint()
                batch = tensor[indices[start:start + batchSize]]
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(batch), batch)
                if not bool(torch.isfinite(loss)):
                    raise ValueError('autoencoder training loss is non-finite; rescale snapshots or lower learningRate')
                loss.backward()
                if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                       for parameter in model.parameters()):
                    raise ValueError('autoencoder training gradients are non-finite')
                optimizer.step()
            finalLoss, finalMae = evaluateLoss()
            if not np.isfinite(finalLoss):
                raise ValueError('autoencoder produced a non-finite training loss')
            history.append(finalLoss)
            maeHistory.append(finalMae)
            if epoch == 1 or epoch % historyInterval == 0:
                saveHistory()
            emitMetrics({'epoch': epoch, 'total': parameters['epochs'], 'progress': epoch / parameters['epochs'],
                         'loss': finalLoss, 'learningRate': parameters['learningRate']})
            checkpoint()
    except BaseException as trainingError:
        try:
            saveHistory()
        except Exception as historyError:
            raise trainingError from historyError
        raise
    else:
        saveHistory()
    # Training loss uses the float32 network input. Reconstruction diagnostics
    # compare against the original float64 snapshots, including quantization.
    squaredError = 0.0
    absoluteSum = 0.0
    maximumError = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, sampleCount, batchSize):
            checkpoint()
            output = model(tensor[start:start + batchSize]).numpy().astype(np.float64)
            difference = output.T - data[:, start:start + batchSize]
            squaredError += float(np.sum(np.square(difference)))
            absoluteSum += float(np.sum(np.abs(difference)))
            maximumError = max(maximumError, float(np.max(np.abs(difference))))
    denominator = float(np.linalg.norm(data))
    absoluteError = squaredError ** 0.5
    relativeError = absoluteError / denominator if denominator else absoluteError
    if not np.isfinite(relativeError):
        raise ValueError('autoencoder produced a non-finite reconstruction error')
    setStage('saving')
    arrays = {}
    for index, (name, value) in enumerate(model.state_dict().items()):
        checkpoint()
        array = value.detach().cpu().numpy()
        if not np.all(np.isfinite(array)):
            raise ValueError('autoencoder produced non-finite model weights')
        arrays[name] = atomicArray(directory / f'weight-{index:03d}.npy', array)
    atomicJson(directory / 'training-history.json', {'initialLoss': initialLoss, 'initialMAE': initialMae,
        'loss': history, 'MSE': history, 'MAE': maeHistory, 'scope': 'training', 'complete': True})
    metrics = {'MAE': absoluteSum / data.size, 'MSE': squaredError / data.size, 'RMSE': (squaredError / data.size) ** 0.5,
               'maxError': maximumError, 'relativeError': relativeError if denominator else None}
    atomicJson(directory / 'metrics.json', {'scope': 'training-reconstruction', **metrics})
    summary = {
        'method': 'autoencoder', 'actualRank': parameters['latentDim'], 'shape': list(data.shape),
        'initialLoss': initialLoss, 'finalLoss': finalLoss, 'epochs': len(history),
        'reconstructionError': relativeError, 'parameterCount': parameterCount,
        'metricScope': 'training', 'trainingInputDtype': 'float32', 'lossAccumulationDtype': 'float64',
        'reconstructionReferenceDtype': 'float64', 'estimatedPeakBytes': estimatedPeak,
        'metrics': metrics,
    }
    manifest = {
        'apiVersion': 1, 'modelVersion': 1, 'kind': 'autoencoder', 'taskId': request['taskId'],
        'nodeId': request['nodeId'], 'backend': 'torch', 'device': 'cpu', 'arrays': arrays,
        'architecture': {'model': 'autoEncoderTorch', 'stateDim': stateDim,
                         'latentDim': parameters['latentDim'], 'hiddenDim': parameters['hiddenDim']},
        'summary': summary, 'input': request['inputs']['snapshots'], 'parameters': parameters,
        'preprocessing': {'axes': ['state', 'snapshot'], 'dtype': 'float32', 'normalization': 'none'},
        'versions': {'python': platform.python_version(), 'mor': mor.__version__,
                     'numpy': np.__version__, 'torch': torch.__version__},
    }
    log(f'Autoencoder training MSE: {initialLoss:.6g} -> {finalLoss:.6g}.')
    return manifest, summary


class AutoencoderModel:
    def __init__(self, model, manifest):
        self.model_ = model.eval()
        self.manifest = jsonCopy(manifest)

    def project(self, snapshots):
        return self._transform(snapshots, self.manifest['architecture']['stateDim'],
                               self.manifest['architecture']['latentDim'], self.model_.encode)

    def reconstruct(self, coefficients):
        return self._transform(coefficients, self.manifest['architecture']['latentDim'],
                               self.manifest['architecture']['stateDim'], self.model_.decode)

    def _transform(self, values, rows, outputRows, operation):
        import numpy as np
        import torch

        raw = np.asarray(values)
        if raw.dtype.kind not in 'fiu' or raw.ndim != 2 or raw.shape[0] != rows or raw.shape[1] < 1:
            raise ValueError(f'input must be a nonempty real 2D matrix with {rows} rows')
        if max(raw.size, outputRows * raw.shape[1]) * 4 > MAX_INPUT_BYTES:
            raise ValueError('autoencoder inference input or output exceeds the configured size limit')
        if not np.all(np.isfinite(raw)) or np.max(np.abs(raw.astype(np.float64))) > np.finfo(np.float32).max:
            raise ValueError('input must contain finite values representable as float32')
        results = []
        batchSize = min(1024, max(1, MAX_INPUT_BYTES // (16 * max(rows, outputRows))))
        with torch.no_grad():
            for start in range(0, raw.shape[1], batchSize):
                tensor = torch.from_numpy(np.array(raw[:, start:start + batchSize].T, dtype=np.float32, copy=True))
                results.append(operation(tensor).numpy().copy())
        result = np.concatenate(results, axis=0).T.copy()
        if not np.all(np.isfinite(result)):
            raise ValueError('autoencoder inference produced non-finite values')
        return result


def loadAutoencoderModel(manifestPath):
    import numpy as np
    import torch

    path = Path(manifestPath).resolve()

    if path.name != 'manifest.json':
        raise ValueError('expected a manifest.json model file')
    manifest = readJsonObject(path)
    if (type(manifest.get('apiVersion')) is not int or manifest['apiVersion'] != 1 or
            type(manifest.get('modelVersion')) is not int or manifest['modelVersion'] != 1 or
            manifest.get('kind') != 'autoencoder' or manifest.get('backend') != 'torch' or
            manifest.get('device') != 'cpu'):
        raise ValueError('unsupported autoencoder manifest version, kind or runtime')
    if not all(isinstance(manifest.get(key), dict) for key in ('architecture', 'arrays', 'summary', 'parameters')):
        raise ValueError('autoencoder manifest requires architecture, arrays, parameters and summary objects')
    validateCommittedTask(manifest, path.parent)
    if manifest.get('preprocessing') != {'axes': ['state', 'snapshot'], 'dtype': 'float32', 'normalization': 'none'}:
        raise ValueError('unsupported autoencoder preprocessing metadata')
    architecture = manifest['architecture']
    if architecture.get('model') != 'autoEncoderTorch':
        raise ValueError('unsupported autoencoder architecture')
    stateDim, latentDim, hiddenDim = (architecture.get(key) for key in ('stateDim', 'latentDim', 'hiddenDim'))
    count = _architecture(stateDim, latentDim, hiddenDim)
    summary = manifest['summary']
    if (summary.get('actualRank') != latentDim or summary.get('parameterCount') != count or
            not isinstance(summary.get('shape'), list) or len(summary['shape']) != 2 or
            summary['shape'][0] != stateDim or any(type(size) is not int or size < 1 for size in summary['shape']) or
            any(manifest['parameters'].get(key) != value for key, value in
                (('latentDim', latentDim), ('hiddenDim', hiddenDim)))):
        raise ValueError('autoencoder architecture and summary metadata are inconsistent')
    model = _newModel(stateDim, latentDim, hiddenDim)
    expected = model.state_dict()
    if set(manifest['arrays']) != set(expected):
        raise ValueError('autoencoder weight names do not match the architecture')
    weights = {}
    for index, (name, tensor) in enumerate(expected.items()):
        descriptor = manifest['arrays'][name]
        filename = f'weight-{index:03d}.npy'
        if not isinstance(descriptor, dict) or descriptor.get('file') != filename:
            raise ValueError('autoencoder weight path is invalid')
        arrayPath = (path.parent / filename).resolve()
        if arrayPath.parent != path.parent or arrayPath.stat().st_size > MAX_INPUT_BYTES + MAX_HEADER_BYTES:
            raise ValueError('autoencoder weight exceeds the permitted directory or size')
        if fileDigest(arrayPath) != descriptor.get('sha256'):
            raise ValueError(f'autoencoder weight checksum mismatch: {name}')
        array = np.load(arrayPath, mmap_mode='r', allow_pickle=False, max_header_size=MAX_HEADER_BYTES)
        try:
            if (not isinstance(array, np.ndarray) or array.dtype != np.dtype('float32') or
                    tuple(array.shape) != tuple(tensor.shape) or list(array.shape) != descriptor.get('shape') or
                    str(array.dtype) != descriptor.get('dtype') or not np.all(np.isfinite(array))):
                raise ValueError(f'autoencoder weight metadata or values are invalid: {name}')
            weights[name] = torch.from_numpy(np.array(array, copy=True))
        finally:
            if hasattr(array, '_mmap') and array._mmap is not None:
                array._mmap.close()
            elif hasattr(array, 'close'):
                array.close()
    model.load_state_dict(weights, strict=True)
    return AutoencoderModel(model, manifest)
