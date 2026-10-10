"""Reduction methods in isolation: no C++ storage, graph traversal or GUI."""
import torch

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from mor.shonta import executeWorkflow, getCapabilities, loadAutoencoderModel, loadPodModel, validateWorkflow


def methodRequest(tmp_path, method, data, **parameters):
    taskId = uuid4().hex
    source = tmp_path / f'{taskId}-source.npy'
    np.save(source, data, allow_pickle=False)
    return {'apiVersion': 1, 'taskId': taskId, 'nodeId': 'method-only', 'caseId': 'isolated-methods',
            'method': method, 'parameters': parameters,
            'inputs': {'snapshots': {'path': str(source), 'format': 'npy', 'axes': ['state', 'snapshot']}},
            'outputDirectory': str(tmp_path / taskId)}


@pytest.fixture
def snapshots():
    rng = np.random.default_rng(7)
    basis, _ = np.linalg.qr(rng.normal(size=(6, 2)))
    return basis @ rng.uniform(-1, 1, size=(2, 32))


def nnRequest(tmp_path, snapshots, **parameters):
    return methodRequest(tmp_path, 'autoencoder', snapshots,
                         **{'latentDim': 2, 'hiddenDim': 16, 'epochs': 80,
                            'batchSize': 32, 'learningRate': 0.01, 'seed': 7, 'verbose': False, **parameters})


def test_pod_roundtrip_with_seeded_low_rank_data(tmp_path, snapshots):
    for parameters in ({'rank': 2}, {'tol': 1e-12}, {'rank': 2, 'centerSnapshots': True}):
        request = methodRequest(tmp_path, 'pod', snapshots, **parameters)
        result = executeWorkflow(request)
        assert result['state'] == 'succeeded', result['error']
        model = loadPodModel(result['modelReference'])
        np.testing.assert_allclose(model.reconstruct(model.project(snapshots)), snapshots, atol=1e-12)


def test_autoencoder_trains_reloads_and_preserves_runtime_rng(tmp_path, snapshots):
    rngState = torch.get_rng_state().clone()
    request = nnRequest(tmp_path, snapshots)
    originalRequest = json.loads(json.dumps(request))
    events = []
    result = executeWorkflow(request, events.append)
    assert result['state'] == 'succeeded', result['error']
    assert request == originalRequest
    assert torch.equal(torch.get_rng_state(), rngState)
    summary = result['summary']
    assert summary['finalLoss'] < summary['initialLoss'] * 0.1
    assert summary['actualRank'] == 2
    assert summary['metricScope'] == 'training'
    model = loadAutoencoderModel(result['modelReference'])
    assert torch.equal(torch.get_rng_state(), rngState)
    coefficients = model.project(snapshots)
    reconstructed = model.reconstruct(coefficients)
    assert coefficients.shape == (2, 32)
    assert reconstructed.shape == snapshots.shape
    np.testing.assert_allclose(np.mean((reconstructed - snapshots) ** 2), summary['finalLoss'], rtol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(reconstructed - snapshots) / np.linalg.norm(snapshots),
                               summary['reconstructionError'], rtol=1e-5)
    epochs = [event['payload'] for event in events if event['type'] == 'metrics' and 'epoch' in event['payload']]
    assert [item['epoch'] for item in epochs] == list(range(1, 81))
    assert epochs[-1]['progress'] == 1
    assert [event['sequence'] for event in events] == list(range(1, len(events) + 1))
    assert not any(event['type'] == 'log' for event in events)
    assert all(Path(path).is_file() for path in result['files'])
    assert json.loads(Path(result['modelReference']).read_text())['preprocessing']['normalization'] == 'none'
    # Model arrays are self-contained; inference must not revisit mutable input.
    Path(request['inputs']['snapshots']['path']).unlink()
    np.testing.assert_array_equal(loadAutoencoderModel(result['modelReference']).project(snapshots), coefficients)


def test_autoencoder_repeated_tasks_are_deterministic_and_isolated(tmp_path, snapshots):
    requests = [nnRequest(tmp_path, snapshots, epochs=20) for _ in range(2)]
    results = [executeWorkflow(request) for request in requests]
    assert all(result['state'] == 'succeeded' for result in results), results
    assert results[0]['modelReference'] != results[1]['modelReference']
    assert results[0]['summary']['finalLoss'] == results[1]['summary']['finalLoss']
    a, b = [loadAutoencoderModel(result['modelReference']).project(snapshots) for result in results]
    np.testing.assert_array_equal(a, b)


def test_training_history_has_bounded_checkpoints_and_complete_final_values(tmp_path, snapshots, monkeypatch):
    from mor.workflow import autoencoder

    writes = []
    original = autoencoder.atomicJson

    def record(path, value):
        if path.name == 'training-history.json':
            writes.append((len(value['loss']), value['complete']))
        return original(path, value)

    monkeypatch.setattr(autoencoder, 'atomicJson', record)
    request = nnRequest(tmp_path, snapshots, epochs=205)
    events = []
    result = executeWorkflow(request, events.append)
    assert result['state'] == 'succeeded', result['error']
    assert len(writes) <= 103
    assert writes[-1] == (205, True)
    history = json.loads((Path(request['outputDirectory']) / 'training-history.json').read_text())
    assert len(history['loss']) == len(history['MAE']) == 205
    assert history['MSE'] == history['loss']
    epochEvents = [event['payload']['epoch'] for event in events if event['type'] == 'metrics' and 'epoch' in event['payload']]
    assert epochEvents == list(range(1, 206))


def test_cancelled_training_flushes_history_between_checkpoints(tmp_path, snapshots):
    cancelled = False

    def observe(event):
        nonlocal cancelled
        if event['type'] == 'metrics' and event['payload'].get('epoch') == 3:
            cancelled = True

    request = nnRequest(tmp_path, snapshots, epochs=400)
    result = executeWorkflow(request, observe, lambda: cancelled)
    assert result['state'] == 'cancelled'
    history = json.loads((Path(request['outputDirectory']) / 'training-history.json').read_text())
    assert len(history['loss']) == len(history['MAE']) == 3
    assert history['complete'] is False
    assert not (Path(request['outputDirectory']) / 'manifest.json').exists()


@pytest.mark.parametrize('cancel', [True, False])
def test_history_flush_failure_preserves_original_task_failure(tmp_path, snapshots, monkeypatch, cancel):
    from mor.workflow import autoencoder
    original = autoencoder.atomicJson
    interrupted = False

    def failHistory(path, value):
        if interrupted and path.name == 'training-history.json':
            raise OSError('history disk unavailable')
        return original(path, value)

    def observe(event):
        nonlocal interrupted
        if event['type'] == 'metrics' and event['payload'].get('epoch') == 1:
            interrupted = True
            if not cancel:
                raise RuntimeError('original observer failure')

    monkeypatch.setattr(autoencoder, 'atomicJson', failHistory)
    request = nnRequest(tmp_path, snapshots, epochs=2)
    result = executeWorkflow(request, observe, lambda: interrupted if cancel else False)
    assert result['state'] == ('cancelled' if cancel else 'failed')
    assert result['error']['code'] == ('cancelled' if cancel else 'observer_error')
    if not cancel:
        assert 'original observer failure' in result['error']['message']
    assert result['modelReference'] is None
    assert not (Path(request['outputDirectory']) / 'manifest.json').exists()


def test_successful_training_history_storage_failure_still_fails_task(tmp_path, snapshots, monkeypatch):
    from mor.workflow import autoencoder
    original = autoencoder.atomicJson
    historyWrites = 0
    def failFinalFlush(path, value):
        nonlocal historyWrites
        if path.name == 'training-history.json':
            historyWrites += 1
            if historyWrites == 2:
                raise OSError('history disk unavailable')
        return original(path, value)
    monkeypatch.setattr(autoencoder, 'atomicJson', failFinalFlush)
    result = executeWorkflow(nnRequest(tmp_path, snapshots, epochs=1))
    assert result['state'] == 'failed'
    assert result['error']['code'] == 'storage_error'
    assert result['modelReference'] is None


def test_autoencoder_reconstruction_error_includes_float32_quantization(tmp_path, monkeypatch):
    from mor.workflow import autoencoder

    # A lossless float32 model isolates input quantization from training error.
    # Adam receives a trainable scalar so this still exercises the full runner.
    class IdentityModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        def forward(self, inputs):
            return inputs * self.scale

    monkeypatch.setattr(autoencoder, '_newModel', lambda *args: IdentityModel())
    data = np.array([[1.00000001, 2.00000007, 3.0000001],
                     [4.00000001, 5.00000007, 6.0000001]], dtype=np.float64)
    result = executeWorkflow(nnRequest(tmp_path, data, epochs=1))
    assert result['state'] == 'succeeded', result['error']
    assert result['summary']['finalLoss'] == 0
    expected = np.linalg.norm(data.astype(np.float32).astype(np.float64) - data) / np.linalg.norm(data)
    assert expected > 0
    assert result['summary']['reconstructionError'] == pytest.approx(expected, rel=1e-14)
    assert result['summary']['trainingInputDtype'] == 'float32'
    assert result['summary']['reconstructionReferenceDtype'] == 'float64'


@pytest.mark.parametrize('parameters', [
    {'latentDim': 0}, {'latentDim': True}, {'hiddenDim': 4097}, {'epochs': 0}, {'epochs': 1.5},
    {'batchSize': 65537}, {'learningRate': 0}, {'learningRate': 2}, {'learningRate': True},
    {'learningRate': float('nan')}, {'seed': -1}, {'seed': 2147483648}, {'rank': 1}, {'tol': 0.1},
    {'centerSnapshots': True}, {'verbose': 'false'},
])
def test_autoencoder_rejects_wrong_parameter_types_and_pod_fields(tmp_path, snapshots, parameters):
    request = nnRequest(tmp_path, snapshots, **parameters)
    assert not validateWorkflow(request)['valid']
    assert executeWorkflow(request)['error']['code'] == 'invalid_request'
    assert not Path(request['outputDirectory']).exists()


@pytest.mark.parametrize('data,parameters', [
    (np.ones((2, 4)), {'latentDim': 3}), (np.ones((2, 1)), {}),
    (np.full((2, 4), np.nan), {}), (np.full((2, 4), 1e100), {}),
    (np.ones((2000, 2)), {'hiddenDim': 4096}),
])
def test_autoencoder_rejects_invalid_data_or_oversized_architecture(tmp_path, data, parameters):
    request = nnRequest(tmp_path, data, **parameters)
    result = executeWorkflow(request)
    assert result['state'] == 'failed'
    assert result['modelReference'] is None
    assert not (Path(request['outputDirectory']) / 'manifest.json').exists()


@pytest.mark.parametrize('cancelAt', ['training', 'epoch', 'saving', 'committing'])
def test_autoencoder_cancellation_never_commits_a_model(tmp_path, snapshots, cancelAt):
    cancelled = False

    def observe(event):
        nonlocal cancelled
        if (event['type'] == 'state' and event['payload'].get('stage') == cancelAt or
                cancelAt == 'epoch' and event['type'] == 'metrics' and event['payload'].get('epoch') == 1):
            cancelled = True

    request = nnRequest(tmp_path, snapshots, epochs=2)
    result = executeWorkflow(request, observe, lambda: cancelled)
    assert result['state'] == 'cancelled'
    assert result['modelReference'] is None
    assert not (Path(request['outputDirectory']) / 'manifest.json').exists()


def test_autoencoder_weight_tampering_is_detected(tmp_path, snapshots):
    result = executeWorkflow(nnRequest(tmp_path, snapshots, epochs=2))
    assert result['state'] == 'succeeded', result['error']
    weightPath = Path(result['modelReference']).parent / 'weight-000.npy'
    np.save(weightPath, np.zeros((16, 6), dtype=np.float32))
    with pytest.raises(ValueError, match='checksum'):
        loadAutoencoderModel(result['modelReference'])


def test_autoencoder_inference_enforces_dimensions_finiteness_and_output_budget(tmp_path, snapshots, monkeypatch):
    from mor.workflow import autoencoder

    result = executeWorkflow(nnRequest(tmp_path, snapshots, epochs=2))
    assert result['state'] == 'succeeded', result['error']
    model = loadAutoencoderModel(result['modelReference'])
    for badData in (np.ones((5, 4)), np.full((6, 4), np.nan), np.ones((6, 4), dtype=complex)):
        with pytest.raises(ValueError):
            model.project(badData)
    # Input is small, but expansion through the decoder must also be bounded.
    monkeypatch.setattr(autoencoder, 'MAX_INPUT_BYTES', 100)
    with pytest.raises(ValueError, match='input or output'):
        model.reconstruct(np.ones((2, 5)))


def test_autoencoder_capability_and_defaults_agree(tmp_path, snapshots):
    capability = next(method for method in getCapabilities()['methods'] if method['id'] == 'autoencoder')
    request = methodRequest(tmp_path, 'autoencoder', snapshots)
    validation = validateWorkflow(request)
    assert validation['valid']
    assert validation['request']['parameters'] == {
        name: value['default'] for name, value in capability['parameters'].items()}


def test_autoencoder_selected_node_snapshot_uses_its_own_defaults(tmp_path, snapshots):
    request = nnRequest(tmp_path, snapshots, epochs=2)
    request['workflow'] = {
        'schemaVersion': 1, 'executionScope': 'selectedNode', 'selectedNodeId': request['nodeId'],
        'graph': {'nodes': [{'uuid': request['nodeId'], 'typeId': 'rom.reductionMethod',
                             'romMetadata': {'methodName': 'autoencoder', 'parameters': request['parameters'].copy()}}],
                  'connections': []},
    }
    assert validateWorkflow(request)['valid']


def test_pod_execution_does_not_import_torch(tmp_path, snapshots):
    import subprocess
    import sys

    request = methodRequest(tmp_path, 'pod', snapshots, rank=2)
    script = """
import builtins, json, sys
originalImport = builtins.__import__
def withoutTorch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise ImportError('torch deliberately unavailable for the POD-only test')
    return originalImport(name, *args, **kwargs)
builtins.__import__ = withoutTorch
from mor.workflow import executeWorkflow
result = executeWorkflow(json.loads(sys.argv[1]))
assert result['state'] == 'succeeded', result
assert 'torch' not in sys.modules
"""
    completed = subprocess.run([sys.executable, '-c', script, json.dumps(request)],
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
