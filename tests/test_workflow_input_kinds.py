import json
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from mor.shonta import executeWorkflow, loadDynamicsModel, validateWorkflow


def requestFor(tmp_path, method, kind, parameters=None):
    task = uuid4().hex
    return {'apiVersion': 1, 'taskId': task, 'nodeId': 'input-kind-test', 'caseId': 'test',
            'method': method, 'inputKind': kind, 'parameters': parameters or {},
            'inputs': {}, 'outputDirectory': str(tmp_path / task)}


def reference(tmp_path, name, array, axes):
    path = tmp_path / (uuid4().hex + name + '.npy')
    np.save(path, array)
    return {'path': str(path), 'format': 'npy', 'axes': axes}


@pytest.mark.parametrize('method', ['dmd', 'pod-dmd'])
def test_time_series_repeated_training_reload_complex_modes(tmp_path, method):
    transition = np.array([[0.8, -0.3], [0.3, 0.8]])
    first = [np.array([1., 0.])]
    second = [np.array([2., -1.])]
    for _ in range(9):
        first.append(transition @ first[-1])
        second.append(transition @ second[-1])
    data = np.column_stack(first + second)
    for _ in range(2):
        request = requestFor(tmp_path, method, 'timeSeries', {'rank': 2} if method == 'dmd' else {'podRank': 2, 'dmdRank': 2})
        request['inputs'] = {'snapshots': reference(tmp_path, method, data, ['state', 'snapshot']),
                             'sequence': {'dt': 0.1, 'pairs': [[i, i + 1] for i in range(19) if i != 9]}}
        result = executeWorkflow(request)
        assert result['state'] == 'succeeded', result['error']
        assert result['summary']['trainingPairs'] == 18
        assert result['summary']['metrics']['MSE'] < 1e-25
        model = loadDynamicsModel(result['modelReference'])
        assert np.iscomplexobj(model.arrays['modes'])
        Path(request['inputs']['snapshots']['path']).unlink()
        np.testing.assert_allclose(model.step(data), transition @ data, atol=1e-12)
        np.testing.assert_allclose(model.reconstruct(model.project(data)), data, atol=1e-12)
        manifest = json.loads(Path(result['modelReference']).read_text())
        assert manifest['arrays']['modes']['dtype'] == 'complex128'


@pytest.mark.parametrize('continuous', [True, False])
def test_bt_descriptor_roundtrip_step_and_output(tmp_path, continuous):
    a = np.diag([-1., -2.]) if continuous else np.diag([0.8, 0.6])
    e = np.diag([2., 1.])
    b = np.array([[1.], [2.]])
    c = np.array([[1., 0.5]])
    request = requestFor(tmp_path, 'bt', 'ltiSystem', {'order': 2})
    request['inputs']['system'] = {'matrices': {key: reference(tmp_path, key, value, ['row', 'column']) for key, value in
                                              {'A': a, 'B': b, 'C': c, 'E': e}.items()}, 'isContinuous': continuous}
    if not continuous:
        request['inputs']['system']['dt'] = 0.2
    result = executeWorkflow(request)
    assert result['state'] == 'succeeded', result['error']
    model = loadDynamicsModel(result['modelReference'])
    x, original = np.zeros((2, 1)), np.zeros((2, 1))
    from scipy.linalg import expm
    operator, control = np.linalg.solve(e, a), np.linalg.solve(e, b)
    if continuous:
        augmented = np.zeros((3, 3))
        augmented[:2, :2], augmented[:2, 2:] = operator, control
        transition = expm(augmented * 0.2)
        operator, control = transition[:2, :2], transition[:2, 2:]
    for _ in range(10):
        x = model.step(x, np.ones((1, 1)), dt=0.2)
        original = operator @ original + control
        np.testing.assert_allclose(model.output(x, np.ones((1, 1))), c @ original, atol=1e-12)


@pytest.mark.parametrize('method', ['dmd', 'pod-dmd', 'bt'])
def test_dynamics_loader_checks_commit_metadata_and_allows_relocation(tmp_path, method):
    if method == 'bt':
        request = requestFor(tmp_path, method, 'ltiSystem', {'order': 1})
        request['inputs']['system'] = {
            'matrices': {key: reference(tmp_path, key, value, ['row', 'column']) for key, value in
                         {'A': -np.diag([1., 2.]), 'B': np.ones((2, 1)), 'C': np.ones((1, 2))}.items()},
            'isContinuous': True,
        }
    else:
        request = requestFor(tmp_path, method, 'timeSeries')
        request['inputs'] = {
            'snapshots': reference(tmp_path, 'series', np.array([[1., 0.8, 0.64], [1., 0.6, 0.36]]),
                                   ['state', 'snapshot']),
            'sequence': {'dt': 1},
        }
    result = executeWorkflow(request)
    assert result['state'] == 'succeeded', result['error']
    source = Path(result['modelReference']).parent
    target = tmp_path / 'relocated' / source.name
    shutil.copytree(source, target)
    manifestPath, terminalPath = target / 'manifest.json', target / 'result.json'
    model = loadDynamicsModel(manifestPath)
    assert model.manifest['kind'] == method
    originals = {path: path.read_text(encoding='utf-8') for path in (manifestPath, terminalPath)}
    mutations = [(manifestPath, 'apiVersion', True), (manifestPath, 'modelVersion', 1.0),
                 (terminalPath, 'apiVersion', True), (terminalPath, 'apiVersion', 2),
                 (terminalPath, 'state', 'failed'), (terminalPath, 'taskId', 'other-task'),
                 (terminalPath, 'nodeId', 'other-node'), (terminalPath, 'modelReference', 'other.json')]
    for path, field, value in mutations:
        metadata = json.loads(originals[path])
        metadata[field] = value
        path.write_text(json.dumps(metadata), encoding='utf-8')
        with pytest.raises(ValueError):
            loadDynamicsModel(manifestPath)
        path.write_text(originals[path], encoding='utf-8')
    for path in (manifestPath, terminalPath):
        metadata = json.loads(originals[path])
        metadata['taskId'] = 123
        path.write_text(json.dumps(metadata), encoding='utf-8')
    with pytest.raises(ValueError, match='successfully committed'):
        loadDynamicsModel(manifestPath)


@pytest.mark.parametrize('method,kind', [('dmd', 'snapshots'), ('pod-dmd', 'snapshots'), ('bt', 'timeSeries'), ('pod', 'ltiSystem'), ('autoencoder', 'unknown')])
def test_incompatible_kind_rejected_before_execution(tmp_path, method, kind):
    result = validateWorkflow(requestFor(tmp_path, method, kind))
    assert not result['valid']
    assert any(error['field'] == 'inputKind' for error in result['errors'])


@pytest.mark.parametrize('method', [[], {}, None])
def test_malformed_method_returns_validation_errors(tmp_path, method):
    result = validateWorkflow(requestFor(tmp_path, method, 'snapshots'))
    assert not result['valid']
    assert any(error['field'] == 'method' for error in result['errors'])


@pytest.mark.parametrize('pairs', [[[1, 2], [0, 1]], [[0, 2]], [[0, 1], [0, 1]], [], [[True, 2]]])
def test_invalid_time_pairs_rejected(tmp_path, pairs):
    request = requestFor(tmp_path, 'dmd', 'timeSeries')
    request['inputs'] = {'snapshots': reference(tmp_path, 'data', np.ones((2, 4)), ['state', 'snapshot']),
                         'sequence': {'dt': 1, 'pairs': pairs}}
    assert not validateWorkflow(request)['valid']


def test_bt_unstable_and_singular_systems_do_not_commit(tmp_path):
    for a, e in ((np.eye(2), np.eye(2)), (-np.eye(2), np.zeros((2, 2)))):
        request = requestFor(tmp_path, 'bt', 'ltiSystem', {'order': 1})
        request['inputs']['system'] = {'matrices': {key: reference(tmp_path, key, value, ['row', 'column']) for key, value in
                                                  {'A': a, 'B': np.ones((2, 1)), 'C': np.ones((1, 2)), 'E': e}.items()}}
        result = executeWorkflow(request)
        assert result['state'] == 'failed'
        assert not (Path(request['outputDirectory']) / 'manifest.json').exists()


def test_final_only_training_keeps_history_without_epoch_callbacks(tmp_path):
    request = requestFor(tmp_path, 'autoencoder', 'snapshots', {'epochs': 3, 'latentDim': 1})
    request['inputs']['snapshots'] = reference(tmp_path, 'nn', np.arange(20.).reshape(2, 10) / 20, ['state', 'snapshot'])
    request['eventPolicy'] = 'finalOnly'
    events = []
    result = executeWorkflow(request, events.append)
    assert result['state'] == 'succeeded', result['error']
    assert len(events) == 1 and events[0]['payload']['state'] == 'succeeded'
    assert events[0]['sequence'] == 1
    assert any('history' in Path(path).name for path in result['files'])
    history = json.loads((Path(result['modelReference']).parent / 'training-history.json').read_text())
    assert history['complete'] and len(history['MAE']) == len(history['MSE']) == 3
    assert 'Training CPU autoencoder' in (Path(result['modelReference']).parent / 'calculation.log').read_text()
