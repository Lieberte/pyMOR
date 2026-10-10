import json
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from mor.workflow import executeWorkflow, getCapabilities, loadPodModel, validateWorkflow


def requestFor(tmp_path, method='mock', **parameters):
    taskId = uuid4().hex
    return {
        'apiVersion': 1, 'taskId': taskId, 'caseId': str(tmp_path / '中文 case'), 'nodeId': 'node-1',
        'method': method, 'parameters': {'steps': 3, 'delayMs': 0, **parameters} if method == 'mock' else parameters,
        'inputs': {}, 'outputDirectory': str(tmp_path / taskId),
    }


def podRequest(tmp_path, data=None, **parameters):
    request = requestFor(tmp_path, 'pod', **parameters)
    source = tmp_path / f'{request["taskId"]}-input.npy'
    np.save(source, np.array([[10.0, 11.0, 9.0], [20.0, 22.0, 18.0]]) if data is None else data)
    request['inputs'] = {'snapshots': {'path': str(source), 'format': 'npy', 'axes': ['state', 'snapshot']}}
    return request


def attachWorkflow(request):
    request['workflow'] = {
        'schemaVersion': 1, 'executionScope': 'selectedNode', 'selectedNodeId': request['nodeId'],
        'graph': {
            'nodes': [{'uuid': request['nodeId'], 'typeId': 'rom.reductionMethod',
                       'romMetadata': {'methodName': request['method'], 'parameters': request['parameters'].copy()}}],
            'connections': [],
        },
    }
    return request['workflow']


def readJson(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def terminalStates(events):
    return [event['payload']['state'] for event in events if event['type'] == 'state' and
            event['payload']['state'] in ('succeeded', 'failed', 'cancelled')]


def test_mock_persists_json_events_and_immutable_request(tmp_path):
    request = requestFor(tmp_path)
    attachWorkflow(request)
    original = json.loads(json.dumps(request))
    events = []
    result = executeWorkflow(request, events.append)
    directory = Path(request['outputDirectory'])

    assert result['state'] == 'succeeded'
    assert request == original
    assert result['modelReference'] == str(directory / 'manifest.json')
    assert readJson(directory / 'result.json') == result
    assert readJson(directory / 'submittedRequest.json')['workflow'] == original['workflow']
    assert readJson(result['modelReference'])['kind'] == 'mock'
    assert [event['sequence'] for event in events] == list(range(1, len(events) + 1))
    assert terminalStates(events) == ['succeeded']
    assert all(event['taskId'] == request['taskId'] and event['nodeId'] == request['nodeId'] for event in events)
    persisted = [json.loads(line) for line in (directory / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert persisted == events
    assert all(Path(path).is_file() for path in result['files'])
    json.dumps(result, allow_nan=False)


def test_verbose_false_keeps_metrics_and_states(tmp_path):
    events = []
    result = executeWorkflow(requestFor(tmp_path, verbose=False), events.append)
    assert result['state'] == 'succeeded'
    assert not any(event['type'] == 'log' for event in events)
    metrics = [event['payload'] for event in events if event['type'] == 'metrics']
    assert [payload['step'] for payload in metrics] == [1, 2, 3]
    assert metrics[-1]['progress'] == 1.0


@pytest.mark.parametrize('stage', ['preparing', 'running', 'saving', 'committing'])
def test_cancel_checkpoint_never_commits_model(tmp_path, stage):
    request = requestFor(tmp_path)
    cancelled = False
    events = []

    def observe(event):
        nonlocal cancelled
        events.append(event)
        if event['type'] == 'state' and event['payload'].get('stage') == stage:
            cancelled = True

    result = executeWorkflow(request, observe, lambda: cancelled)
    assert result['state'] == 'cancelled'
    assert result['modelReference'] is None
    assert not (Path(request['outputDirectory']) / 'manifest.json').exists()
    assert terminalStates(events) == ['cancelled']
    assert readJson(Path(request['outputDirectory']) / 'result.json')['state'] == 'cancelled'


def test_cancel_during_last_step_cannot_be_reported_as_success(tmp_path):
    cancelled = False

    def observe(event):
        nonlocal cancelled
        if event['type'] == 'metrics' and event['payload']['step'] == 3:
            cancelled = True

    result = executeWorkflow(requestFor(tmp_path), observe, lambda: cancelled)
    assert result['state'] == 'cancelled'


def test_cancel_after_manifest_commit_keeps_success(tmp_path):
    cancelled = False

    def observe(event):
        nonlocal cancelled
        if event['type'] == 'state' and event['payload']['state'] == 'succeeded':
            cancelled = True

    result = executeWorkflow(requestFor(tmp_path), observe, lambda: cancelled)
    assert cancelled
    assert result['state'] == 'succeeded'
    assert Path(result['modelReference']).is_file()


def test_mock_failure_has_traceback_and_no_model(tmp_path):
    request = requestFor(tmp_path, failAtStep=2)
    events = []
    result = executeWorkflow(request, events.append)
    assert result['state'] == 'failed'
    assert result['error']['code'] == 'execution_error'
    assert 'Mock failure requested at step 2' in result['error']['traceback']
    assert result['modelReference'] is None
    assert terminalStates(events) == ['failed']
    assert not (Path(request['outputDirectory']) / 'manifest.json').exists()


def test_event_observer_failure_aborts_before_commit(tmp_path):
    def broken(event):
        raise RuntimeError('receiver destroyed')

    request = requestFor(tmp_path)
    result = executeWorkflow(request, broken)
    assert result['state'] == 'failed'
    assert result['error']['code'] == 'observer_error'
    assert not (Path(request['outputDirectory']) / 'manifest.json').exists()


def test_terminal_observer_failure_is_a_warning_after_commit(tmp_path):
    def broken(event):
        if event['type'] == 'state' and event['payload']['state'] == 'succeeded':
            raise RuntimeError('late receiver failure')

    result = executeWorkflow(requestFor(tmp_path), broken)
    assert result['state'] == 'succeeded'
    assert result['error'] is None
    assert result['summary']['warnings']
    assert Path(result['modelReference']).is_file()


def test_cancellation_callback_failure_is_not_success_or_cancelled(tmp_path):
    def broken():
        raise RuntimeError('lost cancellation token')

    result = executeWorkflow(requestFor(tmp_path), isCancelled=broken)
    assert result['state'] == 'failed'
    assert result['error']['code'] == 'observer_error'


def test_callback_mutation_does_not_change_persisted_identity(tmp_path):
    request = requestFor(tmp_path)

    def mutate(event):
        event['taskId'] = 'changed'
        event['payload'].clear()

    result = executeWorkflow(request, mutate)
    persisted = Path(request['outputDirectory']) / 'events.jsonl'
    assert result['state'] == 'succeeded'
    assert all(json.loads(line)['taskId'] == request['taskId'] for line in persisted.read_text().splitlines())


def test_new_task_cannot_enter_during_terminal_delivery(tmp_path):
    request = requestFor(tmp_path)
    nested = requestFor(tmp_path)
    nestedResults = []
    nestedEvents = []

    def observe(event):
        if event['type'] == 'state' and event['payload']['state'] == 'succeeded':
            nestedResults.append(executeWorkflow(nested, nestedEvents.append))

    assert executeWorkflow(request, observe)['state'] == 'succeeded'
    assert nestedResults[0]['error']['code'] == 'task_busy'
    assert nestedEvents[-1]['payload']['stage'] == 'admission'
    assert nestedEvents[-1]['payload']['error']['stage'] == 'admission'
    assert not Path(nested['outputDirectory']).exists()
    assert executeWorkflow(nested)['state'] == 'succeeded'


def test_duplicate_task_id_cannot_write_a_second_output(tmp_path):
    request = requestFor(tmp_path)
    assert executeWorkflow(request)['state'] == 'succeeded'
    request['outputDirectory'] = str(tmp_path / 'another-output')
    events = []
    result = executeWorkflow(request, events.append)
    assert result['error']['code'] == 'duplicate_task'
    assert events[-1]['payload']['stage'] == 'admission'
    assert events[-1]['payload']['error']['stage'] == 'admission'
    assert not Path(request['outputDirectory']).exists()


def test_existing_output_is_preserved(tmp_path):
    request = requestFor(tmp_path)
    output = Path(request['outputDirectory'])
    output.mkdir()
    marker = output / 'keep.txt'
    marker.write_text('user content')
    result = executeWorkflow(request)
    assert result['state'] == 'failed'
    assert marker.read_text() == 'user content'
    assert list(output.iterdir()) == [marker]


def test_failed_manifest_commit_rewrites_success_diagnostics_as_failed(tmp_path, monkeypatch):
    from mor.workflow import service

    original = service.atomicJson

    def failManifest(path, value):
        if Path(path).name == 'manifest.json':
            raise OSError('simulated full disk')
        return original(path, value)

    monkeypatch.setattr(service, 'atomicJson', failManifest)
    request = requestFor(tmp_path)
    result = executeWorkflow(request)
    assert result['state'] == 'failed'
    assert result['error']['code'] == 'storage_error'
    assert result['modelReference'] is None
    assert not (Path(request['outputDirectory']) / 'manifest.json').exists()
    assert readJson(Path(request['outputDirectory']) / 'result.json')['state'] == 'failed'


def test_unreadable_output_does_not_raise_from_error_reporting(tmp_path, monkeypatch):
    original = Path.iterdir
    request = requestFor(tmp_path, failAtStep=1)
    directory = Path(request['outputDirectory'])

    def inaccessible(path):
        if path == directory:
            raise PermissionError('lost directory access')
        return original(path)

    monkeypatch.setattr(Path, 'iterdir', inaccessible)
    result = executeWorkflow(request)
    assert result['state'] == 'failed'
    assert 'Mock failure' in result['error']['message']


@pytest.mark.parametrize('changes', [
    {'apiVersion': True}, {'taskId': ''}, {'nodeId': []}, {'caseId': ''}, {'method': 'nn'},
    {'parameters': []}, {'parameters': {'steps': True}}, {'parameters': {'steps': 0}},
    {'parameters': {'delayMs': -1}}, {'parameters': {'failAtStep': 10001}},
    {'parameters': {'verbose': 'false'}}, {'parameters': {'device': 'cuda'}},
    {'outputDirectory': 'relative-output'}, {'inputs': []}, {'workflow': []},
])
def test_invalid_request_returns_structured_errors_without_writes(tmp_path, changes):
    request = {**requestFor(tmp_path), **changes}
    validation = validateWorkflow(request)
    assert not validation['valid']
    assert validation['request'] is None
    assert all(set(error) == {'field', 'message', 'code'} for error in validation['errors'])
    assert executeWorkflow(request)['state'] == 'failed'


@pytest.mark.parametrize('parameters', [{'rank': True}, {'rank': 0}, {'rank': 1, 'tol': 0.1},
                                      {'tol': -1}, {'tol': 1}, {'tol': 10 ** 1000},
                                      {'centerSnapshots': 1}, {'tol': float('nan')}])
def test_pod_parameter_validation_handles_invalid_json_numbers(tmp_path, parameters):
    validation = validateWorkflow(podRequest(tmp_path, **parameters))
    assert not validation['valid']


def test_capabilities_advertise_packaged_methods_without_probing(monkeypatch):
    import importlib.util
    from mor import shonta

    monkeypatch.setattr(shonta, 'getRuntimeInfo', lambda *args: pytest.fail('must not scan runtime devices'))
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: pytest.fail('must not probe dependencies'))
    capability = shonta.getCapabilities()
    methods = {item['id']: item for item in capability['methods']}
    assert set(methods) == {'mock', 'pod', 'autoencoder', 'dmd', 'pod-dmd', 'bt'}
    assert all(method['available'] and method['reason'] is None for method in methods.values())
    assert methods['pod']['availabilityCheck'] == 'packaged-dependencies'
    assert capability['maxConcurrentTasks'] == 1


def test_missing_deployment_dependency_fails_task_before_commit(tmp_path, monkeypatch):
    from mor.workflow import onnxExport

    def missingDependency(*args):
        raise ModuleNotFoundError('packaged ONNX dependency is missing', name='onnx')

    monkeypatch.setattr(onnxExport, 'exportInference', missingDependency)
    request = podRequest(tmp_path, rank=1)
    result = executeWorkflow(request)
    assert result['state'] == 'failed'
    assert result['modelReference'] is None
    assert result['error']['stage'] == 'exporting'
    assert 'packaged ONNX dependency is missing' in result['error']['message']
    assert not (Path(request['outputDirectory']) / 'manifest.json').exists()


def test_selected_workflow_scope_and_method_are_checked(tmp_path):
    request = requestFor(tmp_path)
    workflow = attachWorkflow(request)
    assert validateWorkflow(request)['valid']
    workflow['executionScope'] = 'wholeGraph'
    assert not validateWorkflow(request)['valid']
    workflow['executionScope'] = 'selectedNode'
    workflow['selectedNodeId'] = 'different-node'
    assert not validateWorkflow(request)['valid']
    workflow['selectedNodeId'] = request['nodeId']
    workflow['graph']['nodes'][0]['romMetadata']['methodName'] = 'pod'
    assert not validateWorkflow(request)['valid']


def test_selected_parameters_match_after_defaults_are_normalized(tmp_path):
    request = podRequest(tmp_path, rank=1)
    workflow = attachWorkflow(request)
    workflow['graph']['nodes'][0]['romMetadata']['parameters']['tol'] = None
    assert validateWorkflow(request)['valid']
    workflow['graph']['nodes'][0]['romMetadata']['parameters']['rank'] = 2
    assert not validateWorkflow(request)['valid']


@pytest.mark.parametrize('typeId', ['rom.datasetSplit', 'rom.dataProcessing', 'rom.reductionMethod'])
def test_unsupported_upstream_graph_nodes_are_rejected(tmp_path, typeId):
    request = requestFor(tmp_path)
    workflow = attachWorkflow(request)
    workflow['graph']['nodes'].append({'uuid': 'input-1', 'typeId': typeId})
    workflow['graph']['connections'].append({'from': {'nodeUuid': 'input-1', 'portId': 'data'},
                                               'to': {'nodeUuid': request['nodeId'], 'portId': 'trainingData'}})
    assert not validateWorkflow(request)['valid']
    workflow['graph']['nodes'][-1]['typeId'] = 'rom.dataInput'
    assert validateWorkflow(request)['valid']


@pytest.mark.parametrize('endpoint,port', [('from', None), ('to', ''), ('from', 42),
                                         ('from', 'missing'), ('to', 'reducedModel')])
def test_invalid_input_connection_ports_are_rejected(tmp_path, endpoint, port):
    request = requestFor(tmp_path)
    workflow = attachWorkflow(request)
    workflow['graph']['nodes'].append({'uuid': 'input-1', 'typeId': 'rom.dataInput'})
    connection = {'from': {'nodeUuid': 'input-1', 'portId': 'data'},
                  'to': {'nodeUuid': request['nodeId'], 'portId': 'trainingData'}}
    workflow['graph']['connections'].append(connection)
    assert validateWorkflow(request)['valid']
    if port is None:
        del connection[endpoint]['portId']
    else:
        connection[endpoint]['portId'] = port
    result = executeWorkflow(request)
    assert result['state'] == 'failed'
    assert result['error']['code'] == 'invalid_request'
    assert not Path(request['outputDirectory']).exists()


def test_centered_pod_model_survives_reload_and_preserves_mean(tmp_path):
    data = np.array([[10.0, 11.0, 9.0], [20.0, 22.0, 18.0]])
    request = podRequest(tmp_path, data, rank=1, centerSnapshots=True, verbose=False)
    events = []
    result = executeWorkflow(request, events.append)
    assert result['state'] == 'succeeded', result
    model = loadPodModel(result['modelReference'])
    np.testing.assert_allclose(model.mean, data.mean(axis=1))
    np.testing.assert_allclose(model.reconstruct(model.project(data)), data, atol=1e-10)
    assert model.basis.shape == (2, 1)
    assert not model.basis.flags.writeable
    assert result['summary']['actualRank'] == 1
    assert result['summary']['reconstructionError'] < 1e-10
    directory = Path(request['outputDirectory'])
    submitted = readJson(directory / 'submittedRequest.json')
    resolved = readJson(directory / 'request.json')
    assert 'sha256' not in submitted['inputs']['snapshots']
    assert len(resolved['inputs']['snapshots']['sha256']) == 64
    assert (directory / resolved['inputs']['snapshots']['frozenFile']).is_file()
    assert model.manifest['versions']['numpy'] == np.__version__
    assert terminalStates(events) == ['succeeded']
    assert not any(event['type'] == 'log' for event in events)
    assert any(event['type'] == 'metrics' and 'reconstructionError' in event['payload'] for event in events)


def test_uncentered_full_pod_defaults_and_zero_mean(tmp_path):
    data = np.diag([4.0, 2.0, 1.0])
    result = executeWorkflow(podRequest(tmp_path, data))
    assert result['state'] == 'succeeded', result
    model = loadPodModel(result['modelReference'])
    assert result['summary']['requestedRank'] is None
    assert result['summary']['actualRank'] == 3
    np.testing.assert_array_equal(model.mean, np.zeros(3))
    np.testing.assert_allclose(model.reconstruct(model.project(data)), data, atol=1e-10)


def test_pod_tolerance_is_discarded_energy_fraction(tmp_path):
    result = executeWorkflow(podRequest(tmp_path, np.diag([10.0, 1.0, 0.1]), tol=0.02))
    assert result['state'] == 'succeeded', result
    assert result['summary']['actualRank'] == 1
    assert result['summary']['discardedEnergyTolerance'] == 0.02


@pytest.mark.parametrize('data,parameters', [
    (np.array([1.0, 2.0]), {}), (np.empty((2, 0)), {}),
    (np.array([[np.nan, 1.0]]), {}), (np.array([[np.inf, 1.0]]), {}),
    (np.array([[1.0 + 1j, 2.0]]), {}), (np.array([['a', 'b']]), {}),
    (np.ones((2, 3)), {'rank': 3}), (np.ones((2, 1)), {'centerSnapshots': True}),
])
def test_pod_rejects_unsupported_inputs_without_model(tmp_path, data, parameters):
    request = podRequest(tmp_path, data, **parameters)
    result = executeWorkflow(request)
    assert result['state'] == 'failed'
    assert result['modelReference'] is None
    assert not (Path(request['outputDirectory']) / 'manifest.json').exists()


@pytest.mark.parametrize('field,value', [('shape', [3, 2]), ('dtype', 'float32'), ('sha256', '0' * 64)])
def test_pod_rejects_input_reference_mismatch(tmp_path, field, value):
    request = podRequest(tmp_path)
    request['inputs']['snapshots'][field] = value
    result = executeWorkflow(request)
    assert result['state'] == 'failed'
    assert result['modelReference'] is None


def test_input_memory_budget_is_checked_before_reduction(tmp_path, monkeypatch):
    from mor.workflow import pod

    monkeypatch.setattr(pod, 'MAX_ESTIMATED_PEAK_BYTES', 1)
    result = executeWorkflow(podRequest(tmp_path))
    assert result['state'] == 'failed'
    assert 'peak-memory' in result['error']['message']


def test_cancel_pod_after_solver_returns_does_not_save_basis(tmp_path, monkeypatch):
    from mor.reductors import podReductor

    cancelled = False
    original = podReductor.reduce

    def reduceAndCancel(*args, **kwargs):
        nonlocal cancelled
        value = original(*args, **kwargs)
        cancelled = True
        return value

    monkeypatch.setattr(podReductor, 'reduce', reduceAndCancel)
    request = podRequest(tmp_path, rank=1)
    result = executeWorkflow(request, isCancelled=lambda: cancelled)
    assert result['state'] == 'cancelled'
    assert not (Path(request['outputDirectory']) / 'basis.npy').exists()


def test_loader_rejects_tampered_arrays_and_incomplete_tasks(tmp_path):
    result = executeWorkflow(podRequest(tmp_path, rank=1))
    assert result['state'] == 'succeeded', result
    manifest = Path(result['modelReference'])
    basisPath = manifest.parent / 'basis.npy'
    original = basisPath.read_bytes()
    basisPath.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match='checksum'):
        loadPodModel(manifest)
    basisPath.write_bytes(original)
    terminal = readJson(manifest.parent / 'result.json')
    terminal['state'] = 'failed'
    (manifest.parent / 'result.json').write_text(json.dumps(terminal), encoding='utf-8')
    with pytest.raises(ValueError, match='successfully committed'):
        loadPodModel(manifest)


def test_loader_rejects_manifest_path_traversal(tmp_path):
    result = executeWorkflow(podRequest(tmp_path, rank=1))
    manifestPath = Path(result['modelReference'])
    manifest = readJson(manifestPath)
    manifest['arrays']['basis']['file'] = '../basis.npy'
    manifestPath.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='array path'):
        loadPodModel(manifestPath)


def test_model_bundle_can_be_relocated_with_case(tmp_path):
    result = executeWorkflow(podRequest(tmp_path, rank=1, centerSnapshots=True))
    source = Path(result['modelReference']).parent
    target = tmp_path / 'relocated-case' / source.name
    shutil.copytree(source, target)
    model = loadPodModel(target / 'manifest.json')
    assert model.basis.shape == (2, 1)
    np.testing.assert_allclose(model.mean, [10.0, 20.0])


@pytest.mark.parametrize('field,value', [('apiVersion', 2), ('nodeId', 'other-node'),
                                      ('taskId', 'other-task'), ('modelReference', 'other.json')])
def test_loader_checks_result_identity(tmp_path, field, value):
    result = executeWorkflow(podRequest(tmp_path, rank=1))
    manifest = Path(result['modelReference'])
    terminal = readJson(manifest.parent / 'result.json')
    terminal[field] = value
    (manifest.parent / 'result.json').write_text(json.dumps(terminal), encoding='utf-8')
    with pytest.raises(ValueError):
        loadPodModel(manifest)


@pytest.mark.parametrize('field,value', [('centerSnapshots', 'false'), ('axes', ['snapshot', 'state']), ('dtype', 'float32')])
def test_loader_checks_preprocessing_contract(tmp_path, field, value):
    result = executeWorkflow(podRequest(tmp_path, rank=1))
    manifest = Path(result['modelReference'])
    data = readJson(manifest)
    data['preprocessing'][field] = value
    manifest.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='preprocessing'):
        loadPodModel(manifest)


@pytest.mark.parametrize('value', [[], None, 1, 'manifest'])
def test_loader_rejects_nonobject_manifest(tmp_path, value):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='JSON object'):
        loadPodModel(path)
