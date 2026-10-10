import json
from pathlib import Path

import numpy as np
import pytest

from mor.shonta import executeWorkflow, loadAutoencoderModel, loadDynamicsModel, loadPodModel
from test_workflow_input_kinds import reference, requestFor

import onnxruntime as ort
import onnx


@pytest.mark.parametrize('method', ['pod', 'autoencoder', 'dmd', 'pod-dmd'])
def test_exported_inference_matches_reloaded_model_on_unseen_data(tmp_path, method):
    transition = np.array([[0.8, -0.3], [0.3, 0.8]])
    series = [np.array([1., 0.])]
    for _ in range(19):
        series.append(transition @ series[-1])
    parameters = {'rank': 2} if method in ('pod', 'dmd') else {'podRank': 2, 'dmdRank': 2} if method == 'pod-dmd' else {'epochs': 3, 'hiddenDim': 4, 'latentDim': 1}
    request = requestFor(tmp_path, method, 'timeSeries', parameters)
    request['inputs']['snapshots'] = reference(tmp_path, method, np.column_stack(series), ['state', 'snapshot'])
    if method in ('dmd', 'pod-dmd'):
        request['inputs']['sequence'] = {'dt': 0.1, 'pairs': [[i, i + 1] for i in range(19)]}
    result = executeWorkflow(request)
    assert result['state'] == 'succeeded', result['error']
    manifest = json.loads(Path(result['modelReference']).read_text())
    assert manifest['inference']['schemaVersion'] == 1
    session = ort.InferenceSession(str(Path(result['modelReference']).parent / manifest['inference']['file']), providers=['CPUExecutionProvider'])
    samples = np.random.default_rng(17).normal(size=(2, 7)) + 0.31
    model = (loadPodModel if method == 'pod' else loadAutoencoderModel if method == 'autoencoder' else loadDynamicsModel)(result['modelReference'])
    expected = model.step(samples) if method in ('dmd', 'pod-dmd') else model.reconstruct(model.project(samples))
    actual = session.run(None, {'values': samples})[0]
    np.testing.assert_allclose(actual, expected, atol=2e-7 if method == 'autoencoder' else 1e-12)


@pytest.mark.parametrize('continuous', [True, False])
def test_bt_onnx_output_and_transition_match_native_model(tmp_path, continuous):
    request = requestFor(tmp_path, 'bt', 'ltiSystem', {'order': 2})
    matrices = {'A': np.diag([-1., -2.]) if continuous else np.diag([0.8, 0.6]),
                'B': np.array([[1.], [2.]]), 'C': np.array([[1., 0.5]]),
                'D': np.array([[0.2]]), 'E': np.diag([2., 1.])}
    request['inputs']['system'] = {'matrices': {key: reference(tmp_path, key, value, ['row', 'column']) for key, value in matrices.items()}, 'isContinuous': continuous, 'dt': 0.2}
    result = executeWorkflow(request)
    assert result['state'] == 'succeeded', result['error']
    model = loadDynamicsModel(result['modelReference'])
    directory = Path(result['modelReference']).parent
    inference = model.manifest['inference']
    generator = np.load(directory / inference['systemMatrix']['file'])
    if continuous:
        from scipy.linalg import expm
        augmented = np.zeros((3, 3))
        augmented[:2] = generator
        generator = expm(augmented * 0.2)[:2].copy()
    session = ort.InferenceSession(str(directory / inference['file']), providers=['CPUExecutionProvider'])
    states = np.array([[0.5, -1.], [0.3, 2.]])
    inputs = np.array([[1., 0.2]])
    result = session.run(None, {'values': np.vstack((states, inputs)), 'transition': generator})[0]
    np.testing.assert_allclose(result[:2], model.step(states, inputs, dt=0.2), atol=1e-12)
    np.testing.assert_allclose(result[2:], model.output(states, inputs), atol=1e-12)
