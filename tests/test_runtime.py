import torch
import pickle

import pytest

import mor
from mor.backends import backendRegistry
from mor.shonta import getRuntimeContract, runDmdProbe


def test_device_choices_are_stable():
    assert mor.DEVICE_CHOICES == ('auto', 'cpu', 'cuda')
    assert mor.resolveDeviceName('cpu') == 'cpu'
    assert mor.resolveDeviceName(' AUTO ') in {'cpu', 'cuda'}


def test_invalid_device_is_rejected():
    with pytest.raises(ValueError, match='unsupported deviceName'):
        mor.resolveDeviceName('gpu')


def test_explicit_cuda_is_strict(monkeypatch):

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA was explicitly requested'):
        mor.resolveDeviceName('cuda')
    assert mor.resolveDeviceName('auto') == 'cpu'


def test_forced_cpu_does_not_open_a_cuda_context(monkeypatch):

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        'current_device',
        lambda: (_ for _ in ()).throw(AssertionError('CUDA context opened')),
    )
    resolution = mor.resolveDevice('cpu')
    assert resolution.resolvedDeviceName == 'cpu'
    assert resolution.gpuName is None


def test_torch_backend_factories_honor_forced_cpu():
    backend = backendRegistry.get('torch', device='cpu')
    assert backend.deviceName == 'cpu'
    assert backend.array.zeros((2, 2)).device.type == 'cpu'
    assert backend.array.eye(2).device.type == 'cpu'
    assert backend.array.toArray([[1.0]]).device.type == 'cpu'
    assert pickle.loads(pickle.dumps(backend)).deviceName == 'cpu'


def test_thermal_operator_trainer_reads_runtime_device():
    from mor.nn.configs import representationLearningConfig
    from mor.nn.trainers import thermalOperatorTrainerTorch

    config = representationLearningConfig()
    config.runtime.deviceName = 'cpu'
    trainer = thermalOperatorTrainerTorch(lossFunction=object(), config=config)
    assert trainer.deviceName == 'cpu'


def test_shonta_runtime_contract_is_json_compatible():
    contract = getRuntimeContract('cpu')
    assert contract['apiVersion'] == 1
    assert contract['deviceChoices'] == ['auto', 'cpu', 'cuda']
    assert contract['requestedDeviceName'] == 'cpu'
    assert contract['resolvedDeviceName'] == 'cpu'


def test_shonta_dmd_probe_and_error_contract():
    snapshots = [[1.0, 2.0, 4.0, 8.0], [2.0, 1.0, 0.5, 0.25]]
    result = runDmdProbe(snapshots, rank=2)
    assert result['apiVersion'] == 1
    assert result['method'] == 'dmd'
    assert result['shape'] == [2, 4]
    assert result['maxAbsError'] < 1e-10
    with pytest.raises(ValueError, match='at least two columns'):
        runDmdProbe([[1.0]])
