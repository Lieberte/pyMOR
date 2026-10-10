import numpy as np
import pytest
import torch

from mor.nn.configs import representationLearningConfig
from mor.nn.configs.dynamicsLearningConfig import dynamicsLearningConfig
from mor.nn.reductors.dynamicsLearningPipelineTorch import runDynamicsLearningTorch
from mor.nn.reductors.representationLearningPipelineTorch import runRepresentationLearningTorch
from mor.nn.models.autoEncoderTorch import autoEncoderTorch
from mor.nn.registry import nnRegistry
from mor.nn.runtime.trainingContext import trainingContext


def configFor(model):
    config = dynamicsLearningConfig()
    config.modelName = model
    config.modelParams = dict(inputDim=2, outputDim=2, hiddenDims=[3])
    if model == 'transformerDynamicsTorch':
        config.modelParams = dict(inputDim=2, outputDim=2, dModel=4, nHead=2, numLayers=1)
    if model in ('ssmDynamicsTorch', 'mambaDynamicsTorch'):
        config.modelParams = dict(inputDim=2, outputDim=2, dModel=4, stateDim=3, convWidth=2, numLayers=1)
    config.dataModuleName = 'trajectoryDataModule'
    config.lossFunction = 'oneStepPredictionLoss'
    config.validationName = 'dynamicsPredictionMetrics'
    config.runtime.deviceName = 'cpu'
    config.runtime.dtypeName = 'float64'
    config.runtime.randomSeed = 7
    config.runtime.deterministic = True
    config.epochs = 1
    config.logging.verbose = False
    config.dataParams = dict(batchSize=2, shuffle=False)
    return config


@pytest.mark.parametrize('model', ['rnnDynamicsTorch', 'gruDynamicsTorch', 'lstmDynamicsTorch', 'transformerDynamicsTorch', 'ssmDynamicsTorch', 'mambaDynamicsTorch'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_dynamics_pipeline_is_reproducible_and_preserves_global_settings(model, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    config = configFor(model)
    config.runtime.deviceName = device
    values = np.arange(30, dtype=float).reshape(5, 3, 2) / 30
    state = torch.get_rng_state().clone()
    deterministic = torch.are_deterministic_algorithms_enabled()
    first = runDynamicsLearningTorch(config, values, values, returnModelState=True)
    second = runDynamicsLearningTorch(config, values, values, returnModelState=True)
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert first['validationResult'] == second['validationResult']
    assert np.isfinite(first['validationResult']['mse'])
    assert first['runtimeInfo']['resolvedDeviceName'] == device
    restored = nnRegistry.get('models', model)(**config.modelParams).double()
    restored.loadState(first['modelState'])
    restored.eval()
    output = restored(torch.from_numpy(values))
    if isinstance(output, tuple): output = output[0]
    expected = first['model'](torch.from_numpy(values).to(device))
    if isinstance(expected, tuple): expected = expected[0]
    torch.testing.assert_close(output, expected.cpu(), atol=1e-8, rtol=1e-6)


def test_representation_pipeline_applies_determinism_and_restores_rng():
    config = representationLearningConfig()
    config.modelName, config.dataModuleName = 'autoEncoderTorch', 'snapshotDataModule'
    config.modelParams = dict(inputDim=2, latentDim=1)
    config.lossFunction, config.validationName = 'mseReconstructionLoss', 'reconstructionMetrics'
    config.epochs, config.logging.verbose = 1, False
    config.runtime.deviceName, config.runtime.randomSeed, config.runtime.deterministic = 'cpu', 7, True
    state = torch.get_rng_state().clone()
    with trainingContext(config.runtime):
        assert torch.are_deterministic_algorithms_enabled()
    first = runRepresentationLearningTorch(config, np.ones((5, 2)))
    second = runRepresentationLearningTorch(config, np.ones((5, 2)))
    assert first['validationResult'] == second['validationResult']
    assert torch.equal(torch.get_rng_state(), state)
    original = torch.are_deterministic_algorithms_enabled()
    with pytest.raises(RuntimeError, match='test failure'):
        with trainingContext(config.runtime):
            raise RuntimeError('test failure')
    assert torch.are_deterministic_algorithms_enabled() == original


def test_auto_device_transfer_falls_back_only_when_allowed(monkeypatch):
    import importlib
    module = importlib.import_module('mor.nn.models.torchModelBase')
    monkeypatch.setattr(module, 'resolveDeviceName', lambda name: 'cuda')
    model = autoEncoderTorch(2, 1)
    calls = []
    def transfer(device):
        calls.append(device)
        if device == 'cuda': raise RuntimeError('device transfer failed')
        return model
    monkeypatch.setattr(model, 'to', transfer)
    assert model.toDevice('auto', allowAutoFallback=True) == 'cpu'
    assert calls == ['cuda', 'cpu']
    for requested, allowed in [('auto', False), ('cuda', True)]:
        with pytest.raises(RuntimeError, match='transfer failed'):
            model.toDevice(requested, allowAutoFallback=allowed)


def test_ssm_and_mamba_registry_entries_are_available():
    assert {'ssmDynamicsTorch', 'mambaDynamicsTorch'} <= set(nnRegistry.list('models'))
