from types import SimpleNamespace

import numpy as np
import pytest

import torch

from mor.nn.data.snapshotDataModule import snapshotDataModule
from mor.nn.losses.mseReconstructionLoss import mseReconstructionLoss
from mor.nn.models.autoEncoderTorch import autoEncoderTorch
from mor.nn.validation.reconstructionMetrics import reconstructionMetrics


@pytest.mark.parametrize('batchSize', [1, 2, 3])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_reconstruction_metrics_match_whole_dataset_with_unequal_batches(batchSize, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    model = autoEncoderTorch(1, 1).to(device)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    data = torch.tensor([[0.], [0.], [3.]])
    module = snapshotDataModule(data, data, batchSize=batchSize, shuffle=False)
    result = reconstructionMetrics(deviceName=device, lossFunction=mseReconstructionLoss()).evaluate(model, module)
    assert result == pytest.approx({'mse': 3., 'relativeError': 1., 'validationLoss': 3.})


def test_metrics_full_data_fallback_preserves_float64_input():
    model = autoEncoderTorch(1, 1).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    inputs = np.array([[1.0000000001], [1.0000000002]], dtype=np.float64)
    module = SimpleNamespace(getValidationData=lambda: (inputs, inputs))
    config = SimpleNamespace(runtime=SimpleNamespace(dtypeName='float64'))
    result = reconstructionMetrics(deviceName='cpu', lossFunction=mseReconstructionLoss(), config=config).evaluate(model, module)
    assert result['mse'] == pytest.approx(np.mean(inputs ** 2), rel=1e-13)
    assert result['validationLoss'] == pytest.approx(np.mean(inputs ** 2), rel=1e-13)


@pytest.mark.parametrize('useLoader', [False, True])
def test_metrics_reject_empty_validation_data(useLoader):
    values = torch.empty((0, 1))
    module = (snapshotDataModule(values, values, shuffle=False) if useLoader else
              SimpleNamespace(getValidationData=lambda: (values, values)))
    with pytest.raises(ValueError, match='validation data is empty'):
        reconstructionMetrics(deviceName='cpu').evaluate(autoEncoderTorch(1, 1), module)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('useLoader', [False, True])
@pytest.mark.parametrize('dtype, value', [('float16', 300.), ('float32', 1e20)])
def test_metrics_square_in_float64_to_avoid_low_precision_overflow(device, useLoader, dtype, value):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    model = autoEncoderTorch(1, 1).to(device=device, dtype=getattr(torch, dtype))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    values = torch.full((3, 1), value, dtype=getattr(torch, dtype))
    module = (snapshotDataModule(values, values, batchSize=2, shuffle=False) if useLoader else
              SimpleNamespace(getValidationData=lambda: (values, values)))
    config = SimpleNamespace(runtime=SimpleNamespace(dtypeName=dtype))
    result = reconstructionMetrics(deviceName=device, config=config,
                                   lossFunction=mseReconstructionLoss()).evaluate(model, module)
    expected = float(values[0, 0].double().square())
    assert result == pytest.approx({'mse': expected, 'relativeError': 1., 'validationLoss': expected})
