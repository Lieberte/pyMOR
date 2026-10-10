from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mor.nn.data.trajectoryDataModule import trajectoryDataModule
from mor.nn.losses.oneStepPredictionLoss import oneStepPredictionLoss
from mor.nn.models.rnnDynamicsTorch import rnnDynamicsTorch
from mor.nn.trainers.dynamicsTrainerTorch import dynamicsTrainerTorch
from mor.nn.validation.dynamicsPredictionMetrics import dynamicsPredictionMetrics


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_dynamics_training_clips_gradients_and_validates_on_actual_runtime(device, dtype, monkeypatch):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    values = np.ones((3, 2, 1), dtype=np.float64)
    data = trajectoryDataModule(values, values, values, values, batchSize=2, shuffle=False)
    assert data.trainInputs.dtype == torch.float64
    model = rnnDynamicsTorch(inputDim=1, hiddenDims=[2]).to(device=device, dtype=dtype)
    trainer = dynamicsTrainerTorch(oneStepPredictionLoss(), gradientClip=0.1, epochs=1)
    calls = []
    original = torch.nn.utils.clip_grad_norm_
    def clip(parameters, maximum):
        parameters = list(parameters)
        result = original(parameters, maximum)
        calls.append(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in parameters])).item())
        return result
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', clip)
    before = [p.detach().clone() for p in model.parameters()]
    loss, steps = trainer._runEpoch(model, data.getTrainLoader(), data, device, dtype,
                                   torch.optim.SGD(model.parameters(), lr=0.01))
    assert np.isfinite(loss) and steps == 2 and len(calls) == 2
    assert max(calls) <= 0.10001
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    result = dynamicsPredictionMetrics().evaluate(model, data)
    with torch.no_grad():
        predictions = model(torch.as_tensor(values, device=device, dtype=dtype))[0]
        expected = (predictions.double() - 1.).square().mean().item()
    assert result['mse'] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize('loader', [True, False])
def test_dynamics_metrics_reject_empty_validation(loader):
    values = torch.empty((0, 2, 1))
    data = (trajectoryDataModule(values, values, values, values, shuffle=False) if loader else
            SimpleNamespace(getValidationData=lambda: (values, values)))
    with pytest.raises(ValueError, match='empty'):
        dynamicsPredictionMetrics().evaluate(rnnDynamicsTorch(inputDim=1, hiddenDims=[2]), data)


def test_trajectory_pair_rejects_sample_mismatch():
    with pytest.raises(ValueError, match='same sample count'):
        trajectoryDataModule(np.ones((3, 2, 1)), np.ones((3, 2, 1)),
                             np.ones((4, 2, 1)), np.ones((3, 2, 1)))
