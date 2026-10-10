from types import SimpleNamespace

import numpy as np
import pytest

import torch

from mor.nn.data.snapshotDataModule import snapshotDataModule
from mor.nn.losses.mseReconstructionLoss import mseReconstructionLoss
from mor.nn.models.autoEncoderTorch import autoEncoderTorch
from mor.nn.trainers.autoEncoderTrainerTorch import autoEncoderTrainerTorch


@pytest.mark.parametrize('dtypeName', ['float32', 'float64'])
def test_trainer_aligns_model_and_data_dtype(dtypeName):
    model = autoEncoderTorch(3, 2, [4])
    data = snapshotDataModule.fromSnapshots(np.ones((5, 3)), batchSize=2, shuffle=False)
    config = SimpleNamespace(epochs=2,
                             runtime=SimpleNamespace(deviceName='cpu', dtypeName=dtypeName),
                             logging=SimpleNamespace(verbose=False))
    result = autoEncoderTrainerTorch(mseReconstructionLoss(), config).fit(model, data)
    assert next(model.parameters()).dtype == getattr(torch, dtypeName)
    assert len(result['trainLossHistory']) == 2
    assert np.isfinite(result['validationLossHistory']).all()


@pytest.mark.parametrize('batchSize', [1, 2, 3])
def test_validation_loss_is_independent_of_batch_partition(batchSize):
    model = autoEncoderTorch(1, 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    x = torch.tensor([[0.], [0.], [3.]])
    data = snapshotDataModule(x, x, batchSize=batchSize, shuffle=False)
    trainer = autoEncoderTrainerTorch(mseReconstructionLoss())
    loss, steps = trainer._runEpoch(model, data.getValidationLoader(), data, 'cpu', torch.float32)
    assert loss == pytest.approx(3.0)
    assert steps == 0


@pytest.mark.parametrize('saveBest,saveLast,earlyStop', [
    (True, True, False), (True, True, True), (False, True, False), (True, False, False),
])
def test_best_and_last_checkpoints_represent_distinct_training_states(saveBest, saveLast, earlyStop, monkeypatch):
    model = autoEncoderTorch(2, 1)
    data = snapshotDataModule.fromSnapshots(np.ones((5, 2)), shuffle=False)
    config = SimpleNamespace(
        epochs=3, runtime=SimpleNamespace(deviceName='cpu'), logging=SimpleNamespace(verbose=False),
        checkpoint=SimpleNamespace(saveBestModel=saveBest, saveLastModel=saveLast),
        earlyStopping=SimpleNamespace(enabled=earlyStop, patience=1),
    )
    trainer = autoEncoderTrainerTorch(mseReconstructionLoss(), config)
    completed = []

    def controlledEpoch(model, loader, dataModule, device, dtype, optimizer=None, globalStep=0):
        if optimizer is not None:
            completed.append(len(completed) + 1)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(float(completed[-1]))
            return float(completed[-1]), globalStep + 1
        return float(completed[-1]), globalStep

    monkeypatch.setattr(trainer, '_runEpoch', controlledEpoch)
    result = trainer.fit(model, data)
    lastEpoch = 2 if earlyStop else 3
    assert result['totalSteps'] == lastEpoch
    assert result['bestEpochIndex'] == 0
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, torch.full_like(tensor, 1. if saveBest else float(lastEpoch)))
        if saveBest:
            torch.testing.assert_close(result['bestModelState'][key], torch.ones_like(tensor))
        if saveLast:
            torch.testing.assert_close(result['lastModelState'][key], torch.full_like(tensor, float(lastEpoch)))
    if not saveBest:
        assert result['bestModelState'] is None
    if not saveLast:
        assert result['lastModelState'] is None


@pytest.mark.parametrize('epochs', [0, -1, True, False, 1.5, '2', None])
def test_trainer_rejects_invalid_epoch_counts(epochs):
    with pytest.raises(ValueError, match='epochs.*positive integer'):
        autoEncoderTrainerTorch(mseReconstructionLoss(), SimpleNamespace(epochs=epochs))


def test_trainer_rejects_invalid_epoch_override():
    with pytest.raises(ValueError, match='epochs.*positive integer'):
        autoEncoderTrainerTorch(mseReconstructionLoss(), epochs=0)


def test_trainer_accepts_numpy_integer_epoch_count():
    trainer = autoEncoderTrainerTorch(mseReconstructionLoss(), epochs=np.int64(2))
    assert trainer.epochs == 2


@pytest.mark.parametrize('emptyPartition', ['training', 'validation'])
def test_fit_rejects_empty_loader_before_building_optimizer(emptyPartition, monkeypatch):
    inputs = torch.ones((2, 2))
    validation = inputs if emptyPartition == 'training' else inputs[:0]
    data = snapshotDataModule(inputs, validation, batchSize=32, shuffle=False,
                              dropLast=emptyPartition == 'training')
    trainer = autoEncoderTrainerTorch(mseReconstructionLoss(), epochs=1, deviceName='cpu')
    monkeypatch.setattr(trainer, '_buildOptimizer', lambda model: pytest.fail('optimizer built for empty loader'))
    with pytest.raises(ValueError, match=emptyPartition + ' loader.*empty'):
        trainer.fit(autoEncoderTorch(2, 1), data)


@pytest.mark.parametrize('training', [False, True])
def test_empty_iterator_is_rejected_without_full_data_fallback(training):
    model = autoEncoderTorch(2, 1)
    trainer = autoEncoderTrainerTorch(mseReconstructionLoss())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) if training else None
    data = SimpleNamespace()
    with pytest.raises(ValueError, match=('training' if training else 'validation') + ' loader.*empty'):
        trainer._runEpoch(model, iter(()), data, 'cpu', torch.float32, optimizer=optimizer)


def test_none_loader_keeps_explicit_full_data_fallback():
    model = autoEncoderTorch(1, 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    inputs = torch.tensor([[0.], [0.], [3.]])
    data = SimpleNamespace(getValidationData=lambda: (inputs, inputs))
    trainer = autoEncoderTrainerTorch(mseReconstructionLoss())
    loss, steps = trainer._runEpoch(model, None, data, 'cpu', torch.float32)
    assert loss == pytest.approx(3.0)
    assert steps == 0


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_epoch_loss_is_weighted_by_samples_for_unequal_batches(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    model = autoEncoderTorch(1, 1).to(device)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    inputs = torch.tensor([[0.], [0.], [3.]])
    data = snapshotDataModule(inputs, inputs, batchSize=2, shuffle=False)
    trainer = autoEncoderTrainerTorch(mseReconstructionLoss())
    loss, steps = trainer._runEpoch(model, data.getValidationLoader(), data, device, torch.float32)
    assert loss == pytest.approx(3.0)
    assert steps == 0


def test_small_training_set_runs_when_drop_last_is_disabled():
    inputs = torch.ones((2, 2))
    data = snapshotDataModule(inputs, inputs, batchSize=32, shuffle=False, dropLast=False)
    config = SimpleNamespace(epochs=1, runtime=SimpleNamespace(deviceName='cpu'),
                             logging=SimpleNamespace(verbose=False))
    result = autoEncoderTrainerTorch(mseReconstructionLoss(), config).fit(autoEncoderTorch(2, 1), data)
    assert result['totalSteps'] == 1
