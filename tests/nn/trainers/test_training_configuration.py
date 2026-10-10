import numpy as np
import pytest
import torch

from mor.nn.configs import representationLearningConfig
from mor.nn.data.snapshotDataModule import snapshotDataModule
from mor.nn.losses.mseReconstructionLoss import mseReconstructionLoss
from mor.nn.models.autoEncoderTorch import autoEncoderTorch
from mor.nn.trainers.autoEncoderTrainerTorch import autoEncoderTrainerTorch


@pytest.mark.parametrize('scheduler, expected', [('none', [0.1] * 3), ('step', [0.05, 0.025, 0.0125]),
    ('multistep', [0.05, 0.025, 0.025]), ('cosine', [0.075, 0.025, 0.])])
@pytest.mark.parametrize('optimizer', ['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad'])
def test_scheduler_learning_rate_history_for_each_optimizer(scheduler, expected, optimizer):
    config = representationLearningConfig()
    config.runtime.deviceName = 'cpu'
    config.epochs, config.logging.verbose = 3, False
    config.optimizer.optimizerName, config.optimizer.learningRate = optimizer, 0.1
    config.scheduler.schedulerName, config.scheduler.stepSize = scheduler, 1
    config.scheduler.gamma, config.scheduler.milestones, config.scheduler.tMax = 0.5, [1, 2], 3
    data = snapshotDataModule(torch.ones(4, 2), torch.ones(2, 2))
    result = autoEncoderTrainerTorch(mseReconstructionLoss(), config).fit(autoEncoderTorch(2, 1, [3]), data)
    np.testing.assert_allclose(result['learningRateHistory'], expected, atol=1e-12)


@pytest.mark.parametrize('train,val', [(True, False), (False, True), (False, False)])
def test_logging_respects_report_flags(train, val, capsys):
    config = representationLearningConfig()
    config.logging.logInterval = 2
    config.logging.reportTrainLoss, config.logging.reportValidationLoss = train, val
    trainer = autoEncoderTrainerTorch(mseReconstructionLoss(), config)
    trainer._logEpoch(1, 1., 2., 0.1)
    output = capsys.readouterr().out
    assert ('train=' in output) == train
    assert ('val=' in output) == val


@pytest.mark.parametrize('field,value', [('schedulerName', 'unknown'), ('stepSize', 0), ('tMax', 0),
    ('gamma', float('nan')), ('milestones', [-1])])
def test_invalid_scheduler_configuration_is_rejected(field, value):
    config = representationLearningConfig()
    setattr(config.scheduler, field, value)
    with pytest.raises(ValueError): autoEncoderTrainerTorch(mseReconstructionLoss(), config)
