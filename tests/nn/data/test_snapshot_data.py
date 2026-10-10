import numpy as np
import pytest

import torch

from mor.nn.data.snapshotDataModule import snapshotDataModule


@pytest.mark.parametrize('split', [-0.5, 1.0, 1.5, np.nan, np.inf, -np.inf, True, False, '0.2', None])
def test_snapshot_split_rejects_invalid_ratios(split):
    with pytest.raises(ValueError, match='validationSplit.*\\[0, 1\\)'):
        snapshotDataModule.fromSnapshots(np.ones((10, 2)), validationSplit=split)


@pytest.mark.parametrize('split,trainCount,validationCount', [(0., 10, 10), (0.2, 8, 2), (0.999, 1, 9)])
def test_snapshot_split_preserves_samples_and_supports_zero_holdout(split, trainCount, validationCount):
    inputs = torch.arange(10, dtype=torch.float32).reshape(-1, 1)
    data = snapshotDataModule.fromSnapshots(inputs, validationSplit=split, shuffle=False)
    train, _ = data.getTrainData()
    validation, _ = data.getValidationData()
    assert len(train) == trainCount
    assert len(validation) == validationCount
    if split == 0:
        torch.testing.assert_close(train, inputs)
        torch.testing.assert_close(validation, inputs)
    else:
        combined = torch.cat((train, validation)).flatten().sort().values
        torch.testing.assert_close(combined, inputs.flatten())


def test_single_snapshot_remains_usable_without_holdout():
    inputs = torch.ones((1, 2))
    data = snapshotDataModule.fromSnapshots(inputs, validationSplit=0.2, shuffle=False)
    torch.testing.assert_close(data.trainInputs, inputs)
    torch.testing.assert_close(data.validationInputs, inputs)


@pytest.mark.parametrize('targetCount', [3, 5])
def test_snapshot_split_rejects_unequal_input_and_target_sample_counts(targetCount):
    with pytest.raises(ValueError, match='same sample count'):
        snapshotDataModule.fromSnapshots(np.ones((4, 2)), np.ones((targetCount, 2)))


@pytest.mark.parametrize('scalarTarget', [False, True])
def test_snapshot_pair_rejects_scalar_without_sample_dimension(scalarTarget):
    with pytest.raises(ValueError, match='sample dimension'):
        snapshotDataModule.fromSnapshots(np.ones((4, 2)) if scalarTarget else 1.,
                                         targets=1. if scalarTarget else None)


def test_snapshot_pair_supports_distinct_input_and_target_feature_counts():
    data = snapshotDataModule.fromSnapshots(np.ones((4, 2)), np.ones((4, 3)), shuffle=False)
    assert data.trainInputs.shape == (3, 2)
    assert data.trainTargets.shape == (3, 3)
