import pytest
import torch

from scripts.train_geometry_dgcnn import (
    _knownAcceptanceThreshold,
    _openSetMetrics,
)


def test_known_acceptance_threshold_does_not_use_unknown_scores():
    threshold = _knownAcceptanceThreshold(
        torch.tensor([0.95, 0.90, 0.85, 0.40]), 0.75
    )

    assert threshold == pytest.approx(0.85)


def test_open_set_metrics_count_rejection_and_classification_errors():
    result = _openSetMetrics(
        torch.tensor([[3.0, 1.0], [1.0, 3.0], [3.0, 1.0]]),
        torch.tensor([0, 1, 1]),
        torch.tensor([0.90, 0.80, 0.70]),
        torch.tensor([0.20, 0.60]),
        0.75,
    )

    assert result['knownAcceptanceRate'] == pytest.approx(2 / 3)
    assert result['knownRecognitionRate'] == pytest.approx(2 / 3)
    assert result['unknownRejectionRate'] == 1.0
    assert result['overallOpenSetAccuracy'] == pytest.approx(4 / 5)
