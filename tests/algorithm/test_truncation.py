import torch
import numpy as np
import pytest

from mor.backends import backendRegistry
from mor.algorithm.decompose.truncation.truncation import truncate
from mor.operators import matrixOperator
from mor.reductors.pod import podReductor


@pytest.mark.parametrize('backendName', ['scipy', 'torch'])
@pytest.mark.parametrize('scale', [1., 1e200, 1e-200])
@pytest.mark.parametrize('tol, expected', [(0., 2), (0.02, 1), (1., 1)])
def test_energy_truncation_is_invariant_to_finite_scaling(backendName, scale, tol, expected):
    backend = backendRegistry.get(backendName)
    spectrum = np.array([scale, scale / 10])
    if backendName == 'torch':
        spectrum = torch.from_numpy(spectrum)
    assert truncate(spectrum, tol=tol, backend=backend) == expected


@pytest.mark.parametrize('backendName', ['scipy', 'torch'])
@pytest.mark.parametrize('values', [[], [0., 0.]])
def test_empty_and_zero_spectrum_has_zero_energy_rank(backendName, values):
    backend = backendRegistry.get(backendName)
    spectrum = np.array(values)
    if backendName == 'torch':
        spectrum = torch.from_numpy(spectrum)
    assert truncate(spectrum, tol=0., backend=backend) == 0


@pytest.mark.parametrize('scale', [1., 1e200, 1e-200])
def test_pod_tolerance_keeps_same_order_at_extreme_scales(scale):
    snapshots = matrixOperator(np.diag([scale, scale / 10]), 'scipy')
    result = podReductor('scipy').reduce(snapshots, tol=0.02, method='economic')
    assert result.order == 1
    assert result.reconstructionError == pytest.approx(1 / np.sqrt(101))
