from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mor.algorithm.linear.shifted import shiftedLinear
from mor.algorithm.lyapunov.lradi import lradi, shiftComputationOptions
from mor.algorithm.lyapunov.sign import sign
from mor.algorithm.lyapunov_discrete import estimateDiscreteRadius
from mor.operators import matrixOperator


@pytest.mark.parametrize('backendName', ['scipy', 'torch'])
@pytest.mark.parametrize('path', ['sign', 'initial_shifts', 'heuristic_shifts'])
@pytest.mark.parametrize('errorKind', ['numerical', 'dependency', 'programming'])
def test_numerical_fallback_preserves_unexpected_errors(backendName, path, errorKind, monkeypatch):
    solver = sign(backendName) if path == 'sign' else lradi(backendName)
    backend = solver.localBackend
    errorType = {'numerical': backend.linalg.LinAlgError,
                 'dependency': ModuleNotFoundError, 'programming': RuntimeError}[errorKind]
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise errorType('original failure')

    aData = np.eye(2) * (1. if path == 'initial_shifts' else -1.)
    bData = np.ones((2, 1))
    if backendName == 'torch':
        aData, bData = torch.from_numpy(aData), torch.from_numpy(bData)
    A, B = matrixOperator(aData, backendName), matrixOperator(bData, backendName)
    if path == 'sign':
        originalSolve = backend.linalg.solve

        def failOnce(*args, **kwargs):
            return originalSolve(*args, **kwargs) if calls else fail()

        monkeypatch.setattr(backend.linalg, 'solve', failOnce)
        run = lambda: solver.solve(A, None, B)
    elif path == 'initial_shifts':
        monkeypatch.setattr(shiftedLinear, 'solve', fail)
        run = lambda: solver.computeInitialShifts(A, None, bData, shiftComputationOptions(initMaxIter=2))
    else:
        monkeypatch.setattr(backend.eigen, 'eigvalsGeneralized', fail)
        run = lambda: solver.computeHeuristicShifts(A, None)

    if errorKind == 'numerical':
        result = run()
        if path == 'sign':
            np.testing.assert_allclose(result.toNumpy(), np.full((2, 2), 0.5))
            assert solver.lastFallback == 'continuous_direct'
        else:
            values = result.cpu().numpy() if backendName == 'torch' else result
            np.testing.assert_allclose(values, [-1e-3, -1e-2, -1e-1, -1., -10.])
    else:
        with pytest.raises(errorType, match='original failure'):
            run()
    assert calls


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_lradi_shift_probes_follow_input_device(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    solver = lradi('torch')
    aData = -2. * torch.eye(2, dtype=torch.float64, device=device)
    bData = torch.ones((2, 1), dtype=torch.float64, device=device)
    A = matrixOperator(aData, 'torch')
    options = shiftComputationOptions(subspaceColumns=1)
    shifts = solver.computeInitialShifts(A, None, bData, options)
    heuristic = solver.computeHeuristicShifts(A, None)
    updated = solver.updateShifts(A, None, bData, [bData], shifts, options)
    for result in (shifts, heuristic, updated):
        assert result.device == aData.device
        np.testing.assert_allclose(result.cpu().numpy(), -2.)


@pytest.mark.parametrize('errorType', [np.linalg.LinAlgError, ModuleNotFoundError, RuntimeError])
def test_discrete_radius_preserves_unexpected_errors(errorType):
    def fail():
        raise errorType('original failure')

    A = SimpleNamespace(shape=(2, 2), isDiagonal=False, toBackendData=fail)
    if errorType is np.linalg.LinAlgError:
        assert estimateDiscreteRadius(A) is None
    else:
        with pytest.raises(errorType, match='original failure'):
            estimateDiscreteRadius(A)
