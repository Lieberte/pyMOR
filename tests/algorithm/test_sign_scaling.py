import torch
import numpy as np
import pytest

from mor.algorithm.lyapunov.sign import sign
from mor.operators import matrixOperator


@pytest.mark.parametrize('device', ['scipy', 'cpu', 'cuda'])
@pytest.mark.parametrize('scale', [1e-8, 1e8])
def test_sign_inverse_determinant_scaling_and_device(scale, device):
    a = -scale * np.eye(8)
    b = np.arange(16, dtype=float).reshape(8, 2) / 16
    if device != 'scipy':
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA unavailable')
        aData, bData = torch.tensor(a, device=device), torch.tensor(b, device=device)
        backend = 'torch'
    else:
        aData, bData, backend = a, b, 'scipy'
    solver = sign(backend, maxIter=10, tol=1e-12)
    x = solver.solve(matrixOperator(aData, backend), None, matrixOperator(bData, backend)).toNumpy()
    np.testing.assert_allclose(x, b @ b.T / (2 * scale), rtol=1e-12, atol=1e-18)
    assert solver.lastFallback is None
    assert solver.lastIterations <= 2
    assert solver.lastRelativeResidual < 1e-12
