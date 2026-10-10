import time

import numpy as np
import pytest
from scipy import linalg
from scipy.sparse import diags

from mor.algorithm.registry import algorithmRegistry
from mor.operators import matrixOperator, assembleLinearTetraThermalOperators


@pytest.mark.benchmark
def test_repeated_shift_lradi_has_small_residual_and_timing():
    a = matrixOperator(diags(-np.linspace(1., 4., 32), format='csr'), 'scipy')
    b = matrixOperator(np.ones((32, 1)), 'scipy')
    solver = algorithmRegistry.get('lyapunov', 'lradi', backendName='scipy', maxIter=80, tol=1e-12)
    solver.computeInitialShifts = lambda *args: np.array([-1., -2.])
    solver.updateShifts = lambda *args, **kwargs: np.array([-1., -2.])
    start = time.perf_counter()
    x = solver.solve(a, None, b).toBackendData()
    elapsed = time.perf_counter() - start
    residual = a.toBackendData() @ x + x @ a.toBackendData().T + b.toBackendData() @ b.toBackendData().T
    assert np.linalg.norm(residual) / 32 < 1e-10
    assert solver.lastConverged and elapsed > 0


@pytest.mark.benchmark
def test_dense_discrete_direct_timing_and_residual():
    a = np.diag(np.linspace(0.1, 0.9, 24))
    b = np.ones((24, 2))
    solver = algorithmRegistry.get('lyapunov', 'auto', backendName='scipy', isContinuous=False,
                                   A=matrixOperator(a, 'scipy'))
    start = time.perf_counter()
    x = solver.solve(matrixOperator(a, 'scipy'), None, matrixOperator(b, 'scipy')).toBackendData()
    elapsed = time.perf_counter() - start
    np.testing.assert_allclose(x, linalg.solve_discrete_lyapunov(a, b @ b.T), rtol=1e-12)
    assert solver.lastResidual < 1e-10 and elapsed > 0


@pytest.mark.benchmark
def test_many_batch_thermal_assembly_preserves_capacity_and_heat():
    nodes = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    elements = np.tile([[0, 1, 2, 3]], (65, 1))
    start = time.perf_counter()
    result = assembleLinearTetraThermalOperators(nodes, elements, conductivity=2., density=3.,
        specificHeat=4., heatSource=6., assemblyBatchSize=3)
    elapsed = time.perf_counter() - start
    assert result.K.isSparse and result.M.isSparse
    assert result.M.toNumpy().sum() == pytest.approx(65 * 2.)
    assert result.Q.toNumpy().sum() == pytest.approx(65.)
    assert elapsed > 0
