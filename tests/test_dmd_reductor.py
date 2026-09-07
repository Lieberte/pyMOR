import numpy as np

from mor.algorithm.dmd import dmdResult
from mor.operators import matrixOperator
from mor.reductors import dmdReductor, reductorRegistry


def test_dmd_reductor_is_registered_and_runs_end_to_end():
    dynamics = np.diag([0.8, 0.95])
    snapshots = [np.array([1.0, 2.0])]
    for _ in range(5):
        snapshots.append(dynamics @ snapshots[-1])
    snapshots = matrixOperator(np.column_stack(snapshots), 'scipy')

    result = dmdReductor('scipy').reduce(snapshots)

    assert isinstance(result, dmdResult)
    assert 'dmd' in reductorRegistry.list()
    np.testing.assert_allclose(result.reconstruct(), snapshots.toNumpy(), atol=1e-11)
