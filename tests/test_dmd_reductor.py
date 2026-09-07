import numpy as np

from mor.algorithm import algorithmRegistry
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


def test_dmd_variants_are_registered():
    assert set(algorithmRegistry.list('dmd')) == {
        'exact', 'tls', 'forwardBackward', 'dmdc', 'hankel', 'optimized'
    }


def test_dmd_reductor_dispatches_dmdc_with_controls():
    controls = np.array([[1.0, -0.5, 0.25, 0.1]])
    stateOperator = np.array([[0.8]])
    controlOperator = np.array([[1.2]])
    states = [np.array([0.3])]
    for control in controls.T:
        states.append(stateOperator @ states[-1] + controlOperator @ control)
    snapshots = matrixOperator(np.column_stack(states), 'scipy')

    result = dmdReductor('scipy').reduce(
        snapshots,
        method='dmdc',
        controlSnapshots=matrixOperator(controls, 'scipy'),
        rank=2,
        stateRank=1,
    )

    np.testing.assert_allclose(result.simulate(states[0], controls), snapshots.toNumpy())
