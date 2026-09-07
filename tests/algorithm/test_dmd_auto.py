import numpy as np
import pytest

from mor.algorithm.auto.dmd import selectDMDVariant
from mor.algorithm.registry import algorithmRegistry
from mor.operators import matrixOperator
from mor.reductors import dmdReductor


@pytest.mark.parametrize(('options', 'expected'), [
    ({}, 'exact'),
    ({'controlSnapshots': object()}, 'dmdc'),
    ({'delay': 2}, 'hankel'),
    ({'noiseAware': True}, 'tls'),
    ({'bidirectional': True}, 'forwardBackward'),
    ({'optimize': True}, 'optimized'),
])
def test_dmd_auto_selection_is_explicit(options, expected):
    assert selectDMDVariant('scipy', **options) == expected


def test_dmd_auto_rejects_conflicting_variant_requests():
    with pytest.raises(ValueError, match='incompatible requests'):
        selectDMDVariant('scipy', delay=2, noiseAware=True)


def test_dmd_auto_rejects_unverified_torch_variant():
    with pytest.raises(NotImplementedError, match='not verified'):
        selectDMDVariant('torch', noiseAware=True)


def test_algorithm_registry_dispatches_default_dmd_to_exact():
    snapshots = matrixOperator(np.array([[1.0, 0.9, 0.81]]), 'scipy')

    algorithm = algorithmRegistry.get(
        'dmd', 'auto', backendName='scipy', xOperator=snapshots
    )

    assert algorithm.__class__.__name__ == 'exactDMD'


def test_dmd_reductor_uses_auto_by_default():
    snapshots = matrixOperator(np.array([[1.0, 0.9, 0.81, 0.729]]), 'scipy')

    result = dmdReductor('scipy').reduce(snapshots)

    assert result.diagnostics['variant'] == 'exact'
    assert result.diagnostics['requestedMethod'] == 'auto'


def test_dmd_reductor_auto_selects_hankel_only_with_delay():
    snapshots = matrixOperator(np.array([[0.9 ** index for index in range(8)]]), 'scipy')

    result = dmdReductor('scipy').reduce(snapshots, delay=2, rank=1)

    assert result.diagnostics['variant'] == 'hankel'
    assert result.diagnostics['requestedMethod'] == 'auto'
