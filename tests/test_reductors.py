"""End-to-end regression smoke for the classical-MOR primary path (scipy).

Guards against the operator/SVD contract changes breaking POD or balanced
truncation. These were validated manually before; locking them in as tests.
"""
import numpy as np
import pytest

from mor.operators import matrixOperator
from mor.models.lti import ltiModel
from mor.reductors.pod import podReductor
from mor.reductors.balancedTruncation import balancedTruncationReductor


@pytest.fixture
def rng():
    return np.random.default_rng(3)


def test_pod_end_to_end(rng):
    snap = matrixOperator(rng.standard_normal((10, 6)), backendName='scipy')
    result = podReductor(globalBackendName='scipy').reduce(snap, rank=3)
    assert result.order == 3
    ref = np.linalg.svd(snap.toBackendData(), compute_uv=False)
    np.testing.assert_allclose(np.sort(result.S)[::-1], np.sort(ref[:3])[::-1])
    assert result.solverInfo['podSolver'] == 'snapshotPodSolver'


def test_balanced_truncation_end_to_end(rng):
    n = 12
    A = -np.eye(n) + 0.1 * rng.standard_normal((n, n))
    A = (A + A.T) / 2  # symmetric -> real, stable spectrum for the Lyapunov solve
    lti = ltiModel(
        matrixOperator(A, backendName='scipy'),
        matrixOperator(rng.standard_normal((n, 3)), backendName='scipy'),
        matrixOperator(rng.standard_normal((2, n)), backendName='scipy'),
        backendName='scipy',
    )
    rom = balancedTruncationReductor(globalBackendName='scipy').reduce(lti, order=5)
    assert rom.order == 5
    assert len(rom.hsv) == 5
    assert all(h > 0 for h in rom.hsv)
    assert rom.solverInfo['lyapunov'] == 'bartelsStewart'


def test_balanced_truncation_accepts_composed_operators(rng):
    n = 8
    baseA = -np.eye(n) + 0.05 * rng.standard_normal((n, n))
    baseA = (baseA + baseA.T) / 2
    A0 = matrixOperator(baseA, backendName='scipy')
    B0 = matrixOperator(rng.standard_normal((n, n)), backendName='scipy')
    C0 = matrixOperator(rng.standard_normal((n, n)), backendName='scipy')
    lti = ltiModel(
        A0 + (A0 * 0.0),
        B0 + (B0 * 0.0),
        C0 + (C0 * 0.0),
        backendName='scipy',
    )

    rom = balancedTruncationReductor(globalBackendName='scipy').reduce(lti, order=3)

    assert rom.order == 3
    assert rom.Ar.shape == (3, 3)
