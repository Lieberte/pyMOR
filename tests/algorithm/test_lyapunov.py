"""Lyapunov contract tests.

Covers:
  * lrsmith.solve must accept a ``trans`` keyword (the unified solver always
    passes one) -- it previously raised TypeError.
  * dense + discrete auto-dispatch must raise a clear NotImplementedError --
    it previously returned the unregistered name 'discreteHr', producing a
    confusing "Algorithm not registered" error deep in the registry.

Note: lrsmith cannot be *executed* end-to-end yet because it calls several
backend array methods (max/ndim/reshape/...) that do not exist -- a separate,
tracked gap. These tests therefore check the contract at the signature level.
"""
import inspect

import pytest

from mor.algorithm.registry import algorithmRegistry
from mor.algorithm.auto.lyapunov import selectLyapunovAlgorithm


class _FakeA:
    isSparse = False


def test_lrsmith_solve_has_trans_parameter():
    smith = algorithmRegistry.get(category='lyapunov', variant='lrsmith', backendName='scipy')
    params = inspect.signature(smith.solve).parameters
    assert 'trans' in params, "lrsmith.solve must accept trans= (unified solver passes it)"
    assert params['trans'].default is False


def test_dense_discrete_dispatch_raises_not_implemented():
    """Dense discrete Lyapunov has no implementation; fail loudly instead of returning a phantom name."""
    with pytest.raises(NotImplementedError, match='discrete'):
        selectLyapunovAlgorithm(variant='auto', A=_FakeA(), backendName='scipy', isContinuous=False)


def test_explicit_variant_is_passed_through():
    """A non-auto variant must be honored (not routed into the discrete dead-end)."""
    assert selectLyapunovAlgorithm(variant='bartelsStewart', A=_FakeA(), backendName='scipy') == 'bartelsStewart'


def test_dense_continuous_still_dispatches_to_bartels_stewart():
    assert selectLyapunovAlgorithm(variant='auto', A=_FakeA(), backendName='scipy', isContinuous=True) == 'bartelsStewart'
