"""Shared pytest setup for the classical-MOR test suite.

This branch is based on ``master`` (no public ``mor`` API yet), so we import the
subpackages explicitly to trigger backend/algorithm/solver/reductor registration
via import-time side effects. On a branch where ``mor/__init__`` is populated,
these imports are harmless no-ops.
"""
import mor.backends      # noqa: F401  registers scipy (+ torch if installed)
import mor.operators     # noqa: F401
import mor.algorithm     # noqa: F401  registers svd / lyapunov algorithms
import mor.solvers       # noqa: F401
import mor.reductors     # noqa: F401
