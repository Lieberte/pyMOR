"""mor — a pyMOR-style model order reduction toolkit.

Importing the top-level package makes the *classical* MOR core available
(registries, operators, models, solvers, reductors) and registers the default
``scipy`` backend. The ``torch`` backend and the neural-network reductors live
under :mod:`mor.nn` and are imported on demand (``import mor.nn``) so that
``import mor`` itself does not require torch.
"""

__version__ = "0.1.0"

# Backends: importing this registers the scipy backend (and torch if present).
from . import backends
from .backends.registry import backendRegistry, registerBackend

# Registries (importing each subpackage registers its implementations).
from .algorithm.registry import algorithmRegistry, registerAlgorithm
from .solvers.registry import solverRegistry, registerLyapunovSolver
from .reductors.registry import reductorRegistry, registerReductor

# Operator algebra.
from .operators import (
    operatorBase,
    matrixOperator,
    lowRankOperator,
    sumOperator,
    scaledOperator,
)

# System models.
from .models import modelBase, ltiModel

# Submodules exposed for direct access.
from . import algorithm, solvers, reductors, operators, models

__all__ = [
    "__version__",
    # backends / registry
    "backends",
    "backendRegistry",
    "registerBackend",
    # registries
    "algorithmRegistry",
    "registerAlgorithm",
    "solverRegistry",
    "registerLyapunovSolver",
    "reductorRegistry",
    "registerReductor",
    # operators
    "operatorBase",
    "matrixOperator",
    "lowRankOperator",
    "sumOperator",
    "scaledOperator",
    # models
    "modelBase",
    "ltiModel",
    # submodules
    "algorithm",
    "solvers",
    "reductors",
    "operators",
    "models",
]
