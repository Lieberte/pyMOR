"""mor - a pyMOR-style model order reduction toolkit.

Importing the top-level package makes the classical MOR core available
(registries, operators, models, solvers, reductors) and registers the default
``scipy`` backend. The ``torch`` backend and the neural-network stack remain
optional and load on demand.
"""

__version__ = "0.1.0"

from . import algorithm, backends, models, operators, reductors, solvers
from .algorithm.registry import algorithmRegistry, registerAlgorithm
from .backends.registry import backendRegistry, registerBackend
from .models import ltiModel, modelBase
from .operators import (
    lowRankOperator,
    matrixOperator,
    operatorBase,
    scaledOperator,
    sumOperator,
)
from .reductors.registry import reductorRegistry, registerReductor
from .solvers.registry import registerLyapunovSolver, solverRegistry

__all__ = [
    "__version__",
    "algorithm",
    "backends",
    "models",
    "operators",
    "reductors",
    "solvers",
    "algorithmRegistry",
    "registerAlgorithm",
    "backendRegistry",
    "registerBackend",
    "solverRegistry",
    "registerLyapunovSolver",
    "reductorRegistry",
    "registerReductor",
    "operatorBase",
    "matrixOperator",
    "lowRankOperator",
    "sumOperator",
    "scaledOperator",
    "modelBase",
    "ltiModel",
]
