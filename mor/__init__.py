"""mor - a pyMOR-style model order reduction toolkit.

Importing the top-level package makes the classical MOR core available
(registries, operators, models, solvers, reductors) and registers the default
``scipy`` backend. The ``torch`` backend is registered unconditionally and loads on demand.
Feature distributions provide the dependencies for their selected capabilities.
"""

__version__ = "0.1.0"

from . import algorithm, applications, backends, models, operators, reductors, solvers
from .algorithm.registry import algorithmRegistry, registerAlgorithm
from .algorithm.dmd import dmdResult, exactDMD
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
from .reductors import dmdReductor, podDmdReductor, podDmdResult
from .runtime import DEVICE_CHOICES, getRuntimeInfo, resolveDevice, resolveDeviceName
from .solvers.registry import registerLyapunovSolver, solverRegistry

__all__ = [
    "__version__",
    "algorithm",
    "applications",
    "backends",
    "models",
    "operators",
    "reductors",
    "solvers",
    "algorithmRegistry",
    "registerAlgorithm",
    "exactDMD",
    "dmdResult",
    "backendRegistry",
    "registerBackend",
    "solverRegistry",
    "registerLyapunovSolver",
    "reductorRegistry",
    "registerReductor",
    "dmdReductor",
    "podDmdReductor",
    "podDmdResult",
    "DEVICE_CHOICES",
    "resolveDevice",
    "resolveDeviceName",
    "getRuntimeInfo",
    "operatorBase",
    "matrixOperator",
    "lowRankOperator",
    "sumOperator",
    "scaledOperator",
    "modelBase",
    "ltiModel",
]
