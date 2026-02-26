from typing import Dict, Callable
from mor.operators import matrixOperator
from .lyapunov import selectLyapunovSolver

_autoDispatchers: Dict[str, Callable] = {
    'lyapunov': selectLyapunovSolver
}

def selectSolver(solverType: str, **kwargs) -> str:
    if solverType not in _autoDispatchers:
        raise ValueError(f"Unknown solver type for auto-selection: {solverType}. Available: {list(_autoDispatchers.keys())}")
    return _autoDispatchers[solverType](**kwargs)

def registerAutoDispatcher(solverType: str, dispatcher: Callable):
    _autoDispatchers[solverType] = dispatcher
