from typing import Any, Optional
from mor.solvers.registry import solverRegistry

def selectSolver(solverType: str, backendName: str = 'scipy', **kwargs) -> str:
    if solverType == 'lyapunov':
        from .lyapunov import selectLyapunovSolver
        return selectLyapunovSolver(backendName, **kwargs)
    if solverType == 'pod':
        return 'snapshot'
    return 'default'
