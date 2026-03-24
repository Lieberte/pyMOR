from typing import Any, Optional

def selectSolver(solverType: str, backendName: str = 'scipy', **kwargs) -> str:
    if solverType == 'lyapunov':
        from .lyapunov import selectLyapunovSolver
        return selectLyapunovSolver(backendName=backendName, **kwargs)
    if solverType == 'pod':
        return 'snapshot'
    return 'default'
