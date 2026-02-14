from typing import Dict, Type, Optional, Any

from mor.backends import backendRegistry

class lyapunovRegistry:
    _solvers: Dict[str, Type] = {}
    _defaultBackend: Optional[str] = None

    @classmethod
    def register(cls, name: str, solverClass: Type):
        cls._solvers[name] = solverClass

    @classmethod
    def _resolveBackend(cls, backendName: Optional[str]) -> str:
        if backendName and backendName != 'auto':
            return backendName
        return cls._defaultBackend or backendRegistry.getPreferredForLyapunov()

    @classmethod
    def _selectSolver(cls, A, E, B, isContinuous: bool = True) -> str:
        # TODO: move to mor.solvers.lyapunov.auto
        if isContinuous:
            return 'continuousLrGeneralized' if E is not None else 'continuousLr'
        return 'discreteLrGeneralized' if E is not None else 'discreteLr'

    @classmethod
    def get(cls, name: Optional[str] = None, backendName: Optional[str] = None, A=None, E=None, B=None, isContinuous: bool = True, **kwargs) -> Any:
        if name is None:
            name = cls._selectSolver(A, E, B, isContinuous)
        if name not in cls._solvers:
            raise ValueError(
                f"Lyapunov solver '{name}' not registered. "
                f"Available: {list(cls._solvers.keys())}"
            )
        backend = cls._resolveBackend(backendName)
        return cls._solvers[name](backendName=backend, **kwargs)

    @classmethod
    def list(cls) -> list:
        return list(cls._solvers.keys())

    @classmethod
    def setDefaultBackend(cls, name: str):
        if name not in backendRegistry.list():
            raise ValueError(f"Backend '{name}' not registered")
        cls._defaultBackend = name

    @classmethod
    def getDefaultBackend(cls) -> Optional[str]:
        return cls._defaultBackend


def registerLyapunovSolver(name: str):
    def decorator(solverClass):
        lyapunovRegistry.register(name, solverClass)
        return solverClass
    return decorator
