from typing import Dict, Type, Any, Optional
from mor.backends import backendRegistry
from mor.solvers.auto import selectSolver

class solverRegistry:
    _solvers: Dict[str, Dict[str, Type]] = {}
    _defaultBackends: Dict[str, str] = {}

    @classmethod
    def register(cls, solverType: str, name: str, solverClass: Type):
        if solverType not in cls._solvers:
            cls._solvers[solverType] = {}
        cls._solvers[solverType][name] = solverClass

    @classmethod
    def _resolveBackend(cls, solverType: str, backendName: str | None) -> str:
        if backendName and backendName != 'auto':
            return backendName
        if solverType in cls._defaultBackends:
            return cls._defaultBackends[solverType]
        methodName = f"getPreferredFor{solverType.capitalize()}"
        if hasattr(backendRegistry, methodName):
            return getattr(backendRegistry, methodName)()
        return backendRegistry.getDefaultBackendName()

    @classmethod
    def get(cls, solverType: str, name: str | None = None, forceOptions: dict | None = None, backendName: str | None = None, **kwargs) -> Any:
        backend = cls._resolveBackend(solverType, backendName)
        if forceOptions:
            kwargs.update(forceOptions)
            if 'name' in forceOptions:
                name = forceOptions['name']
        if name is None or name == 'auto':
            name = selectSolver(solverType, backendName=backend, **kwargs)
        if solverType not in cls._solvers or name not in cls._solvers[solverType]:
            raise ValueError(f"Solver '{name}' not registered in {solverType} registry.")
        return cls._solvers[solverType][name](backendName=backend, **kwargs)

    @classmethod
    def list(cls, solverType: str) -> list:
        return list(cls._solvers.get(solverType, {}).keys())

def registerLyapunovSolver(name: str):
    def decorator(solverClass):
        solverRegistry.register('lyapunov', name, solverClass)
        return solverClass
    return decorator
