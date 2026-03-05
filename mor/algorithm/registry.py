from typing import Dict, Type, Any, Optional

from mor.backends import backendRegistry
from mor.algorithm.auto import selectAlgorithm


class algorithmRegistry:
    _algorithms, _defaultBackends = {}, {}

    @classmethod
    def register(cls, category: str, variant: str, algorithmClass: Type):
        if category not in cls._algorithms: cls._algorithms[category] = {}
        cls._algorithms[category][variant] = algorithmClass

    @classmethod
    def resolveBackend(cls, category: str, backendName: str | None) -> str:
        if backendName and backendName != 'auto': return backendName
        if category in cls._defaultBackends: return cls._defaultBackends[category]
        # Check if the current backend is already set in registry
        defaultName = backendRegistry.getDefaultBackendName()
        if defaultName: return defaultName
        methodName = f"getPreferredFor{category.capitalize()}"
        if hasattr(backendRegistry, methodName): return getattr(backendRegistry, methodName)()
        return 'scipy'

    @classmethod
    def get(cls, category: str, variant: str | None = None, forceOptions: dict | None = None, backendName: str | None = None, **kwargs) -> Any:
        backend = cls.resolveBackend(category, backendName)
        if forceOptions:
            kwargs.update(forceOptions)
            if 'variant' in forceOptions: variant = forceOptions['variant']
        if variant is None or variant == 'auto': variant = selectAlgorithm(category, backendName=backend, **kwargs)
        if category not in cls._algorithms or variant not in cls._algorithms[category]: raise ValueError(f"Algorithm '{variant}' not registered in {category} registry.")
        return cls._algorithms[category][variant](backendName=backend, **kwargs)

    @classmethod
    def list(cls, category: str) -> list: return list(cls._algorithms.get(category, {}).keys())

def registerAlgorithm(category: str, variant: str = 'default'):
    def decorator(algorithmClass):
        algorithmRegistry.register(category, variant, algorithmClass)
        return algorithmClass
    return decorator
