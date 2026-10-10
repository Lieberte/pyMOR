from typing import Dict, Type, Optional, Any
from importlib import import_module

class backendRegistry:
    _backends: Dict[str, Type] = {}
    _default: Optional[str] = None
    _priority: Dict[str, int] = {}
    _lazyBackends: Dict[str, str] = {}

    @classmethod
    def register(cls, name: str, backendClass: Type, priority: int = 0):
        cls._backends[name] = backendClass
        cls._lazyBackends.pop(name, None)
        cls._priority[name] = priority
        if cls._default is None:
            cls._default = name

    @classmethod
    def registerLazy(cls, name: str, moduleName: str, priority: int = 0):
        if name not in cls._backends:
            cls._lazyBackends[name] = moduleName
            cls._priority[name] = priority
            if cls._default is None:
                cls._default = name

    @classmethod
    def get(cls, name: str = None, **kwargs):
        backendName = name or cls._default
        moduleName = cls._lazyBackends.get(backendName)
        if moduleName is not None:
            import_module(moduleName)
        if backendName not in cls._backends:
            raise ValueError(f"Backend '{backendName}' not registered")
        return cls._backends[backendName](**kwargs)

    @classmethod
    def list(cls) -> list:
        return list(cls._backends) + list(cls._lazyBackends)

    @classmethod
    def setDefault(cls, name: str):
        if name not in cls._backends and name not in cls._lazyBackends:
            raise ValueError(f"Backend '{name}' not registered")
        cls._default = name

    @classmethod
    def getDefaultBackendName(cls) -> Optional[str]:
        return cls._default

    @classmethod
    def getPreferredForLyapunov(cls) -> str:
        # Just return default or highest priority, as all backends now support Lyapunov via Algorithm layer
        names = sorted(cls.list(), key=lambda n: -cls._priority.get(n, 0))
        return cls._default or (names[0] if names else None)

def registerBackend(name: str, priority: int = 0):
    def decorator(backendClass):
        backendRegistry.register(name, backendClass, priority=priority)
        return backendClass
    return decorator
