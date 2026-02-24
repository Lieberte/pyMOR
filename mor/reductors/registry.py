from typing import Dict, Type, Any

from mor.backends import backendRegistry


class reductorRegistry:
    _reductors: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str, reductorClass: Type):
        cls._reductors[name] = reductorClass

    @classmethod
    def get(cls, name: str, forceOptions: dict | None = None, globalBackendName: str | None = None, **kwargs) -> Any:
        if forceOptions:
            kwargs.update(forceOptions)
            if 'name' in forceOptions:
                name = forceOptions['name']
        if name not in cls._reductors:
            raise ValueError(f"Reductor '{name}' not registered. Available: {list(cls._reductors.keys())}")
        return cls._reductors[name](globalBackendName=globalBackendName, **kwargs)

    @classmethod
    def list(cls) -> list:
        return list(cls._reductors.keys())

def registerReductor(name: str):
    def decorator(reductorClass):
        reductorRegistry.register(name, reductorClass)
        return reductorClass
    return decorator
