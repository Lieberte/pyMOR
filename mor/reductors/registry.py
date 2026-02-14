"""Reductor registry for layered selection."""
from typing import Dict, Type, Optional, Any

from mor.backends import backendRegistry


class reductorRegistry:
    _reductors: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str, reductorClass: Type):
        cls._reductors[name] = reductorClass

    @classmethod
    def get(cls, name: str, backendName: Optional[str] = None, **kwargs) -> Any:
        if name not in cls._reductors:
            raise ValueError(
                f"Reductor '{name}' not registered. "
                f"Available: {list(cls._reductors.keys())}"
            )
        return cls._reductors[name](backendName=backendName, **kwargs)

    @classmethod
    def list(cls) -> list:
        return list(cls._reductors.keys())


def registerReductor(name: str):
    def decorator(reductorClass):
        reductorRegistry.register(name, reductorClass)
        return reductorClass
    return decorator
