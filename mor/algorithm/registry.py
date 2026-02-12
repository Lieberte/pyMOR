from typing import Dict, Type, Any


class algorithmRegistry:
    _algorithms: Dict[str, Dict[str, Type]] = {}

    @classmethod
    def register(cls, category: str, variant: str, algorithmClass: Type):
        if category not in cls._algorithms:
            cls._algorithms[category] = {}

        cls._algorithms[category][variant] = algorithmClass

    @classmethod
    def get(cls, category: str, variant: str = 'default', **kwargs):
        if category not in cls._algorithms:
            raise ValueError(f"Algorithm category '{category}' not registered")

        if variant not in cls._algorithms[category]:
            available = ', '.join(cls._algorithms[category].keys())
            raise ValueError(f"Variant '{variant}' not found. Available: {available}")

        algorithmClass = cls._algorithms[category][variant]
        return algorithmClass(**kwargs)

    @classmethod
    def list(cls, category: str = None):
        if category is None:
            return list(cls._algorithms.keys())

        if category not in cls._algorithms:
            return []

        return list(cls._algorithms[category].keys())


def registerAlgorithm(category: str, variant: str = 'default'):
    def decorator(algorithmClass):
        algorithmRegistry.register(category, variant, algorithmClass)
        return algorithmClass

    return decorator