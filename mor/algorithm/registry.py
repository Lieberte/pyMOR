from typing import Dict, Type, Any, Optional


class algorithmRegistry:
    _algorithms: Dict[str, Dict[str, Type]] = {}

    @classmethod
    def register(cls, category: str, variant: str, algorithmClass: Type):
        if category not in cls._algorithms:
            cls._algorithms[category] = {}

        cls._algorithms[category][variant] = algorithmClass

    @classmethod
    def _selectSVDVariant(cls, A=None, E=None, B=None) -> str:
        # TODO: move to mor.algorithm.auto, add variant selection by problem structure
        return 'static'

    @classmethod
    def get(cls, category: str, variant: str = 'default', A=None, E=None, B=None, **kwargs):
        if category not in cls._algorithms:
            raise ValueError(f"Algorithm category '{category}' not registered")
        if variant == 'auto':
            variant = cls._selectSVDVariant(A, E, B) if category == 'svd' else variant
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