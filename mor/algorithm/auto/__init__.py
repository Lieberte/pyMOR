from typing import Dict, Callable

from mor.algorithm.auto.svd import selectSVDVariant
from mor.algorithm.auto.lyapunov import selectLyapunovAlgorithm
from mor.algorithm.auto.linear import selectLinearAlgorithm

_autoDispatchers: Dict[str, Callable] = {
    'svd': selectSVDVariant,
    'lyapunov': selectLyapunovAlgorithm,
    'linear': selectLinearAlgorithm
}

def selectAlgorithm(category: str, **kwargs) -> str:
    if category not in _autoDispatchers:
        raise ValueError(f"Unknown algorithm category for auto-selection: {category}. Available: {list(_autoDispatchers.keys())}")
    return _autoDispatchers[category](**kwargs)

def registerAutoDispatcher(category: str, dispatcher: Callable):
    _autoDispatchers[category] = dispatcher
