from .registry import reductorRegistry, registerReductor
from .balancedTruncation import balancedTruncationReductor, reducedSystem

reductorRegistry.register('balancedTruncation', balancedTruncationReductor)

BalancedTruncationReductor = balancedTruncationReductor
ReducedSystem = reducedSystem

__all__ = [
    'reductorRegistry',
    'registerReductor',
    'balancedTruncationReductor',
    'reducedSystem',
    'BalancedTruncationReductor',
    'ReducedSystem',
]
