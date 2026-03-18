from .registry import reductorRegistry, registerReductor
from .balancedTruncation import balancedTruncationReductor, reducedSystem
from .pod import podReductor, podResult

reductorRegistry.register('balancedTruncation', balancedTruncationReductor)
reductorRegistry.register('pod', podReductor)

BalancedTruncationReductor = balancedTruncationReductor
ReducedSystem = reducedSystem
PodReductor = podReductor
PodResult = podResult

__all__ = [
    'reductorRegistry',
    'registerReductor',
    'balancedTruncationReductor',
    'reducedSystem',
    'podReductor',
    'podResult',
    'BalancedTruncationReductor',
    'ReducedSystem',
    'PodReductor',
    'PodResult',
]
