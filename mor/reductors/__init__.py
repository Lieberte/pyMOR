from .registry import reductorRegistry, registerReductor
from .balancedTruncation import balancedTruncationReductor, reducedSystem
from .pod import podReductor, podResult
from .frequencyResponse import frequencyResponseReductor
from .dmd import dmdReductor

reductorRegistry.register('balancedTruncation', balancedTruncationReductor)
reductorRegistry.register('pod', podReductor)
reductorRegistry.register('frequencyResponse', frequencyResponseReductor)
reductorRegistry.register('dmd', dmdReductor)

BalancedTruncationReductor = balancedTruncationReductor
ReducedSystem = reducedSystem
PodReductor = podReductor
PodResult = podResult
FrequencyResponseReductor = frequencyResponseReductor
DMDReductor = dmdReductor

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
    'frequencyResponseReductor',
    'FrequencyResponseReductor',
    'dmdReductor',
    'DMDReductor',
]
