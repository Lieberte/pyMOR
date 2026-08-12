from .registry import reductorRegistry, registerReductor
from .balancedTruncation import balancedTruncationReductor, reducedSystem
from .pod import podReductor, podResult
from .frequencyResponse import frequencyResponseReductor

reductorRegistry.register('balancedTruncation', balancedTruncationReductor)
reductorRegistry.register('pod', podReductor)
reductorRegistry.register('frequencyResponse', frequencyResponseReductor)

BalancedTruncationReductor = balancedTruncationReductor
ReducedSystem = reducedSystem
PodReductor = podReductor
PodResult = podResult
FrequencyResponseReductor = frequencyResponseReductor

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
]
