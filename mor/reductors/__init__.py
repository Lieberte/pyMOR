from .registry import reductorRegistry, registerReductor
from .balancedTruncation import balancedTruncationReductor, reducedSystem
from .pod import podReductor, podResult
from .frequencyResponse import frequencyResponseReductor
from .dmd import dmdReductor
from .podDmd import podDmdReductor, podDmdResult
from .thermalCoupling import thermalCouplingReductor

reductorRegistry.register('balancedTruncation', balancedTruncationReductor)
reductorRegistry.register('pod', podReductor)
reductorRegistry.register('frequencyResponse', frequencyResponseReductor)
reductorRegistry.register('dmd', dmdReductor)
reductorRegistry.register('podDmd', podDmdReductor)
reductorRegistry.register('thermalCoupling', thermalCouplingReductor)

BalancedTruncationReductor = balancedTruncationReductor
ReducedSystem = reducedSystem
PodReductor = podReductor
PodResult = podResult
FrequencyResponseReductor = frequencyResponseReductor
DMDReductor = dmdReductor
PODDMDReductor = podDmdReductor
PODDMDResult = podDmdResult

__all__ = [
    'reductorRegistry',
    'registerReductor',
    'balancedTruncationReductor',
    'reducedSystem',
    'podReductor',
    'podResult',
    'thermalCouplingReductor',
    'BalancedTruncationReductor',
    'ReducedSystem',
    'PodReductor',
    'PodResult',
    'frequencyResponseReductor',
    'FrequencyResponseReductor',
    'dmdReductor',
    'DMDReductor',
    'podDmdReductor',
    'podDmdResult',
    'PODDMDReductor',
    'PODDMDResult',
]
