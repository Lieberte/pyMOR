"""Compatibility exports for thermal operator representations and assembly."""
from .thermalSet import thermalOperatorSet
from mor.assembly.thermal import assembleLinearTetraThermalOperators, applyDirichletConditions

__all__ = ['thermalOperatorSet', 'assembleLinearTetraThermalOperators', 'applyDirichletConditions']
