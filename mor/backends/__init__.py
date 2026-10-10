from .registry import backendRegistry, registerBackend
from .backendsBase import backendBase

from . import scipyBackend

backendRegistry.registerLazy('torch', 'mor.backends.torchBackend', priority=70)

__all__ = ['backendRegistry', 'backendBase', 'registerBackend']
