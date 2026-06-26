from .registry import backendRegistry, registerBackend
from .backendsBase import backendBase

from . import scipyBackend  # default backend (numpy/scipy only, always available)

try:
    from . import torchBackend  # optional; requires torch
except ImportError:
    torchBackend = None  # torch not installed — scipy backend remains usable

__all__ = ['backendRegistry', 'backendBase', 'registerBackend']
