from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any


DEVICE_CHOICES = ('auto', 'cpu', 'cuda')


@dataclass(frozen=True)
class deviceResolution:
    requestedDeviceName: str
    resolvedDeviceName: str
    cudaAvailable: bool
    cudaVersion: str | None
    gpuName: str | None

    def toDict(self) -> dict[str, Any]:
        return asdict(self)


def normalizeDeviceName(deviceName: str = 'auto') -> str:
    normalized = str(deviceName).strip().lower()
    if normalized not in DEVICE_CHOICES:
        choices = ', '.join(DEVICE_CHOICES)
        raise ValueError(f'unsupported deviceName {deviceName!r}; expected one of: {choices}')
    return normalized


def resolveDevice(deviceName: str = 'auto') -> deviceResolution:
    import torch

    requested = normalizeDeviceName(deviceName)
    cudaAvailable = bool(torch.cuda.is_available())
    if requested == 'cuda' and not cudaAvailable:
        raise RuntimeError(
            'CUDA was explicitly requested, but PyTorch cannot access a CUDA device. '
            'Select auto or cpu, or check the NVIDIA driver and GPU compatibility.'
        )
    resolved = 'cuda' if requested == 'cuda' or (requested == 'auto' and cudaAvailable) else 'cpu'
    gpuName = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if resolved == 'cuda'
        else None
    )
    return deviceResolution(
        requestedDeviceName=requested,
        resolvedDeviceName=resolved,
        cudaAvailable=cudaAvailable,
        cudaVersion=torch.version.cuda,
        gpuName=gpuName,
    )


def resolveDeviceName(deviceName: str = 'auto') -> str:
    return resolveDevice(deviceName).resolvedDeviceName


def getRuntimeInfo(deviceName: str = 'auto') -> dict[str, Any]:
    resolution = resolveDevice(deviceName)

    return {
        'pythonVersion': platform.python_version(),
        'pythonExecutable': sys.executable,
        'architecture': platform.machine(),
        'platform': platform.platform(),
        'packages': {
            'mor': metadata.version('mor'),
            'numpy': metadata.version('numpy'),
            'scipy': metadata.version('scipy'),
            'torch': metadata.version('torch'),
        },
        **resolution.toDict(),
    }
