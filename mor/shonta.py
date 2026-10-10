from __future__ import annotations

from typing import Any

import numpy as np

from mor.operators import matrixOperator
from mor.reductors import dmdReductor
from mor.runtime import DEVICE_CHOICES, getRuntimeInfo


def getRuntimeContract(deviceName: str = 'auto') -> dict[str, Any]:
    """Return the JSON-compatible runtime contract consumed by shonTA."""
    return {
        'apiVersion': 1,
        'deviceChoices': list(DEVICE_CHOICES),
        **getRuntimeInfo(deviceName),
    }


def runDmdProbe(snapshots: Any, rank: int | None = None) -> dict[str, Any]:
    """Run a small DMD calculation for embedded-interpreter integration tests."""
    data = np.asarray(snapshots, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 1 or data.shape[1] < 2:
        raise ValueError('snapshots must be a 2D array with at least two columns')
    if not np.all(np.isfinite(data)):
        raise ValueError('snapshots must contain only finite values')
    if rank is not None and (isinstance(rank, bool) or int(rank) != rank or rank < 1):
        raise ValueError('rank must be a positive integer or None')

    result = dmdReductor('scipy').reduce(
        matrixOperator(data, backendName='scipy'),
        rank=None if rank is None else int(rank),
    )
    reconstruction = np.asarray(result.reconstruct(realOutput=True), dtype=np.float64)
    return {
        'apiVersion': 1,
        'method': 'dmd',
        'rank': int(result.rank),
        'shape': [int(value) for value in reconstruction.shape],
        'reconstruction': reconstruction.tolist(),
        'maxAbsError': float(np.max(np.abs(reconstruction - data))),
    }


def getCapabilities() -> dict[str, Any]:
    from mor.workflow import getCapabilities as implementation

    return implementation()


def validateWorkflow(request: dict[str, Any]) -> dict[str, Any]:
    from mor.workflow import validateWorkflow as implementation

    return implementation(request)


def executeWorkflow(request: dict[str, Any], onEvent=None, isCancelled=None) -> dict[str, Any]:
    from mor.workflow import executeWorkflow as implementation

    return implementation(request, onEvent=onEvent, isCancelled=isCancelled)


def loadPodModel(manifestPath: str):
    from mor.workflow import loadPodModel as implementation

    return implementation(manifestPath)


def loadAutoencoderModel(manifestPath: str):
    from mor.workflow import loadAutoencoderModel as implementation

    return implementation(manifestPath)


def loadDynamicsModel(manifestPath: str):
    from mor.workflow.dynamics import loadDynamicsModel as implementation

    return implementation(manifestPath)
