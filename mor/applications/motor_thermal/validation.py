from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class motorThermalErrorReport:
    relativeL2: float
    maxAbsolute: float
    hotspotAbsolute: float


def evaluateMotorThermalError(
    reference: Any,
    approximation: Any,
    stateAxis: int = 0,
) -> motorThermalErrorReport:
    referenceArray = np.asarray(reference)
    approximationArray = np.asarray(approximation)
    if referenceArray.shape != approximationArray.shape:
        raise ValueError("reference and approximation must have the same shape")
    if referenceArray.size == 0:
        raise ValueError("temperature data must not be empty")
    if not np.all(np.isfinite(referenceArray)) or not np.all(np.isfinite(approximationArray)):
        raise ValueError("temperature data must be finite")

    residual = referenceArray - approximationArray
    referenceNorm = np.linalg.norm(referenceArray.ravel())
    relativeL2 = np.linalg.norm(residual.ravel()) / max(referenceNorm, np.finfo(float).eps)
    maxAbsolute = np.max(np.abs(residual))
    referenceHotspot = np.max(referenceArray, axis=stateAxis)
    approximationHotspot = np.max(approximationArray, axis=stateAxis)
    hotspotAbsolute = np.max(np.abs(referenceHotspot - approximationHotspot))
    return motorThermalErrorReport(
        relativeL2=float(relativeL2),
        maxAbsolute=float(maxAbsolute),
        hotspotAbsolute=float(hotspotAbsolute),
    )
