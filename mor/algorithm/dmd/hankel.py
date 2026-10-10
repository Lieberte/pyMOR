from dataclasses import dataclass
from numbers import Integral

import numpy as np

from mor.algorithm.registry import registerAlgorithm
from mor.operators.matrix import matrixOperator
from mor.operators.operatorsBase import operatorBase

from .base import dmdResult
from .exact import exactDMD


@dataclass
class hankelDMDResult(dmdResult):
    augmentedModes: matrixOperator | None = None
    delay: int = 1


@registerAlgorithm('dmd', 'hankel')
class hankelDMD(exactDMD):
    def fit(
        self,
        snapshots: operatorBase,
        shiftedSnapshots: operatorBase | None = None,
        rank: int | None = None,
        tol: float | None = None,
        dt: float | None = None,
        delay: int = 2,
        mode: str = 'exact',
        amplitudeMethod: str = 'first',
        maxAugmentedBytes: int | None = None,
        **kwargs,
    ) -> hankelDMDResult:
        self._rejectTrainingTimes(kwargs)
        if self.localBackend.name != 'scipy':
            raise NotImplementedError("Hankel DMD currently supports only the scipy backend.")
        if shiftedSnapshots is not None:
            raise ValueError("Hankel DMD requires a single ordered snapshot sequence.")
        if not isinstance(delay, Integral) or isinstance(delay, bool) or delay < 2:
            raise ValueError("delay must be an integer greater than or equal to 2.")
        data = snapshots.toBackendData()
        if len(getattr(data, 'shape', ())) != 2 or data.shape[1] <= delay:
            raise ValueError("Hankel DMD requires more snapshots than the delay.")
        if not np.all(np.isfinite(data)):
            raise ValueError("Snapshots must contain only finite values.")
        columns = data.shape[1] - delay + 1
        augmentedBytes = int(data.shape[0]) * int(delay) * int(columns) * data.dtype.itemsize
        if maxAugmentedBytes is not None:
            if (not isinstance(maxAugmentedBytes, Integral) or isinstance(maxAugmentedBytes, bool)
                    or maxAugmentedBytes < 0):
                raise ValueError('maxAugmentedBytes must be a non-negative integer or None')
            if augmentedBytes > maxAugmentedBytes:
                raise MemoryError(f'Hankel augmentation requires {augmentedBytes} bytes, exceeding maxAugmentedBytes')
        augmented = np.vstack([data[:, offset:offset + columns] for offset in range(delay)])
        baseResult = super().fit(
            matrixOperator(augmented, 'scipy'),
            rank=rank,
            tol=tol,
            dt=dt,
            mode=mode,
            amplitudeMethod=amplitudeMethod,
        )
        stateModes = baseResult.modes.toNumpy()[:data.shape[0], :]
        result = hankelDMDResult(
            modes=matrixOperator(stateModes, 'scipy'),
            eigenvalues=baseResult.eigenvalues,
            amplitudes=baseResult.amplitudes,
            reducedOperator=baseResult.reducedOperator,
            singularValues=baseResult.singularValues,
            rank=baseResult.rank,
            dt=dt,
            diagnostics={**baseResult.diagnostics, 'variant': 'hankel', 'delay': int(delay),
                         'augmentedBytes': augmentedBytes, 'augmentedShape': augmented.shape},
            trainingSteps=data.shape[1],
            inputWasReal=not np.iscomplexobj(data),
            augmentedModes=baseResult.modes,
            delay=int(delay),
        )
        result.diagnostics['reconstructionError'] = self._relativeNorm(
            data - result.reconstruct(realOutput=False), data
        )
        return result


HankelDMD = hankelDMD
HankelDMDResult = hankelDMDResult
