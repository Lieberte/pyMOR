import numpy as np
from scipy import linalg

from mor.algorithm.decompose.truncation.truncation import truncate
from mor.algorithm.registry import registerAlgorithm
from mor.operators.matrix import matrixOperator
from mor.operators.operatorsBase import operatorBase

from .base import dmdResult
from .exact import exactDMD


@registerAlgorithm('dmd', 'tls')
class tlsDMD(exactDMD):
    def fit(
        self,
        snapshots: operatorBase,
        shiftedSnapshots: operatorBase | None = None,
        rank: int | None = None,
        tol: float | None = None,
        dt: float | None = None,
        mode: str = 'exact',
        amplitudeMethod: str = 'first',
        **kwargs,
    ) -> dmdResult:
        self._rejectTrainingTimes(kwargs)
        if self.localBackend.name != 'scipy':
            raise NotImplementedError("TLS-DMD currently supports only the scipy backend.")
        xData, yData, trainingSteps, inputWasReal = self._pairedData(
            snapshots, shiftedSnapshots, dt
        )
        stacked = np.vstack([xData, yData])
        _, singularValues, Vh = linalg.svd(stacked, full_matrices=False)
        requestedRank = truncate(singularValues, rank=rank, tol=tol, backend=self.localBackend)
        numericalTolerance = self._numericalTolerance(singularValues, stacked.shape)
        numericalRank = int(np.count_nonzero(singularValues > numericalTolerance))
        effectiveRank = min(requestedRank, numericalRank)
        if effectiveRank == 0:
            raise ValueError("TLS-DMD requires a positive numerical snapshot rank.")
        V = Vh[:effectiveRank, :].conj().T
        projection = V @ V.conj().T
        projectedX = xData @ projection
        projectedY = yData @ projection
        result = super().fit(
            matrixOperator(projectedX, 'scipy'),
            matrixOperator(projectedY, 'scipy'),
            rank=effectiveRank,
            dt=dt,
            mode=mode,
            amplitudeMethod=amplitudeMethod,
        )
        result.trainingSteps = trainingSteps
        result.inputWasReal = inputWasReal
        result.diagnostics.update({
            'variant': 'tls',
            'tlsRank': effectiveRank,
            'tlsNumericalRank': numericalRank,
            'tlsProjectionResidual': self._relativeNorm(
                stacked - stacked @ projection, stacked
            ),
        })
        if trainingSteps is not None:
            fullSnapshots = np.column_stack([xData, yData[:, -1]])
            result.diagnostics['reconstructionError'] = self._relativeNorm(
                fullSnapshots - result.reconstruct(realOutput=False), fullSnapshots
            )
        return result


TLSDMD = tlsDMD
