import numpy as np
from scipy import linalg

from mor.algorithm.decompose.truncation.truncation import truncate
from mor.algorithm.registry import registerAlgorithm
from mor.operators.matrix import matrixOperator
from mor.operators.operatorsBase import operatorBase

from .base import dmdResult
from .exact import exactDMD


@registerAlgorithm('dmd', 'forwardBackward')
class forwardBackwardDMD(exactDMD):
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
            raise NotImplementedError("Forward-backward DMD currently supports only the scipy backend.")
        if mode not in {'exact', 'projected'}:
            raise ValueError("mode must be 'exact' or 'projected'.")
        if amplitudeMethod not in {'first', 'global'}:
            raise ValueError("amplitudeMethod must be 'first' or 'global'.")
        xData, yData, trainingSteps, inputWasReal = self._pairedData(
            snapshots, shiftedSnapshots, dt
        )
        Ux, sx, _ = linalg.svd(xData, full_matrices=False)
        requestedRank = truncate(sx, rank=rank, tol=tol, backend=self.localBackend)
        numericalRank = int(np.count_nonzero(
            sx > self._numericalTolerance(sx, xData.shape)
        ))
        effectiveRank = min(requestedRank, numericalRank)
        if effectiveRank == 0:
            raise ValueError("Forward-backward DMD requires a positive numerical snapshot rank.")
        Ux, sx = Ux[:, :effectiveRank], sx[:effectiveRank]
        xReduced = Ux.conj().T @ xData
        yReduced = Ux.conj().T @ yData
        forward = yReduced @ linalg.pinv(xReduced)
        backward = xReduced @ linalg.pinv(yReduced)
        projectedForward = yData @ linalg.pinv(xReduced)
        try:
            product = forward @ linalg.solve(backward, np.eye(effectiveRank))
        except linalg.LinAlgError as error:
            raise ValueError("Backward reduced operator is singular.") from error
        combined = linalg.sqrtm(product)
        eigenvalues, eigenvectors = linalg.eig(combined)
        modes = projectedForward @ eigenvectors if mode == 'exact' else Ux @ eigenvectors
        amplitudes = self._fitAmplitudes(
            modes, eigenvalues, xData, amplitudeMethod, self.localBackend
        )
        result = dmdResult(
            modes=matrixOperator(modes, 'scipy'),
            eigenvalues=eigenvalues,
            amplitudes=amplitudes,
            reducedOperator=matrixOperator(combined, 'scipy'),
            singularValues=sx,
            rank=effectiveRank,
            dt=dt,
            diagnostics={
                'variant': 'forwardBackward',
                'mode': mode,
                'amplitudeMethod': amplitudeMethod,
                'requestedRank': requestedRank,
                'numericalRank': numericalRank,
                'conditionNumber': float(sx[0] / sx[-1]),
            },
            trainingSteps=trainingSteps,
            inputWasReal=inputWasReal,
        )
        result.diagnostics.update(self._diagnostics(result, xData, yData, Ux))
        return result


ForwardBackwardDMD = forwardBackwardDMD
