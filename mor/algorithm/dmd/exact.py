from typing import Any

import numpy as np
from scipy import linalg

from mor.algorithm.decompose.truncation.truncation import truncate
from mor.algorithm.registry import registerAlgorithm
from mor.operators.matrix import matrixOperator
from mor.operators.operatorsBase import operatorBase

from .base import dmd, dmdResult


@registerAlgorithm('dmd', 'exact')
class exactDMD(dmd):
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
        if self.localBackend.name != 'scipy':
            raise NotImplementedError("Exact DMD initially supports only the scipy backend.")
        xData, yData, trainingSteps, inputWasReal = self._pairedData(snapshots, shiftedSnapshots, dt)
        xData = self.localBackend.array.toNumpy(xData)
        yData = self.localBackend.array.toNumpy(yData)
        U, singularValues, Vh = linalg.svd(xData, full_matrices=False)
        requestedRank = truncate(singularValues, rank=rank, tol=tol, backend=self.localBackend)
        if requestedRank == 0:
            raise ValueError("DMD requires a positive numerical snapshot rank.")
        numericalTolerance = self._numericalTolerance(singularValues, xData.shape)
        numericalRank = int(np.count_nonzero(singularValues > numericalTolerance))
        effectiveRank = min(requestedRank, numericalRank)
        if effectiveRank == 0:
            raise ValueError("DMD requires a positive numerical snapshot rank.")
        if mode not in {'exact', 'projected'}:
            raise ValueError("mode must be 'exact' or 'projected'.")
        if amplitudeMethod not in {'first', 'global'}:
            raise ValueError("amplitudeMethod must be 'first' or 'global'.")
        U = U[:, :effectiveRank]
        selectedSingularValues = singularValues[:effectiveRank]
        Vh = Vh[:effectiveRank, :]
        projectedY = yData @ Vh.conj().T / selectedSingularValues[None, :]
        reducedOperator = U.conj().T @ projectedY
        eigenvalues, eigenvectors = linalg.eig(reducedOperator)
        modes = projectedY @ eigenvectors if mode == 'exact' else U @ eigenvectors
        amplitudes = self._fitAmplitudes(modes, eigenvalues, xData, amplitudeMethod)
        result = dmdResult(
            modes=matrixOperator(modes, backendName='scipy'),
            eigenvalues=eigenvalues,
            amplitudes=amplitudes,
            reducedOperator=matrixOperator(reducedOperator, backendName='scipy'),
            singularValues=selectedSingularValues,
            rank=effectiveRank,
            dt=dt,
            diagnostics={
                'variant': 'exact',
                'mode': mode,
                'amplitudeMethod': amplitudeMethod,
                'requestedRank': requestedRank,
                'numericalRank': numericalRank,
                'numericalTolerance': numericalTolerance,
                'conditionNumber': float(selectedSingularValues[0] / selectedSingularValues[-1]),
            },
            trainingSteps=trainingSteps,
            inputWasReal=inputWasReal,
        )
        result.diagnostics.update(self._diagnostics(result, xData, yData, U))
        return result

    @staticmethod
    def _numericalTolerance(singularValues: np.ndarray, shape: tuple[int, int]) -> float:
        if singularValues.size == 0 or singularValues[0] == 0:
            return 0.0
        realDtype = singularValues.real.dtype
        return float(max(shape) * np.finfo(realDtype).eps * singularValues[0])

    @staticmethod
    def _fitAmplitudes(
        modes: np.ndarray,
        eigenvalues: np.ndarray,
        xData: np.ndarray,
        amplitudeMethod: str,
    ) -> np.ndarray:
        if amplitudeMethod == 'first':
            return linalg.lstsq(modes, xData[:, 0])[0]
        powers = np.arange(xData.shape[1])
        dynamics = eigenvalues[:, None] ** powers[None, :]
        design = np.column_stack([
            (modes[:, index, None] * dynamics[index, None, :]).reshape(-1)
            for index in range(modes.shape[1])
        ])
        return linalg.lstsq(design, xData.reshape(-1))[0]

    @staticmethod
    def _relativeNorm(residual: np.ndarray, reference: np.ndarray) -> float:
        denominator = linalg.norm(reference)
        return 0.0 if denominator == 0 else float(linalg.norm(residual) / denominator)

    def _diagnostics(
        self,
        result: dmdResult,
        xData: np.ndarray,
        yData: np.ndarray,
        basis: np.ndarray,
    ) -> dict[str, float]:
        projectedSnapshots = basis @ (basis.conj().T @ xData)
        modalInverse = linalg.pinv(result.modes.toNumpy())
        fittedOperator = (
            result.modes.toNumpy()
            @ np.diag(result.eigenvalues)
            @ modalInverse
        )
        diagnostics = {
            'snapshotResidual': self._relativeNorm(xData - projectedSnapshots, xData),
            'oneStepResidual': self._relativeNorm(yData - fittedOperator @ xData, yData),
        }
        if result.trainingSteps is not None:
            reconstructed = result.reconstruct(realOutput=False)
            fullSnapshots = np.column_stack([xData, yData[:, -1]])
            diagnostics['reconstructionError'] = self._relativeNorm(
                fullSnapshots - reconstructed, fullSnapshots
            )
        return diagnostics
