from typing import Any

import numpy as np

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
        self._rejectTrainingTimes(kwargs)
        backend = self.localBackend
        xData, yData, trainingSteps, inputWasReal = self._pairedData(snapshots, shiftedSnapshots, dt)
        U, singularValues, Vh = backend.decomposition.svdDense(xData, fullMatrices=False)
        requestedRank = truncate(singularValues, rank=rank, tol=tol, backend=backend)
        if requestedRank == 0:
            raise ValueError("DMD requires a positive numerical snapshot rank.")
        singularValuesNumpy = backend.array.toNumpy(singularValues)
        numericalTolerance = self._numericalTolerance(singularValuesNumpy, xData.shape)
        numericalRank = int(np.count_nonzero(singularValuesNumpy > numericalTolerance))
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
        v = backend.linalg.transpose(backend.linalg.conj(Vh))
        projectedY = backend.linalg.dot(yData, v) / selectedSingularValues[None, :]
        uAdjoint = backend.linalg.transpose(backend.linalg.conj(U))
        reducedOperator = backend.linalg.dot(uAdjoint, projectedY)
        eigenvalues, eigenvectors = backend.eigen.eig(reducedOperator)
        if backend.name == 'torch':
            projectedY = projectedY.to(eigenvectors.dtype)
            U = U.to(eigenvectors.dtype)
        modes = (
            backend.linalg.dot(projectedY, eigenvectors)
            if mode == 'exact'
            else backend.linalg.dot(U, eigenvectors)
        )
        amplitudes = self._fitAmplitudes(
            modes, eigenvalues, xData, amplitudeMethod, backend
        )
        result = dmdResult(
            modes=matrixOperator(modes, backendName=backend.name),
            eigenvalues=eigenvalues,
            amplitudes=amplitudes,
            reducedOperator=matrixOperator(reducedOperator, backendName=backend.name),
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
                'conditionNumber': float(
                    singularValuesNumpy[0] / singularValuesNumpy[effectiveRank - 1]
                ),
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
        modes: Any,
        eigenvalues: Any,
        xData: Any,
        amplitudeMethod: str,
        backend: Any,
    ) -> Any:
        if backend.name == 'torch' and xData.dtype != modes.dtype:
            xData = xData.to(modes.dtype)
        if amplitudeMethod == 'first':
            return backend.linalg.lstsq(modes, xData[:, 0])
        powers = backend.array.array(list(range(xData.shape[1])))
        dynamics = eigenvalues[:, None] ** powers[None, :]
        design = backend.array.hstack([
            backend.array.reshape(
                modes[:, index, None] * dynamics[index, None, :], (-1, 1)
            )
            for index in range(modes.shape[1])
        ])
        return backend.linalg.lstsq(design, backend.array.reshape(xData, (-1,)))

    @staticmethod
    def _relativeNorm(residual: np.ndarray, reference: np.ndarray) -> float:
        denominator = np.linalg.norm(reference)
        return 0.0 if denominator == 0 else float(np.linalg.norm(residual) / denominator)

    def _diagnostics(
        self,
        result: dmdResult,
        xData: Any,
        yData: Any,
        basis: Any,
    ) -> dict[str, float]:
        backend = result.modes.localBackend
        xNumpy = backend.array.toNumpy(xData)
        yNumpy = backend.array.toNumpy(yData)
        basisNumpy = backend.array.toNumpy(basis)
        projectedSnapshots = basisNumpy @ (basisNumpy.conj().T @ xNumpy)
        modalInverse = np.linalg.pinv(result.modes.toNumpy())
        fittedOperator = (
            result.modes.toNumpy()
            @ np.diag(backend.array.toNumpy(result.eigenvalues))
            @ modalInverse
        )
        diagnostics = {
            'snapshotResidual': self._relativeNorm(xNumpy - projectedSnapshots, xNumpy),
            'oneStepResidual': self._relativeNorm(yNumpy - fittedOperator @ xNumpy, yNumpy),
        }
        if result.trainingSteps is not None:
            reconstructed = backend.array.toNumpy(result.reconstruct(realOutput=False))
            fullSnapshots = np.column_stack([xNumpy, yNumpy[:, -1]])
            diagnostics['reconstructionError'] = self._relativeNorm(
                fullSnapshots - reconstructed, fullSnapshots
            )
        return diagnostics
