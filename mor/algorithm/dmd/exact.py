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
        U = U[:, :requestedRank]
        singularValues = singularValues[:requestedRank]
        Vh = Vh[:requestedRank, :]
        if np.any(singularValues == 0):
            raise ValueError("Selected snapshot subspace contains zero singular values.")
        projectedY = yData @ Vh.conj().T / singularValues[None, :]
        reducedOperator = U.conj().T @ projectedY
        eigenvalues, eigenvectors = linalg.eig(reducedOperator)
        modes = projectedY @ eigenvectors
        amplitudes = linalg.lstsq(modes, xData[:, 0])[0]
        return dmdResult(
            modes=matrixOperator(modes, backendName='scipy'),
            eigenvalues=eigenvalues,
            amplitudes=amplitudes,
            reducedOperator=matrixOperator(reducedOperator, backendName='scipy'),
            singularValues=singularValues,
            rank=requestedRank,
            dt=dt,
            diagnostics={'variant': 'exact', 'amplitudeMethod': 'first'},
            trainingSteps=trainingSteps,
            inputWasReal=inputWasReal,
        )
