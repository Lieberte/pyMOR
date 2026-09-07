from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy import linalg

from mor.algorithm.decompose.truncation.truncation import truncate
from mor.algorithm.registry import registerAlgorithm
from mor.operators.matrix import matrixOperator
from mor.operators.operatorsBase import operatorBase

from .base import dmdResult
from .exact import exactDMD


@dataclass
class dmdcResult(dmdResult):
    stateBasis: matrixOperator | None = None
    controlOperator: matrixOperator | None = None

    def reconstruct(self, realOutput: bool | None = None):
        raise ValueError("DMDc reconstruction requires controls; use simulate instead.")

    def predict(self, steps=None, times=None, realOutput: bool | None = None):
        raise ValueError("DMDc prediction requires controls; use simulate instead.")

    def simulate(self, initialState, controls, includeInitial: bool = True):
        basis = self.stateBasis.toNumpy()
        reducedA = self.reducedOperator.toNumpy()
        reducedB = self.controlOperator.toNumpy()
        initial = np.asarray(initialState)
        controlData = np.asarray(controls)
        if initial.ndim != 1 or initial.shape[0] != basis.shape[0]:
            raise ValueError("initialState has an incompatible shape.")
        if controlData.ndim != 2 or controlData.shape[0] != reducedB.shape[1]:
            raise ValueError("controls must have shape (control_dimension, steps).")
        state = basis.conj().T @ initial
        states = [basis @ state]
        for control in controlData.T:
            state = reducedA @ state + reducedB @ control
            states.append(basis @ state)
        output = np.column_stack(states if includeInitial else states[1:])
        output = self._realIfAppropriate(output, None)
        return self._fromNumpy(output)


@registerAlgorithm('dmd', 'dmdc')
class dmdc(exactDMD):
    def fit(
        self,
        snapshots: operatorBase,
        controlSnapshots: operatorBase,
        shiftedSnapshots: operatorBase | None = None,
        rank: int | None = None,
        stateRank: int | None = None,
        tol: float | None = None,
        dt: float | None = None,
        **kwargs,
    ) -> dmdcResult:
        self._rejectTrainingTimes(kwargs)
        if self.localBackend.name != 'scipy':
            raise NotImplementedError("DMDc currently supports only the scipy backend.")
        xData, yData, trainingSteps, inputWasReal = self._pairedData(
            snapshots, shiftedSnapshots, dt
        )
        if not isinstance(controlSnapshots, operatorBase):
            raise TypeError("controlSnapshots must be an operatorBase instance.")
        if controlSnapshots.backendName != snapshots.backendName:
            raise ValueError("Control and state snapshot backends must match.")
        controls = controlSnapshots.toBackendData()
        if len(getattr(controls, 'shape', ())) != 2 or controls.shape[1] != xData.shape[1]:
            raise ValueError("controlSnapshots must have one column per snapshot pair.")
        if not np.all(np.isfinite(controls)):
            raise ValueError("Control snapshots must contain only finite values.")
        omega = np.vstack([xData, controls])
        Uomega, somega, Vhomega = linalg.svd(omega, full_matrices=False)
        requestedRank = truncate(somega, rank=rank, tol=tol, backend=self.localBackend)
        numericalRank = int(np.count_nonzero(
            somega > self._numericalTolerance(somega, omega.shape)
        ))
        inputRank = min(requestedRank, numericalRank)
        if inputRank == 0:
            raise ValueError("DMDc requires a positive numerical augmented snapshot rank.")
        Uomega = Uomega[:, :inputRank]
        somega = somega[:inputRank]
        Vhomega = Vhomega[:inputRank, :]
        identified = yData @ Vhomega.conj().T / somega[None, :]
        stateDimension = xData.shape[0]
        stateOperator = identified @ Uomega[:stateDimension, :].conj().T
        controlOperator = identified @ Uomega[stateDimension:, :].conj().T
        Uy, sy, _ = linalg.svd(yData, full_matrices=False)
        stateNumericalRank = int(np.count_nonzero(
            sy > self._numericalTolerance(sy, yData.shape)
        ))
        if stateRank is None:
            selectedStateRank = stateNumericalRank
        elif not isinstance(stateRank, Integral) or isinstance(stateRank, bool) or stateRank < 1:
            raise ValueError("stateRank must be a positive integer.")
        else:
            selectedStateRank = min(int(stateRank), stateNumericalRank)
        if selectedStateRank == 0:
            raise ValueError("DMDc requires a positive numerical state rank.")
        basis = Uy[:, :selectedStateRank]
        reducedA = basis.conj().T @ stateOperator @ basis
        reducedB = basis.conj().T @ controlOperator
        eigenvalues, eigenvectors = linalg.eig(reducedA)
        modes = basis @ eigenvectors
        amplitudes = linalg.lstsq(modes, xData[:, 0])[0]
        fittedY = basis @ (reducedA @ (basis.conj().T @ xData) + reducedB @ controls)
        return dmdcResult(
            modes=matrixOperator(modes, 'scipy'),
            eigenvalues=eigenvalues,
            amplitudes=amplitudes,
            reducedOperator=matrixOperator(reducedA, 'scipy'),
            singularValues=sy[:selectedStateRank],
            rank=selectedStateRank,
            dt=dt,
            diagnostics={
                'variant': 'dmdc',
                'inputRank': inputRank,
                'inputNumericalRank': numericalRank,
                'stateRank': selectedStateRank,
                'oneStepResidual': self._relativeNorm(yData - fittedY, yData),
            },
            trainingSteps=trainingSteps,
            inputWasReal=inputWasReal,
            stateBasis=matrixOperator(basis, 'scipy'),
            controlOperator=matrixOperator(reducedB, 'scipy'),
        )


DMDc = dmdc
DMDcResult = dmdcResult
