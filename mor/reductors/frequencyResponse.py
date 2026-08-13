from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy import linalg

from mor.models.frequency import frequencyPoints, prepareTransferData, solveShiftedSystem, validateFrequencies
from mor.models.lti import ltiModel
from mor.operators import matrixOperator
from .balancedTruncation import reducedSystem


class frequencyResponseReductor:
    def __init__(self, globalBackendName: str | None = None, frequencies=None):
        self.backendName = globalBackendName
        self.frequencies = frequencies

    def reduce(self, model: ltiModel, *, order: int) -> reducedSystem:
        if (self.backendName or model.backendName) != 'scipy':
            raise NotImplementedError("Frequency-response reduction currently supports only scipy.")
        frequencies = validateFrequencies(self.frequencies, allowEmpty=False)
        if not isinstance(order, Integral) or isinstance(order, bool) or order <= 0:
            raise ValueError("order must be a positive integer.")
        rawData = {
            'A': model.A.toBackendData(),
            'B': model.B.toBackendData(),
            'C': model.C.toBackendData(),
            'D': model.D.toBackendData() if model.D is not None else None,
            'E': model.E.toBackendData() if model.E is not None else None,
        }
        A, B, C, E, D = prepareTransferData(
            rawData['A'], rawData['B'], rawData['C'], rawData['E'], rawData['D'], forceComplex=False
        )
        isComplexDynamics = any(
            np.issubdtype(value.dtype, np.complexfloating)
            for name, value in rawData.items()
            if name != 'D'
            if value is not None
        )
        rightSnapshots = []
        leftSnapshots = []
        Ah = A.conj().T
        Eh = E.conj().T
        Ch = C.conj().T
        for point in frequencyPoints(frequencies, isContinuous=model.isContinuous):
            right = solveShiftedSystem(A, E, B, point)
            left = solveShiftedSystem(Ah, Eh, Ch, point.conjugate())
            if isComplexDynamics:
                rightSnapshots.append(right)
                leftSnapshots.append(left)
            else:
                rightSnapshots.extend((right.real, right.imag))
                leftSnapshots.extend((left.real, left.imag))
        rightSnapshotMatrix = np.hstack(rightSnapshots)
        leftSnapshotMatrix = np.hstack(leftSnapshots)
        rightBasis, rightValues, _ = linalg.svd(rightSnapshotMatrix, full_matrices=False)
        leftBasis, leftValues, _ = linalg.svd(leftSnapshotMatrix, full_matrices=False)
        rightRank = self._numericalRank(rightValues, rightSnapshotMatrix.shape)
        leftRank = self._numericalRank(leftValues, leftSnapshotMatrix.shape)
        availableOrder = min(rightRank, leftRank)
        if order > availableOrder:
            raise ValueError(f"Requested order {order} exceeds numerical snapshot rank {availableOrder}.")
        V = rightBasis[:, :order]
        W = leftBasis[:, :order]
        Wh = W.conj().T
        Er = Wh @ (E @ V)
        descriptorKind = model.descriptorKind
        projectedDescriptorRank = int(np.linalg.matrix_rank(Er))
        if descriptorKind == 'regular' and projectedDescriptorRank < order:
            raise ValueError("Projected descriptor is singular; left and right snapshot spaces are incompatible.")
        return reducedSystem(
            isContinuous=model.isContinuous,
            Ar=matrixOperator(Wh @ (A @ V), backendName='scipy'),
            Br=matrixOperator(Wh @ B, backendName='scipy'),
            Cr=matrixOperator(C @ V, backendName='scipy'),
            Dr=matrixOperator(D, backendName='scipy'),
            Er=matrixOperator(Er, backendName='scipy'),
            hsv=[],
            order=order,
            solverInfo={
                'reductor': 'frequencyResponse',
                'trainingFrequencies': int(frequencies.size),
                'rightSnapshotTail': float(np.linalg.norm(rightValues[order:])),
                'leftSnapshotTail': float(np.linalg.norm(leftValues[order:])),
                'rightSnapshotRank': rightRank,
                'leftSnapshotRank': leftRank,
                'projectedDescriptorRank': projectedDescriptorRank,
                'projectedDescriptorCondition': float(np.linalg.cond(Er)),
                'descriptorKind': descriptorKind,
            },
        )

    @staticmethod
    def _numericalRank(singularValues: np.ndarray, shape: tuple[int, int]) -> int:
        if singularValues.size == 0 or singularValues[0] == 0:
            return 0
        tolerance = max(shape) * np.finfo(singularValues.dtype).eps * singularValues[0]
        return int(np.count_nonzero(singularValues > tolerance))
