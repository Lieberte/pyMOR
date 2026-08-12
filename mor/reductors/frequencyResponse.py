from __future__ import annotations

import numpy as np
from scipy import linalg

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
        frequencies = np.asarray(self.frequencies, dtype=float).reshape(-1)
        if frequencies.size == 0 or np.any(frequencies < 0):
            raise ValueError("frequencies must be non-empty and non-negative.")
        if order <= 0:
            raise ValueError("order must be positive.")
        data = model.toNumpy()
        dtype = np.result_type(data['A'].dtype, np.float64)
        A = np.asarray(data['A'], dtype=dtype)
        E = np.eye(model.order, dtype=dtype) if data['E'] is None else np.asarray(data['E'], dtype=dtype)
        B = np.asarray(data['B'], dtype=dtype)
        C = np.asarray(data['C'], dtype=dtype)
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        rightSnapshots = []
        leftSnapshots = []
        for frequency in frequencies:
            right = linalg.solve(1j * frequency * E - A, B)
            left = linalg.solve((-1j * frequency * E - A).T, C.T)
            rightSnapshots.extend((right.real, right.imag))
            leftSnapshots.extend((left.real, left.imag))
        rightBasis, rightValues, _ = linalg.svd(np.hstack(rightSnapshots), full_matrices=False)
        leftBasis, leftValues, _ = linalg.svd(np.hstack(leftSnapshots), full_matrices=False)
        availableOrder = min(rightBasis.shape[1], leftBasis.shape[1])
        if order > availableOrder:
            raise ValueError(f"Requested order {order} exceeds snapshot rank bound {availableOrder}.")
        V = rightBasis[:, :order]
        W = leftBasis[:, :order]
        Dr = data['D'] if data['D'] is not None else np.zeros((model.outputDim, model.inputDim), dtype=dtype)
        return reducedSystem(
            isContinuous=model.isContinuous,
            Ar=matrixOperator(W.T @ A @ V, backendName='scipy'),
            Br=matrixOperator(W.T @ B, backendName='scipy'),
            Cr=matrixOperator(C @ V, backendName='scipy'),
            Dr=matrixOperator(Dr, backendName='scipy'),
            Er=matrixOperator(W.T @ E @ V, backendName='scipy'),
            hsv=[],
            order=order,
            solverInfo={
                'reductor': 'frequencyResponse',
                'trainingFrequencies': int(frequencies.size),
                'rightSnapshotTail': float(np.linalg.norm(rightValues[order:])),
                'leftSnapshotTail': float(np.linalg.norm(leftValues[order:])),
            },
        )
