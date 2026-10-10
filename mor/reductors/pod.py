import dataclasses
from typing import Any
from dataclasses import dataclass
from mor.solvers.registry import solverRegistry
from mor.backends import backendRegistry
from mor.operators.operatorsBase import operatorBase
from mor.operators.matrix import matrixOperator

@dataclass
class podResult:
    V: matrixOperator
    S: Any
    order: int
    reconstructionError: float | None = None
    snapshotMean: Any | None = None
    product: operatorBase | None = None
    solverInfo: dict = dataclasses.field(default_factory=dict)

    @property
    def basis(self) -> matrixOperator:
        return self.V

class podReductor:
    def __init__(self, globalBackendName: str | None = None, **kwargs):
        self.backendName = globalBackendName
        self.options = kwargs
        self.localBackend = None
        self.solver = None

    def updateOptions(self, snapshots: operatorBase):
        backendName = self.backendName or snapshots.backendName
        self.localBackend = backendRegistry.get(backendName)
        solverName = self.options.get('podSolver', 'auto')
        self.solver = solverRegistry.get(solverType='pod',name=solverName,forceOptions=self.options, backendName=backendName)

    def reduce(self, snapshots: operatorBase, rank: int | None = None, tol: float | None = None, method: str = 'auto') -> podResult:
        self.updateOptions(snapshots)
        backend = self.localBackend
        U, S, Vt = self.solver.solve(snapshots, rank=rank, tol=tol, method=method)
        order = backend.array.size(S)
        reconstructionError = self._reconstructionError(
            snapshots,
            U,
            getattr(self.solver, 'snapshotMean', None),
            getattr(self.solver, 'product', None),
        )
        return podResult(
            V=matrixOperator(U, backendName=backend.name),
            S=S,
            order=order,
            reconstructionError=reconstructionError,
            snapshotMean=getattr(self.solver, 'snapshotMean', None),
            product=getattr(self.solver, 'product', None),
            solverInfo={'podSolver': self.solver.__class__.__name__, 'svd': getattr(self.solver, '_lastAlgorithmName', 'unknown')},
        )

    def reconstruct(self, podBasis: matrixOperator | podResult, coefficients: Any) -> matrixOperator:
        result = podBasis if isinstance(podBasis, podResult) else None
        basis = result.V if result is not None else podBasis
        backend = self.localBackend or basis.localBackend
        reconstructed = basis.apply(coefficients)
        if result is not None and result.snapshotMean is not None:
            reconstructed = reconstructed + backend.array.reshape(result.snapshotMean, (-1, 1))
        return matrixOperator(reconstructed, backendName=backend.name)

    def _reconstructionError(self, snapshots: operatorBase, basis: Any, snapshotMean: Any, product: operatorBase | None) -> float:
        backend = self.localBackend
        data = snapshots.toBackendData()
        if backend.name == 'scipy' and (backend.array.isSparse(data) or data.nbytes > 4 * 1024 ** 2):
            return self._blockedReconstructionError(data, basis, snapshotMean, product)
        if snapshotMean is not None:
            data = data - backend.array.reshape(snapshotMean, (-1, 1))
        if product is not None and backend.array.size(data):
            scale = backend.array.max(backend.array.abs(data))
            if float(scale) == 0:
                return 0.0
            data = data / scale
        weightedData = product.apply(data) if product is not None else data
        coefficients = backend.linalg.dot(backend.linalg.transpose(backend.linalg.conj(basis)), weightedData)
        residual = data - backend.linalg.dot(basis, coefficients)
        if product is None:
            denominator = backend.linalg.norm(data)
            numerator = backend.linalg.norm(residual)
        else:
            weightedResidual = product.apply(residual)
            denominatorSquared = backend.array.real(backend.array.sum(backend.linalg.conj(data) * weightedData))
            numeratorSquared = backend.array.real(backend.array.sum(backend.linalg.conj(residual) * weightedResidual))
            denominator = backend.array.sqrt(backend.array.max(backend.array.array([0.0, denominatorSquared])))
            numerator = backend.array.sqrt(backend.array.max(backend.array.array([0.0, numeratorSquared])))
        denominator = float(denominator)
        return 0.0 if denominator == 0 else float(numerator) / denominator

    def _blockedReconstructionError(self, data: Any, basis: Any, snapshotMean: Any,
                                    product: operatorBase | None) -> float:
        import numpy as np
        from math import hypot

        sparse = self.localBackend.array.isSparse(data)
        if sparse:
            data = data.tocsc()
        itemsize = np.result_type(data.dtype, basis.dtype).itemsize
        blockSize = max(1, (1024 ** 2) // max(data.shape[0] * itemsize, 1))
        basisAdjoint = basis.conj().T
        weightedScale = 1.0
        if product is not None:
            values = data.data if sparse else data
            weightedScale = float(np.max(np.abs(values))) if values.size else 0.0
            if snapshotMean is not None:
                weightedScale = max(weightedScale, float(np.max(np.abs(snapshotMean))))
            if weightedScale == 0:
                return 0.0
        numeratorSquared = denominatorSquared = 0.0
        numeratorNorm = denominatorNorm = 0.0
        if (not sparse and product is None and snapshotMean is None
                and basis.shape[1] * data.shape[1] * itemsize <= 1024 ** 2):
            coefficientsAll = basisAdjoint @ data
            rowBlockSize = max(1, (1024 ** 2) // max(data.shape[1] * itemsize, 1))
            for start in range(0, data.shape[0], rowBlockSize):
                block = data[start:start + rowBlockSize]
                residual = block - basis[start:start + rowBlockSize] @ coefficientsAll
                denominatorNorm = hypot(denominatorNorm, self.localBackend.linalg.norm(block))
                numeratorNorm = hypot(numeratorNorm, self.localBackend.linalg.norm(residual))
            return 0.0 if denominatorNorm == 0 else numeratorNorm / denominatorNorm
        for start in range(0, data.shape[1], blockSize):
            block = data[:, start:start + blockSize]
            if sparse:
                block = block.toarray()
            if product is not None:
                block = block / weightedScale
            if snapshotMean is not None:
                block = block - np.asarray(snapshotMean).reshape((-1, 1)) / weightedScale
            weightedBlock = product.apply(block) if product is not None else block
            coefficients = basisAdjoint @ weightedBlock
            residual = block - basis @ coefficients
            if product is None:
                denominatorNorm = hypot(denominatorNorm, self.localBackend.linalg.norm(block))
                numeratorNorm = hypot(numeratorNorm, self.localBackend.linalg.norm(residual))
                continue
            weightedResidual = product.apply(residual) if product is not None else residual
            denominatorSquared += float(np.real(np.vdot(block, weightedBlock)))
            numeratorSquared += float(np.real(np.vdot(residual, weightedResidual)))
        if product is None:
            return 0.0 if denominatorNorm == 0 else numeratorNorm / denominatorNorm
        return (0.0 if denominatorSquared == 0 else
                float(np.sqrt(max(numeratorSquared, 0.0) / denominatorSquared)))
