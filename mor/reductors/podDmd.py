from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from mor.algorithm.dmd import dmdResult
from mor.backends import backendRegistry
from mor.operators.matrix import matrixOperator
from mor.operators.operatorsBase import operatorBase

from .dmd import dmdReductor
from .pod import podReductor, podResult
from .registry import registerReductor


@dataclass
class podDmdResult(dmdResult):
    pod: podResult | None = None
    reducedDmd: dmdResult | None = None
    snapshotMean: Any | None = None
    product: operatorBase | None = None

    @property
    def basis(self) -> matrixOperator:
        return self.pod.V

    @property
    def podRank(self) -> int:
        return self.pod.order

    @property
    def dmdRank(self) -> int:
        return self.rank

    @property
    def reducedModes(self) -> matrixOperator:
        return self.reducedDmd.modes

    def project(self, states: operatorBase | Any) -> Any:
        backend = self.modes.localBackend
        if isinstance(states, operatorBase):
            if states.backendName != self.modes.backendName:
                raise ValueError("State and POD-DMD backends must match.")
            data = states.toBackendData()
        else:
            data = backend.array.toArray(states)
        if len(getattr(data, 'shape', ())) != 2 or data.shape[0] != self.basis.shape[0]:
            raise ValueError("states must have shape (state_dimension, snapshots).")
        if self.snapshotMean is not None:
            data = data - backend.array.reshape(self.snapshotMean, (-1, 1))
        weighted = self.product.apply(data) if self.product is not None else data
        basisAdjoint = backend.linalg.transpose(
            backend.linalg.conj(self.basis.toBackendData())
        )
        basisAdjoint = self._promoteForProduct(basisAdjoint, weighted)
        return backend.linalg.dot(basisAdjoint, weighted)

    def lift(self, coordinates: Any, includeMean: bool = True) -> Any:
        backend = self.modes.localBackend
        data = backend.array.toArray(coordinates)
        if len(getattr(data, 'shape', ())) != 2 or data.shape[0] != self.podRank:
            raise ValueError("coordinates must have shape (pod_rank, snapshots).")
        basis = self._promoteForProduct(self.basis.toBackendData(), data)
        states = backend.linalg.dot(basis, data)
        if includeMean and self.snapshotMean is not None:
            states = states + backend.array.reshape(self.snapshotMean, (-1, 1))
        return states

    def predict(
        self,
        steps: int | Any | None = None,
        times: Any | None = None,
        realOutput: bool | None = None,
    ) -> Any:
        prediction = super().predict(steps=steps, times=times, realOutput=realOutput)
        if self.snapshotMean is None:
            return prediction
        backend = self.modes.localBackend
        return prediction + backend.array.reshape(self.snapshotMean, (-1, 1))

    def _promoteForProduct(self, left: Any, right: Any) -> Any:
        if self.modes.backendName != 'torch' or left.dtype == right.dtype:
            return left
        import torch

        return left.to(torch.promote_types(left.dtype, right.dtype))


@registerReductor('podDmd')
class podDmdReductor:
    def __init__(
        self,
        globalBackendName: str | None = None,
        product: operatorBase | None = None,
        centerSnapshots: bool = False,
        podOptions: dict[str, Any] | None = None,
        dmdOptions: dict[str, Any] | None = None,
    ):
        self.backendName = globalBackendName
        self.product = product
        self.centerSnapshots = centerSnapshots
        self.podOptions = dict(podOptions or {})
        self.dmdOptions = dict(dmdOptions or {})

    def reduce(
        self,
        snapshots: operatorBase,
        shiftedSnapshots: operatorBase | None = None,
        podRank: int | None = None,
        podTol: float | None = None,
        dmdRank: int | None = None,
        dmdTol: float | None = None,
        podMethod: str = 'auto',
        dmdMethod: str = 'auto',
        dt: float | None = None,
        podBasis: podResult | None = None,
        **kwargs,
    ) -> podDmdResult:
        if not isinstance(snapshots, operatorBase):
            raise TypeError("snapshots must be an operatorBase instance.")
        if kwargs.get('controlSnapshots') is not None:
            raise NotImplementedError("POD-DMDc requires a dedicated controlled result contract.")
        self._validateRank('podRank', podRank)
        self._validateRank('dmdRank', dmdRank)
        backendName = self.backendName or snapshots.backendName
        if snapshots.backendName != backendName:
            raise ValueError("Snapshot and POD-DMD backends must match.")
        backend = backendRegistry.get(backendName)
        xData, yData, trainingSteps = self._pairedData(snapshots, shiftedSnapshots)
        pod = podBasis or self._fitPod(
            snapshots,
            shiftedSnapshots,
            podRank,
            podTol,
            podMethod,
            backend,
        )
        self._validatePodBasis(pod, snapshots)
        if shiftedSnapshots is None:
            reducedSnapshots = self._project(
                pod, snapshots.toBackendData(), backend
            )
            reducedX = reducedSnapshots[:, :-1]
            reducedY = reducedSnapshots[:, 1:]
            dmdSnapshots = matrixOperator(reducedSnapshots, backendName=backendName)
            dmdShiftedSnapshots = None
        else:
            reducedX = self._project(pod, xData, backend)
            reducedY = self._project(pod, yData, backend)
            dmdSnapshots = matrixOperator(reducedX, backendName=backendName)
            dmdShiftedSnapshots = matrixOperator(reducedY, backendName=backendName)
        dmdOptions = {**self.dmdOptions, **kwargs}
        reservedOptions = {'rank', 'tol', 'method', 'dt', 'shiftedSnapshots'}
        conflictingOptions = sorted(reservedOptions.intersection(dmdOptions))
        if conflictingOptions:
            raise ValueError(
                "Use POD-DMD arguments instead of reserved dmdOptions: "
                + ', '.join(conflictingOptions)
            )
        dmd = dmdReductor(globalBackendName=backendName).reduce(
            dmdSnapshots,
            shiftedSnapshots=dmdShiftedSnapshots,
            rank=dmdRank,
            tol=dmdTol,
            method=dmdMethod,
            dt=dt,
            **dmdOptions,
        )
        basis = pod.V.toBackendData()
        reducedModes = dmd.modes.toBackendData()
        if backendName == 'torch' and basis.dtype != reducedModes.dtype:
            import torch

            basis = basis.to(torch.promote_types(basis.dtype, reducedModes.dtype))
        liftedModes = backend.linalg.dot(basis, reducedModes)
        result = podDmdResult(
            modes=matrixOperator(liftedModes, backendName=backendName),
            eigenvalues=dmd.eigenvalues,
            amplitudes=dmd.amplitudes,
            reducedOperator=dmd.reducedOperator,
            singularValues=dmd.singularValues,
            rank=dmd.rank,
            dt=dmd.dt,
            diagnostics={
                'variant': 'podDmd',
                'dmdVariant': dmd.diagnostics.get('variant', dmdMethod),
                'podRank': pod.order,
                'dmdRank': dmd.rank,
                'podProjectionError': pod.reconstructionError,
                'reducedDmdSnapshotResidual': dmd.diagnostics.get('snapshotResidual'),
                'reducedDmdOneStepResidual': dmd.diagnostics.get('oneStepResidual'),
                'reducedDmdReconstructionError': dmd.diagnostics.get('reconstructionError'),
                'podSolverInfo': pod.solverInfo,
                'dmdDiagnostics': dmd.diagnostics,
            },
            trainingSteps=trainingSteps,
            inputWasReal=not backend.array.iscomplexobj(snapshots.toBackendData()),
            pod=pod,
            reducedDmd=dmd,
            snapshotMean=pod.snapshotMean,
            product=pod.product,
        )
        result.diagnostics['fullOneStepResidual'] = self._fullOneStepResidual(
            result, reducedX, yData, backend
        )
        if trainingSteps is not None:
            original = snapshots.toNumpy()
            reconstructed = backend.array.toNumpy(result.reconstruct(realOutput=False))
            reducedOriginal = backend.array.toNumpy(reducedSnapshots)
            reducedReconstructed = backend.array.toNumpy(
                dmd.reconstruct(realOutput=False)
            )
            result.diagnostics['reducedDmdReconstructionError'] = self._relativeNorm(
                reducedOriginal - reducedReconstructed, reducedOriginal
            )
            result.diagnostics['fullReconstructionError'] = self._relativeNorm(
                original - reconstructed, original
            )
        return result

    def _fitPod(
        self,
        snapshots: operatorBase,
        shiftedSnapshots: operatorBase | None,
        rank: int | None,
        tol: float | None,
        method: str,
        backend: Any,
    ) -> podResult:
        options = dict(self.podOptions)
        options['centerSnapshots'] = self.centerSnapshots
        if self.product is not None:
            options['product'] = self.product
        trainingSnapshots = snapshots
        if shiftedSnapshots is not None:
            combined = backend.array.hstack([
                snapshots.toBackendData(), shiftedSnapshots.toBackendData()
            ])
            trainingSnapshots = matrixOperator(combined, backendName=snapshots.backendName)
        return podReductor(
            globalBackendName=snapshots.backendName, **options
        ).reduce(trainingSnapshots, rank=rank, tol=tol, method=method)

    @staticmethod
    def _pairedData(
        snapshots: operatorBase,
        shiftedSnapshots: operatorBase | None,
    ) -> tuple[Any, Any, int | None]:
        data = snapshots.toBackendData()
        if len(getattr(data, 'shape', ())) != 2:
            raise ValueError("snapshots must be a two-dimensional matrix.")
        if shiftedSnapshots is None:
            if data.shape[1] < 2:
                raise ValueError("At least two snapshots are required.")
            return data[:, :-1], data[:, 1:], data.shape[1]
        if not isinstance(shiftedSnapshots, operatorBase):
            raise TypeError("shiftedSnapshots must be an operatorBase instance.")
        if shiftedSnapshots.backendName != snapshots.backendName:
            raise ValueError("Snapshot pair backends must match.")
        shiftedData = shiftedSnapshots.toBackendData()
        if len(getattr(shiftedData, 'shape', ())) != 2 or shiftedData.shape != data.shape:
            raise ValueError("Explicit X and Y snapshots must have identical two-dimensional shapes.")
        return data, shiftedData, None

    @staticmethod
    def _validateRank(name: str, rank: int | None) -> None:
        if rank is not None and (
            not isinstance(rank, Integral) or isinstance(rank, bool) or rank < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")

    @staticmethod
    def _validatePodBasis(pod: podResult, snapshots: operatorBase) -> None:
        if not isinstance(pod, podResult):
            raise TypeError("podBasis must be a podResult instance.")
        if pod.order < 1:
            raise ValueError("POD basis must have positive rank.")
        if pod.V.backendName != snapshots.backendName:
            raise ValueError("POD basis and snapshot backends must match.")
        if pod.V.shape[0] != snapshots.shape[0]:
            raise ValueError("POD basis and snapshots have incompatible state dimensions.")
        if pod.product is not None and (
            pod.product.backendName != snapshots.backendName
            or pod.product.shape != (snapshots.shape[0], snapshots.shape[0])
        ):
            raise ValueError("POD product is incompatible with the snapshots.")

    @staticmethod
    def _project(pod: podResult, data: Any, backend: Any) -> Any:
        if pod.snapshotMean is not None:
            data = data - backend.array.reshape(pod.snapshotMean, (-1, 1))
        weighted = pod.product.apply(data) if pod.product is not None else data
        basis = pod.V.toBackendData()
        basisAdjoint = backend.linalg.transpose(backend.linalg.conj(basis))
        if backend.name == 'torch' and basisAdjoint.dtype != weighted.dtype:
            import torch

            basisAdjoint = basisAdjoint.to(
                torch.promote_types(basisAdjoint.dtype, weighted.dtype)
            )
        return backend.linalg.dot(basisAdjoint, weighted)

    @staticmethod
    def _relativeNorm(residual: np.ndarray, reference: np.ndarray) -> float:
        denominator = np.linalg.norm(reference)
        return 0.0 if denominator == 0 else float(np.linalg.norm(residual) / denominator)

    def _fullOneStepResidual(
        self,
        result: podDmdResult,
        reducedX: Any,
        yData: Any,
        backend: Any,
    ) -> float:
        modes = result.reducedModes.toNumpy()
        eigenvalues = backend.array.toNumpy(result.eigenvalues)
        reducedX = backend.array.toNumpy(reducedX)
        predictedCoordinates = modes @ np.diag(eigenvalues) @ np.linalg.pinv(modes) @ reducedX
        basis = result.basis.toNumpy()
        predicted = basis @ predictedCoordinates
        if result.snapshotMean is not None:
            predicted = predicted + backend.array.toNumpy(result.snapshotMean)[:, None]
        return self._relativeNorm(
            backend.array.toNumpy(yData) - predicted,
            backend.array.toNumpy(yData),
        )


PODDMDReductor = podDmdReductor
PODDMDResult = podDmdResult
