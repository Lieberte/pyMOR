from numbers import Integral

import numpy as np
from scipy import optimize

from mor.algorithm.registry import registerAlgorithm
from mor.operators.matrix import matrixOperator
from mor.operators.operatorsBase import operatorBase

from .base import dmdResult
from .exact import exactDMD


@registerAlgorithm('dmd', 'optimized')
class optimizedDMD(exactDMD):
    def fit(
        self,
        snapshots: operatorBase,
        shiftedSnapshots: operatorBase | None = None,
        rank: int | None = None,
        tol: float | None = None,
        dt: float | None = None,
        maxNfev: int = 200,
        **kwargs,
    ) -> dmdResult:
        self._rejectTrainingTimes(kwargs)
        if self.localBackend.name != 'scipy':
            raise NotImplementedError("Optimized DMD currently supports only the scipy backend.")
        if shiftedSnapshots is not None:
            raise ValueError("Optimized DMD requires a single ordered snapshot sequence.")
        if not isinstance(maxNfev, Integral) or isinstance(maxNfev, bool) or maxNfev < 1:
            raise ValueError("maxNfev must be a positive integer.")
        initial = super().fit(
            snapshots,
            rank=rank,
            tol=tol,
            dt=dt,
            amplitudeMethod='global',
        )
        data = snapshots.toBackendData()
        timeStep = 1.0 if dt is None else dt
        times = np.arange(data.shape[1], dtype=float) * timeStep
        initialRates = np.log(initial.eigenvalues.astype(complex)) / timeStep
        if not np.all(np.isfinite(initialRates)):
            raise ValueError("Optimized DMD requires nonzero finite initial eigenvalues.")
        if not np.all(np.isfinite(times)):
            raise ValueError('Optimized DMD training times must remain finite')
        maxGrowth = (np.log(np.finfo(float).max) - 4.) / times[-1]
        if np.any(initialRates.real > maxGrowth):
            raise ValueError('Optimized DMD initial rates overflow the training horizon; rescale the snapshots')

        def unpack(parameters):
            half = parameters.size // 2
            return parameters[:half] + 1j * parameters[half:]

        def evaluate(rates):
            try:
                with np.errstate(over='raise', invalid='raise'):
                    dynamics = np.exp(rates[:, None] * times[None, :])
            except FloatingPointError as error:
                raise ValueError('Optimized DMD rates overflow the training horizon') from error
            modes = data @ np.linalg.pinv(dynamics)
            return modes, modes @ dynamics - data

        def residual(parameters):
            _, error = evaluate(unpack(parameters))
            return np.concatenate([error.real.reshape(-1), error.imag.reshape(-1)])

        parameters = np.concatenate([initialRates.real, initialRates.imag])
        upper = np.concatenate([np.full(initial.rank, maxGrowth), np.full(initial.rank, np.inf)])
        solution = optimize.least_squares(residual, parameters, bounds=(-np.inf, upper), max_nfev=int(maxNfev))
        rates = unpack(solution.x)
        modes, error = evaluate(rates)
        eigenvalues = np.exp(rates * timeStep)
        amplitudes = np.ones(initial.rank, dtype=complex)
        reducedOperator = np.diag(eigenvalues)
        result = dmdResult(
            modes=matrixOperator(modes, 'scipy'),
            eigenvalues=eigenvalues,
            amplitudes=amplitudes,
            reducedOperator=matrixOperator(reducedOperator, 'scipy'),
            singularValues=initial.singularValues,
            rank=initial.rank,
            dt=dt,
            diagnostics={
                **initial.diagnostics,
                'variant': 'optimized',
                'optimizerSuccess': bool(solution.success),
                'optimizerStatus': int(solution.status),
                'optimizerEvaluations': int(solution.nfev),
                'optimizerCost': float(solution.cost),
                'initialReconstructionError': initial.diagnostics['reconstructionError'],
                'reconstructionError': self._relativeNorm(error, data),
            },
            trainingSteps=data.shape[1],
            inputWasReal=not np.iscomplexobj(data),
        )
        return result


OptimizedDMD = optimizedDMD
