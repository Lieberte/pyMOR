from mor.algorithm.registry import algorithmRegistry
from mor.operators.operatorsBase import operatorBase

from .registry import registerReductor


@registerReductor('dmd')
class dmdReductor:
    def __init__(self, globalBackendName: str | None = None, **kwargs):
        self.backendName = globalBackendName
        self.options = kwargs

    def reduce(
        self,
        snapshots: operatorBase,
        shiftedSnapshots: operatorBase | None = None,
        rank: int | None = None,
        tol: float | None = None,
        method: str = 'auto',
        dt: float | None = None,
        **kwargs,
    ):
        backendName = self.backendName or snapshots.backendName
        algorithm = algorithmRegistry.get(
            category='dmd',
            variant=method,
            forceOptions=self.options,
            backendName=backendName,
            xOperator=snapshots,
            rank=rank,
            **kwargs,
        )
        result = algorithm.fit(
            snapshots,
            shiftedSnapshots=shiftedSnapshots,
            rank=rank,
            tol=tol,
            dt=dt,
            **kwargs,
        )
        result.diagnostics['requestedMethod'] = method
        return result


DMDReductor = dmdReductor
