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
    solverInfo: dict = dataclasses.field(default_factory=dict)

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
        return podResult(V=matrixOperator(U, backendName=backend.name),S=S,order=order,solverInfo={'podSolver': self.solver.__class__.__name__, 'svd': getattr(self.solver, '_lastAlgorithmName', 'unknown')})

    def reconstruct(self, podBasis: matrixOperator, coefficients: Any) -> matrixOperator:
        backend = self.localBackend or podBasis.localBackend
        reconstructed = podBasis.apply(coefficients)
        return matrixOperator(reconstructed, backendName=backend.name)
