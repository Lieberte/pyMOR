from dataclasses import dataclass, field
from typing import Any

from mor.models.lti import ltiModel
from mor.operators.matrix import matrixOperator
from mor.operators.operatorsBase import operatorBase
from mor.reductors.pod import podReductor

from .config import motorThermalReductionConfig
from .problem import motorThermalProblem


@dataclass
class motorThermalPODResult:
    model: ltiModel
    basis: matrixOperator
    singularValues: Any
    order: int
    solverInfo: dict[str, Any] = field(default_factory=dict)


class motorThermalPODReductor:
    def __init__(self, globalBackendName: str | None = None, **options):
        self.backendName = globalBackendName
        self.options = options

    def reduce(
        self,
        problem: motorThermalProblem,
        snapshots: operatorBase,
        config: motorThermalReductionConfig | None = None,
    ) -> motorThermalPODResult:
        if not isinstance(problem, motorThermalProblem):
            raise TypeError("problem must be a motorThermalProblem")
        if not isinstance(snapshots, operatorBase):
            raise TypeError("snapshots must be an operatorBase")
        if len(snapshots.shape) != 2 or snapshots.shape[0] != problem.stateDim:
            raise ValueError("snapshots must have one row per thermal state")
        if snapshots.backendName != problem.backendName:
            raise ValueError("snapshots and problem must use the same backend")

        config = config or motorThermalReductionConfig()
        backendName = config.backendName or self.backendName or problem.backendName
        if backendName != problem.backendName:
            raise ValueError("configured backend must match the problem backend")

        options = dict(self.options)
        options.update(config.options)
        pod = podReductor(globalBackendName=backendName, **options)
        podResult = pod.reduce(snapshots, rank=config.rank, tol=config.tol, method=config.method)
        basis = podResult.basis
        leftBasis = basis.T

        reducedCapacity = matrixOperator(
            leftBasis.apply(problem.capacity.apply(basis)), backendName=backendName
        )
        reducedConductivity = matrixOperator(
            leftBasis.apply(problem.conductivity.apply(basis)), backendName=backendName
        )
        reducedInput = matrixOperator(leftBasis.apply(problem.inputMap), backendName=backendName)
        reducedOutput = matrixOperator(problem.outputMap.apply(basis), backendName=backendName)
        reducedModel = ltiModel(
            A=-reducedConductivity,
            B=reducedInput,
            C=reducedOutput,
            D=problem.feedthrough,
            E=reducedCapacity,
            backendName=backendName,
            isContinuous=True,
        )

        return motorThermalPODResult(
            model=reducedModel,
            basis=basis,
            singularValues=podResult.S,
            order=podResult.order,
            solverInfo=podResult.solverInfo,
        )
