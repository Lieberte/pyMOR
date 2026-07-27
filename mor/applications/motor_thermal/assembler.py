from mor.models.lti import ltiModel

from .problem import motorThermalProblem


def assembleMotorThermalLTI(problem: motorThermalProblem) -> ltiModel:
    return ltiModel(
        A=-problem.conductivity,
        B=problem.inputMap,
        C=problem.outputMap,
        D=problem.feedthrough,
        E=problem.capacity,
        backendName=problem.backendName,
        isContinuous=True,
    )
