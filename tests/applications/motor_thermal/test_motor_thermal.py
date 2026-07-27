import numpy as np
import pytest

from mor.applications.motor_thermal import (
    assembleMotorThermalLTI,
    evaluateMotorThermalError,
    motorThermalPODReductor,
    motorThermalProblem,
    motorThermalReductionConfig,
)
from mor.operators import matrixOperator


def makeProblem(stateDim=4):
    return motorThermalProblem(
        capacity=matrixOperator(np.diag(np.linspace(2.0, 3.0, stateDim)), backendName="scipy"),
        conductivity=matrixOperator(np.diag(np.linspace(1.0, 2.0, stateDim)), backendName="scipy"),
        inputMap=matrixOperator(np.ones((stateDim, 1)), backendName="scipy"),
        outputMap=matrixOperator(np.eye(stateDim)[:2], backendName="scipy"),
        metadata={"application": "motor_thermal"},
    )


def test_assemble_descriptor_lti():
    problem = makeProblem()

    model = assembleMotorThermalLTI(problem)

    assert model.isContinuous
    assert model.order == problem.stateDim
    np.testing.assert_allclose(model.E.toNumpy(), problem.capacity.toNumpy())
    np.testing.assert_allclose(model.A.toNumpy(), -problem.conductivity.toNumpy())


def test_pod_reductor_projects_thermal_operators():
    problem = makeProblem()
    snapshots = matrixOperator(
        np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 1.0, 2.0],
                [2.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
            ]
        ),
        backendName="scipy",
    )

    result = motorThermalPODReductor().reduce(
        problem,
        snapshots,
        motorThermalReductionConfig(rank=2),
    )

    basis = result.basis.toNumpy()
    assert result.order == 2
    assert result.model.order == 2
    np.testing.assert_allclose(
        result.model.E.toNumpy(), basis.T @ problem.capacity.toNumpy() @ basis
    )
    np.testing.assert_allclose(
        result.model.A.toNumpy(), -(basis.T @ problem.conductivity.toNumpy() @ basis)
    )
    np.testing.assert_allclose(result.model.B.toNumpy(), basis.T @ problem.inputMap.toNumpy())
    np.testing.assert_allclose(result.model.C.toNumpy(), problem.outputMap.toNumpy() @ basis)


def test_problem_rejects_mismatched_operator_shape():
    with pytest.raises(ValueError, match="conductivity"):
        motorThermalProblem(
            capacity=matrixOperator(np.eye(3), backendName="scipy"),
            conductivity=matrixOperator(np.eye(2), backendName="scipy"),
            inputMap=matrixOperator(np.ones((3, 1)), backendName="scipy"),
            outputMap=matrixOperator(np.ones((1, 3)), backendName="scipy"),
        )


def test_thermal_error_report_tracks_hotspot_and_field_error():
    reference = np.array([[10.0, 12.0], [8.0, 15.0], [9.0, 11.0]])
    approximation = np.array([[9.0, 11.0], [8.5, 13.0], [9.0, 12.0]])

    report = evaluateMotorThermalError(reference, approximation)

    assert report.relativeL2 > 0.0
    assert report.maxAbsolute == pytest.approx(2.0)
    assert report.hotspotAbsolute == pytest.approx(2.0)
