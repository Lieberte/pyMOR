import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mor.models import (
    tabulatedThermalProperty,
    thermalBoundaryField,
    thermalConvectionBoundary,
    thermalLoadCase,
    thermalLinearSolverOptions,
    thermalMaterial,
    thermalMaterialLibrary,
    thermalModel,
)
from mor.operators import matrixOperator, thermalOperatorSet


def twoNodeModel():
    operators = thermalOperatorSet(
        conductivity=matrixOperator(
            csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]])),
            backendName='scipy',
        ),
        capacity=matrixOperator(csr_matrix(np.eye(2)), backendName='scipy'),
        load=matrixOperator(np.array([[0.0], [3.0]]), backendName='scipy'),
        elementVolumes=np.empty(0),
    )
    return thermalModel(
        operators,
        dirichletNodes=[0],
        dirichletValues=10.0,
        metadata={'operating': {'speedRpm': 12000}},
    )


def test_material_library_and_load_case_are_json_ready():
    library = thermalMaterialLibrary(
        {
            'copper': thermalMaterial('copper', 400.0, 8900.0, 385.0),
            'lamination': thermalMaterial(
                'lamination',
                np.diag([25.0, 25.0, 2.0]),
                7650.0,
                460.0,
            ),
        }
    )
    loadCase = thermalLoadCase(
        heatSources={'winding': 1.2e6},
        convection={'jacket': thermalConvectionBoundary(2500.0, 333.15)},
        dirichletTemperatures={'shaftEnd': 320.0},
        operatingMetadata={'speedRpm': 12000, 'torqueNm': 180.0},
    )

    restoredLibrary = thermalMaterialLibrary.fromDict(library.toDict())
    restoredLoadCase = thermalLoadCase.fromDict(loadCase.toDict())

    assert restoredLibrary.names == ('copper', 'lamination')
    np.testing.assert_allclose(
        restoredLibrary['lamination'].conductivity,
        np.diag([25.0, 25.0, 2.0]),
    )
    assert restoredLoadCase.toDict() == loadCase.toDict()


def test_nodal_boundary_fields_are_json_ready():
    loadCase = thermalLoadCase(
        convection={
            'jacket': thermalConvectionBoundary(
                coefficient=thermalBoundaryField([100.0, 200.0, 300.0], location='node'),
                ambientTemperature=thermalBoundaryField([320.0, 325.0, 330.0], location='node'),
            )
        },
        heatFlux={
            'end': thermalBoundaryField([10.0, 20.0, 30.0], location='globalNode')
        },
    )

    restored = thermalLoadCase.fromDict(loadCase.toDict())

    assert isinstance(restored.convection['jacket'].coefficient, thermalBoundaryField)
    assert restored.convection['jacket'].coefficient.location == 'node'
    assert isinstance(restored.heatFlux['end'], thermalBoundaryField)
    assert restored.toDict() == loadCase.toDict()


def test_material_validation_rejects_nonphysical_values():
    with pytest.raises(ValueError, match='conductivity'):
        thermalMaterial('bad', 0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match='density'):
        thermalMaterial('bad', 1.0, -1.0, 1.0)
    with pytest.raises(ValueError, match='positive definite'):
        thermalMaterial('bad', np.diag([1.0, 1.0, -1.0]), 1.0, 1.0)


def test_tabulated_thermal_property_interpolates_and_clamps():
    propertyTable = tabulatedThermalProperty(
        temperatures=[300.0, 400.0, 500.0],
        values=[10.0, 20.0, 40.0],
    )

    np.testing.assert_allclose(
        propertyTable.evaluate([250.0, 350.0, 450.0, 550.0]),
        [10.0, 15.0, 30.0, 40.0],
    )
    restored = tabulatedThermalProperty.fromDict(propertyTable.toDict())
    np.testing.assert_allclose(restored.values, propertyTable.values)


def test_temperature_dependent_material_evaluates_scalar_and_tensor_tables():
    material = thermalMaterial(
        'lamination',
        conductivity={
            'temperature': [300.0, 400.0],
            'value': [
                [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 2.0]],
                [[18.0, 0.0, 0.0], [0.0, 18.0, 0.0], [0.0, 0.0, 1.8]],
            ],
        },
        density=7650.0,
        specificHeat={
            'temperatures': [300.0, 400.0],
            'values': [460.0, 500.0],
        },
    )

    assert material.isTemperatureDependent
    np.testing.assert_allclose(
        material.conductivityAt([350.0])[0],
        np.diag([19.0, 19.0, 1.9]),
    )
    np.testing.assert_allclose(material.specificHeatAt([325.0, 375.0]), [470.0, 490.0])
    restored = thermalMaterial.fromDict('lamination', material.toDict())
    np.testing.assert_allclose(restored.conductivityAt(350.0), np.diag([19.0, 19.0, 1.9]))


def test_thermal_model_solves_steady_and_implicit_euler_systems():
    model = twoNodeModel()

    steady = model.solveSteady()
    transient = model.implicitEulerStep(np.array([10.0, 0.0]), timeStep=1.0)

    assert steady.shape == (2, 1)
    assert transient.shape == (2, 1)
    np.testing.assert_allclose(steady, [[10.0], [6.5]])
    np.testing.assert_allclose(transient, [[10.0], [13.0 / 3.0]])
    assert model.toNumpy()['metadata']['operating']['speedRpm'] == 12000


def test_thermal_model_runs_complete_implicit_euler_history():
    model = twoNodeModel()

    times, history = model.solveTransient(
        np.array([10.0, 0.0]),
        timeStep=0.5,
        stepCount=3,
    )

    np.testing.assert_allclose(times, [0.0, 0.5, 1.0, 1.5])
    assert history.shape == (4, 2, 1)
    np.testing.assert_allclose(history[:, 0, 0], 10.0)
    np.testing.assert_allclose(
        history[1],
        model.implicitEulerStep(history[0], timeStep=0.5),
    )


def test_thermal_model_transient_residual_uses_temperature_increment():
    model = twoNodeModel()
    previous = np.array([[10.0], [2.0]])
    current = np.array([[10.0], [3.0]])

    residual = model.transientResidual(current, previous, timeStep=0.5)
    expected = np.eye(2) @ ((current - previous) / 0.5)
    expected += np.array([[2.0, -1.0], [-1.0, 2.0]]) @ current
    expected -= np.array([[0.0], [3.0]])

    np.testing.assert_allclose(residual, expected)


def test_thermal_model_rejects_conflicting_duplicate_constraints():
    operators = twoNodeModel().operators
    with pytest.raises(ValueError, match='conflicting Dirichlet'):
        thermalModel(
            operators,
            dirichletNodes=[0, 0],
            dirichletValues=[300.0, 310.0],
        )


def test_iterative_and_direct_linear_solvers_agree_and_report_diagnostics():
    direct = twoNodeModel()
    iterative = thermalModel(
        direct.operators,
        dirichletNodes=[0],
        dirichletValues=10.0,
        linearSolverOptions=thermalLinearSolverOptions(
            method='cg',
            relativeTolerance=1e-12,
            preconditioner='jacobi',
        ),
    )

    np.testing.assert_allclose(iterative.solveSteady(), direct.solveSteady())
    assert iterative.lastLinearSolveInfo['converged']
    assert iterative.lastLinearSolveInfo['method'] == 'cg'
    assert iterative.lastLinearSolveInfo['relativeResidual'] <= 1e-12
