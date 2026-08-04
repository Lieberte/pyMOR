import json

import numpy as np
import pytest

from mor.models import thermalLoadCase, thermalMaterialLibrary
from mor.nn.data.geometry import meshIr, thermalProblem


def twoRegionMesh(*, overlapping=False, missing=False):
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=int)
    second = np.array([0 if overlapping else 1], dtype=int)
    if missing:
        second = np.empty(0, dtype=int)
    return meshIr(
        nodes=nodes,
        cells=[('tetra', cells)],
        cellSets={
            'WINDING_SOLID': [np.array([0], dtype=int)],
            'HOUSING_SOLID': [second],
        },
        boundaryNodeIndices={'fixed': np.array([0], dtype=int)},
    )


def problemConfig():
    return {
        'materials': {
            'copper': {
                'conductivity': 400.0,
                'density': 8900.0,
                'specificHeat': 385.0,
            },
            'aluminum': {
                'conductivity': 205.0,
                'density': 2700.0,
                'specificHeat': 900.0,
            },
        },
        'regionMaterials': {
            'WINDING_SOLID': 'copper',
            'HOUSING_SOLID': 'aluminum',
        },
        'loadCase': {
            'heatSources': {'WINDING_SOLID': 1200.0},
            'dirichletTemperatures': {'fixed': 300.0},
            'operatingMetadata': {'speedRpm': 8000, 'torqueNm': 100.0},
        },
        'capacityMode': 'lumped',
    }


def test_thermal_problem_builds_two_material_model_with_partial_heat_source():
    problem = thermalProblem.fromDict(twoRegionMesh(), problemConfig())

    model = problem.buildModel()

    assert model.order == 5
    assert model.K.isSparse
    assert model.M.isSparse
    assert model.Q.shape == (5, 1)
    assert np.isclose(model.Q.toNumpy().sum(), 1200.0 / 6.0)
    np.testing.assert_array_equal(model.dirichletNodes, [0])
    np.testing.assert_allclose(model.dirichletValues, [300.0])
    assert model.metadata['operating'] == {'speedRpm': 8000, 'torqueNm': 100.0}
    steady = model.solveSteady()
    times, history = model.solveTransient(np.full(5, 300.0), timeStep=0.05, stepCount=2)
    assert steady.shape == (5, 1)
    assert np.all(np.isfinite(steady))
    assert steady[0, 0] == 300.0
    np.testing.assert_allclose(times, [0.0, 0.05, 0.1])
    assert history.shape == (3, 5, 1)
    np.testing.assert_allclose(history[:, 0, 0], 300.0)


def test_thermal_problem_supports_mixed_scalar_and_tensor_conductivity():
    config = problemConfig()
    config['materials']['aluminum']['conductivity'] = [
        [205.0, 0.0, 0.0],
        [0.0, 205.0, 0.0],
        [0.0, 0.0, 180.0],
    ]

    model = thermalProblem.fromDict(twoRegionMesh(), config).buildModel()

    assert model.K.shape == (5, 5)
    assert np.all(np.isfinite(model.K.toNumpy()))


def test_temperature_dependent_material_reassembles_and_solves_nonlinear_problem():
    mesh = meshIr(
        nodes=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        cells=[('tetra', np.array([[0, 1, 2, 3]], dtype=int))],
        cellSets={'SOLID': [np.array([0], dtype=int)]},
        boundaryNodeIndices={'fixed': np.array([0], dtype=int)},
    )
    problem = thermalProblem.fromDict(
        mesh,
        {
            'materials': {
                'temperatureDependent': {
                    'conductivity': {
                        'temperatures': [300.0, 500.0],
                        'values': [10.0, 8.0],
                    },
                    'density': 1.0,
                    'specificHeat': {
                        'temperatures': [300.0, 500.0],
                        'values': [1.0, 2.0],
                    },
                }
            },
            'regionMaterials': {'SOLID': 'temperatureDependent'},
            'loadCase': {
                'heatSources': {'SOLID': 4000.0},
                'dirichletTemperatures': {'fixed': 300.0},
            },
            'referenceTemperature': 300.0,
            'nonlinearOptions': {
                'tolerance': 1e-9,
                'maxIterations': 100,
                'relaxation': 0.8,
            },
        },
    )

    coldOperators = problem.assembleOperators(np.full(4, 300.0))
    hotOperators = problem.assembleOperators(np.full(4, 500.0))
    assert not np.allclose(coldOperators.K.toNumpy(), hotOperators.K.toNumpy())
    assert not np.allclose(coldOperators.M.toNumpy(), hotOperators.M.toNumpy())

    model = problem.buildModel()
    steady = model.solveSteady()
    transient = model.implicitEulerStep(np.full((4, 1), 300.0), timeStep=0.1)

    assert model.isNonlinear
    assert steady.shape == (4, 1)
    assert transient.shape == (4, 1)
    assert model.lastNonlinearInfo['converged']
    assert model.lastNonlinearInfo['iterations'] > 1
    assert model.lastNonlinearInfo['relativeResidual'] <= problem.nonlinearOptions.residualTolerance
    np.testing.assert_allclose(steady[0, 0], 300.0)
    np.testing.assert_allclose(transient[0, 0], 300.0)
    np.testing.assert_allclose(model.steadyResidual(steady)[1:], 0.0, atol=1e-5)


def test_thermal_problem_configuration_round_trip_is_json_serializable():
    problem = thermalProblem.fromDict(twoRegionMesh(), problemConfig())

    serialized = json.loads(json.dumps(problem.toDict()))
    restored = thermalProblem.fromDict(twoRegionMesh(), serialized)

    assert restored.toDict() == serialized


def test_thermal_problem_rejects_missing_and_overlapping_material_assignment():
    with pytest.raises(ValueError, match='unassigned'):
        thermalProblem.fromDict(twoRegionMesh(missing=True), problemConfig()).assembleOperators()
    with pytest.raises(ValueError, match='overlapping'):
        thermalProblem.fromDict(twoRegionMesh(overlapping=True), problemConfig()).assembleOperators()


def test_thermal_problem_rejects_unknown_material_and_conflicting_dirichlet_values():
    config = problemConfig()
    config['regionMaterials']['HOUSING_SOLID'] = 'steel'
    with pytest.raises(KeyError, match='unknown thermal material'):
        thermalProblem.fromDict(twoRegionMesh(), config)

    mesh = twoRegionMesh()
    mesh.boundaryNodeIndices['alsoFixed'] = np.array([0], dtype=int)
    loadCase = thermalLoadCase(
        dirichletTemperatures={'fixed': 300.0, 'alsoFixed': 310.0}
    )
    problem = thermalProblem(
        mesh,
        materials=thermalMaterialLibrary.fromDict(problemConfig()['materials']),
        regionMaterials=problemConfig()['regionMaterials'],
        loadCase=loadCase,
    )
    with pytest.raises(ValueError, match='conflicting Dirichlet'):
        problem.buildModel()


def test_meter_and_millimeter_meshes_assemble_the_same_physical_operators():
    meterMesh = twoRegionMesh()
    millimeterMesh = meshIr(
        nodes=meterMesh.nodes * 1000.0,
        cells=meterMesh.cells,
        boundaryNodeIndices=meterMesh.boundaryNodeIndices,
        cellSets=meterMesh.cellSets,
    )
    meterProblem = thermalProblem.fromDict(meterMesh, problemConfig())
    millimeterConfig = problemConfig()
    millimeterConfig['lengthUnit'] = 'mm'
    millimeterProblem = thermalProblem.fromDict(millimeterMesh, millimeterConfig)

    meterOperators = meterProblem.assembleOperators()
    millimeterOperators = millimeterProblem.assembleOperators()

    np.testing.assert_allclose(millimeterOperators.K.toNumpy(), meterOperators.K.toNumpy())
    np.testing.assert_allclose(millimeterOperators.M.toNumpy(), meterOperators.M.toNumpy())
    np.testing.assert_allclose(millimeterOperators.Q.toNumpy(), meterOperators.Q.toNumpy())
    np.testing.assert_allclose(millimeterOperators.elementVolumes, meterOperators.elementVolumes)
    np.testing.assert_allclose(millimeterProblem.mesh.nodes, millimeterMesh.nodes)


def contactMesh():
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    return meshIr(
        nodes=nodes,
        cells=[('tetra', np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=int))],
        cellSets={
            'HOT_SOLID': [np.array([0], dtype=int)],
            'COLD_SOLID': [np.array([1], dtype=int)],
        },
        boundaryFaces={
            'hotContact': [('triangle', np.array([[0, 1, 2]], dtype=int))],
            'coldContact': [('triangle', np.array([[4, 5, 6]], dtype=int))],
        },
        boundaryNodeIndices={
            'hotContact': np.array([0, 1, 2], dtype=int),
            'coldContact': np.array([4, 5, 6], dtype=int),
            'fixed': np.array([7], dtype=int),
        },
    )


def contactConfig(coefficient):
    return {
        'materials': {
            'solid': {'conductivity': 10.0, 'density': 1.0, 'specificHeat': 1.0},
        },
        'regionMaterials': {'HOT_SOLID': 'solid', 'COLD_SOLID': 'solid'},
        'loadCase': {
            'heatSources': {'HOT_SOLID': 60.0},
            'dirichletTemperatures': {'fixed': 300.0},
            'contacts': {
                'joint': {
                    'primaryBoundary': 'hotContact',
                    'secondaryBoundary': 'coldContact',
                    'coefficient': coefficient,
                    'mapping': 'matching',
                    'maxDistance': 1e-9,
                }
            },
        },
    }


def test_contact_interface_closes_energy_balance_and_high_tcc_approaches_equal_temperature():
    lowModel = thermalProblem.fromDict(contactMesh(), contactConfig(1.0)).buildModel()
    highModel = thermalProblem.fromDict(contactMesh(), contactConfig(1e8)).buildModel()

    low = lowModel.solveSteady()
    high = highModel.solveSteady()
    lowGap = np.mean(np.abs(low[:3, 0] - low[4:7, 0]))
    highGap = np.mean(np.abs(high[:3, 0] - high[4:7, 0]))

    assert highGap < lowGap * 1e-5
    np.testing.assert_allclose(highModel.steadyResidual(high)[np.arange(7)], 0.0, atol=2e-6)
    assert highModel.operators.metadata['contacts']['joint']['coverage'] == 1.0
    assert highModel.operators.metadata['contacts']['joint']['oneToOne']


def test_internal_contact_boundary_cannot_be_reused_as_external_convection():
    config = contactConfig(100.0)
    config['loadCase']['convection'] = {
        'hotContact': {'coefficient': 20.0, 'ambientTemperature': 300.0}
    }
    with pytest.raises(ValueError, match='contact boundaries cannot also be external'):
        thermalProblem.fromDict(contactMesh(), config)
