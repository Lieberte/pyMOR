import json

import numpy as np

from mor.nn.data import (
    loadThermalComponentCase,
    loadThermalComponentDataset,
    saveThermalComponentCase,
    saveThermalComponentDataset,
    thermalBoundarySamples,
    thermalComponentCase,
    thermalComponentFeatureSchema,
    thermalOperatorDataModule,
    thermalOperatorFeatureScaling,
    thermalTemperatureSamples,
    thermalVolumeSamples,
)


def makeComponentCase(caseId='shaft_coarse', geometryId='shaft_v1', scale=1.0, extraPoint=False):
    coordinates = scale * np.array(
        [
            [-1.0, -0.5, -0.25],
            [1.0, -0.5, -0.25],
            [-1.0, 0.5, -0.25],
            [1.0, 0.5, -0.25],
            [-1.0, -0.5, 0.25],
            [1.0, -0.5, 0.25],
            [-1.0, 0.5, 0.25],
            [1.0, 0.5, 0.25],
        ]
    )
    if extraPoint:
        coordinates = np.vstack((coordinates, [[0.0, 0.0, 0.0]]))
    count = coordinates.shape[0]
    boundaryCoordinates = coordinates[:6]
    normals = boundaryCoordinates / np.linalg.norm(boundaryCoordinates, axis=1)[:, None]
    return thermalComponentCase(
        caseId=caseId,
        geometryId=geometryId,
        componentType='shaft',
        volume=thermalVolumeSamples(
            coordinatesMeters=coordinates,
            conductivityWPerMK=np.linspace(35.0, 40.0, count),
            densityKgPerM3=np.full(count, 7800.0),
            specificHeatJPerKgK=np.full(count, 500.0),
            heatSourceWPerM3=np.linspace(1e5, 2e5, count),
            weightsM3=np.full(count, 0.001 / count),
        ),
        boundary=thermalBoundarySamples(
            coordinatesMeters=boundaryCoordinates,
            normals=normals,
            htcWPerM2K=np.linspace(100.0, 200.0, 6),
            ambientTemperatureK=np.full(6, 300.0),
            weightsM2=np.full(6, 0.02 / 6),
            boundaryTypeIds=[0, 0, 0, 1, 1, 1],
        ),
        temperature=thermalTemperatureSamples(
            coordinatesMeters=coordinates,
            temperatureK=np.linspace(310.0, 330.0, count),
            weightsM3=np.full(count, 0.001 / count),
        ),
        operatingFeatures={'speedRpm': 12000.0},
        metadata={'meshName': caseId},
    )


def test_component_case_derives_geometry_frame_and_operator_features_from_points():
    case = makeComponentCase()
    schema = thermalComponentFeatureSchema(
        operatingFeatureNames=['speedRpm'], boundaryTypeCount=2
    )
    scaling = thermalOperatorFeatureScaling(
        lengthMeters=1.0,
        temperatureReferenceK=300.0,
        temperatureScaleK=10.0,
        operatingScales={'speedRpm': 12000.0},
    )

    operatorCase = case.toOperatorCase(schema=schema, scaling=scaling)

    np.testing.assert_allclose(case.coordinateFrame.centerMeters, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(case.coordinateFrame.halfExtentMeters, [1.0, 0.5, 0.25])
    assert operatorCase.volumeFeatures.shape == (8, 15)
    assert operatorCase.boundaryFeatures.shape == (6, 10)
    assert operatorCase.globalFeatures.shape == (6,)
    np.testing.assert_allclose(operatorCase.globalFeatures[:3], [1.0, 0.5, 0.25])
    np.testing.assert_allclose(operatorCase.globalFeatures[3:], [0.001, 0.02, 1.0])
    np.testing.assert_allclose(operatorCase.queryCoordinates.min(axis=0), -1.0)
    np.testing.assert_allclose(operatorCase.queryCoordinates.max(axis=0), 1.0)
    np.testing.assert_allclose(operatorCase.queryTemperatures[[0, -1], 0], [1.0, 3.0])


def test_component_case_directory_round_trip_preserves_physical_arrays(tmp_path):
    case = makeComponentCase()

    manifestPath = saveThermalComponentCase(case, tmp_path / 'shaft_case')
    restored = loadThermalComponentCase(tmp_path / 'shaft_case')
    manifest = json.loads(manifestPath.read_text(encoding='utf-8'))

    assert manifest['formatVersion'] == 'thermal-component-case-v1'
    assert restored.caseId == case.caseId
    assert restored.geometryId == case.geometryId
    assert restored.componentType == 'shaft'
    np.testing.assert_allclose(restored.volume.coordinatesMeters, case.volume.coordinatesMeters)
    np.testing.assert_allclose(restored.volume.conductivityWPerMK, case.volume.conductivityWPerMK)
    np.testing.assert_allclose(restored.boundary.htcWPerM2K, case.boundary.htcWPerM2K)
    np.testing.assert_allclose(restored.temperature.temperatureK, case.temperature.temperatureK)
    assert restored.metadata == case.metadata


def test_component_dataset_keeps_remeshed_geometry_in_same_validation_split(tmp_path):
    cases = (
        makeComponentCase('shaft_coarse', 'shaft_v1'),
        makeComponentCase('shaft_fine', 'shaft_v1', extraPoint=True),
        makeComponentCase('housing', 'housing_v1', scale=1.2),
    )
    indexPath = saveThermalComponentDataset(
        cases,
        tmp_path / 'components',
        boundaryTypeNames={0: 'convection', 1: 'fixedTemperature'},
    )
    restored = loadThermalComponentDataset(tmp_path / 'components')
    assert len(restored) == 3
    index = json.loads(indexPath.read_text(encoding='utf-8'))
    assert index['featureSchema']['boundaryTypeNames'] == {
        '0': 'convection',
        '1': 'fixedTemperature',
    }
    module = thermalOperatorDataModule.fromComponentDataset(
        tmp_path / 'components',
        validationGeometryIds=['shaft_v1'],
        schema=thermalComponentFeatureSchema(boundaryTypeCount=2),
        batchSize=2,
    )

    assert {case.caseId for case in module.validationCases} == {
        'shaft_coarse',
        'shaft_fine',
    }
    assert {case.caseId for case in module.trainCases} == {'housing'}
