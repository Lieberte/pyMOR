import json
from dataclasses import replace

import numpy as np
import pytest

from mor.nn.data import (
    loadThermalDatasetAuto,
    saveThermalComponentDataset,
    saveWholeMotorDataset,
    splitThermalCasesByMotorId,
    subsampleThermalComponentCase,
)
from tests.nn.data.test_thermal_component_case import makeComponentCase


def test_component_dataset_boundary_semantics_are_remapped_to_motor_schema(tmp_path):
    case = replace(
        makeComponentCase(),
        metadata={'motorId': 'motor-a', 'parentCaseId': '12000'},
    )
    root = tmp_path / 'dataset'
    saveThermalComponentDataset(
        (case,),
        root,
        boundaryTypeNames={0: 'convection', 1: 'fixedTemperature'},
        datasetMetadata={'motorId': 'motor-a'},
    )

    dataset = loadThermalDatasetAuto(root)

    assert dataset.viewType == 'components'
    assert len(dataset.cases) == 1
    np.testing.assert_array_equal(
        dataset.cases[0].boundary.boundaryTypeIds,
        [1, 1, 1, 2, 2, 2],
    )
    assert dataset.cases[0].operatingFeatures['absoluteTemperatureAnchor'] == 1.0
    assert dataset.cases[0].operatingFeatures['speedRpm'] == 12000.0
    assert dataset.groups[0].motorId == 'motor-a'
    assert dataset.groups[0].parentCaseId == '12000'


def test_cases_subdirectory_resolves_its_parent_dataset(tmp_path):
    case = replace(
        makeComponentCase(),
        operatingFeatures={},
        metadata={'motorId': 'motor-a', 'parentCaseId': '7500'},
    )
    root = tmp_path / 'dataset'
    saveThermalComponentDataset(
        (case,), root, boundaryTypeNames={0: 'convection', 1: 'fixedTemperature'}
    )

    dataset = loadThermalDatasetAuto(root / 'cases')

    assert dataset.sourcePath == root / 'dataset.json'
    assert dataset.cases[0].operatingFeatures['speedRpm'] == 7500.0


def test_whole_motor_view_expands_component_references_and_keeps_grouping(tmp_path):
    base = makeComponentCase()
    case = replace(
        base,
        metadata={
            **base.metadata,
            'motorId': 'motor-a',
            'parentCaseId': '12000',
            'regionUuid': 'region-a',
            'meshUuid': 'mesh-a',
            'materialUuid': 'material-a',
            'surfacePatches': [],
            'absoluteTemperatureAnchorAvailable': True,
            'configuredHeatSourcePowerW': base.volume.totalHeatSourcePowerW,
            'expectedBoundaryHeatFlowRateW': base.boundary.totalHeatFlowRateW,
        },
    )
    root = tmp_path / 'dataset'
    saveThermalComponentDataset(
        (case,),
        root,
        boundaryTypeNames={0: 'convection', 1: 'fixedTemperature'},
        datasetMetadata={'motorId': 'motor-a'},
    )
    saveWholeMotorDataset(
        (case,),
        root / 'whole_motor',
        componentDatasetDirectory=root,
        motorId='motor-a',
        parentCaseId='12000',
    )

    dataset = loadThermalDatasetAuto(root, view='wholeMotor')

    assert dataset.viewType == 'wholeMotor'
    assert len(dataset.cases) == 1
    assert dataset.groups[0].caseIds == (case.caseId,)
    assert dataset.cases[0].metadata['motorId'] == 'motor-a'
    assert dataset.cases[0].metadata['parentCaseId'] == '12000'
    np.testing.assert_array_equal(
        dataset.cases[0].boundary.boundaryTypeIds,
        [1, 1, 1, 2, 2, 2],
    )


def test_views_manifest_selects_component_or_whole_motor_view(tmp_path):
    base = makeComponentCase()
    case = replace(
        base,
        metadata={
            **base.metadata,
            'regionUuid': 'region-a',
            'meshUuid': 'mesh-a',
            'materialUuid': 'material-a',
            'surfacePatches': [],
            'absoluteTemperatureAnchorAvailable': True,
            'configuredHeatSourcePowerW': base.volume.totalHeatSourcePowerW,
            'expectedBoundaryHeatFlowRateW': base.boundary.totalHeatFlowRateW,
        },
    )
    root = tmp_path / 'dataset'
    saveThermalComponentDataset(
        (case,), root, boundaryTypeNames={0: 'convection', 1: 'fixedTemperature'}
    )
    saveWholeMotorDataset(
        (case,),
        root / 'whole_motor',
        componentDatasetDirectory=root,
        motorId='motor-a',
        parentCaseId='baseline',
    )
    (root / 'views.json').write_text(
        json.dumps(
            {
                'views': {
                    'component': 'dataset.json',
                    'wholeMotor': 'whole_motor/dataset.json',
                }
            }
        ),
        encoding='utf-8',
    )

    assert loadThermalDatasetAuto(root).viewType == 'components'
    assert loadThermalDatasetAuto(root, view='wholeMotor').viewType == 'wholeMotor'


def test_used_boundary_id_without_schema_name_fails_clearly(tmp_path):
    case = makeComponentCase()
    root = tmp_path / 'dataset'
    indexPath = saveThermalComponentDataset(
        (case,), root, boundaryTypeNames={0: 'convection'}
    )
    index = json.loads(indexPath.read_text(encoding='utf-8'))
    index['featureSchema']['boundaryTypeNames'] = {'0': 'convection'}
    indexPath.write_text(json.dumps(index), encoding='utf-8')

    with pytest.raises(ValueError, match='missing from dataset featureSchema'):
        loadThermalDatasetAuto(root)


def test_cross_motor_split_keeps_every_case_of_a_motor_on_one_side():
    cases = (
        replace(makeComponentCase('a-2500'), metadata={'motorId': 'motor-a'}),
        replace(makeComponentCase('a-5000'), metadata={'motorId': 'motor-a'}),
        replace(makeComponentCase('b-2500'), metadata={'motorId': 'motor-b'}),
    )

    trainCases, validationCases = splitThermalCasesByMotorId(
        cases, validationMotorIds=('motor-b',)
    )

    assert {case.metadata['motorId'] for case in trainCases} == {'motor-a'}
    assert {case.metadata['motorId'] for case in validationCases} == {'motor-b'}


def test_training_subsample_preserves_volume_and_patch_integrals():
    case = makeComponentCase(extraPoint=True)

    sampled = subsampleThermalComponentCase(
        case,
        maximumVolumeSamples=4,
        maximumBoundarySamples=2,
        maximumQuerySamples=3,
        randomSeed=4,
    )

    assert len(sampled.volume.weightsM3) == 4
    assert len(sampled.boundary.weightsM2) == 2
    assert len(sampled.temperature.weightsM3) == 3
    np.testing.assert_allclose(
        sampled.volume.totalHeatSourcePowerW,
        case.volume.totalHeatSourcePowerW,
    )
    np.testing.assert_allclose(sampled.volume.weightsM3.sum(), case.volume.weightsM3.sum())
    np.testing.assert_allclose(
        sampled.temperature.weightsM3.sum(), case.temperature.weightsM3.sum()
    )
    np.testing.assert_allclose(
        list(sampled.boundary.heatFlowRateWByPatch().values()),
        list(case.boundary.heatFlowRateWByPatch().values()),
    )
