import numpy as np
import pytest

from mor.nn.data import (
    collateThermalOperatorCases,
    thermalOperatorCase,
    thermalOperatorDataModule,
)


def makeCase(caseId, geometryId, volumeCount, boundaryCount, queryCount):
    return thermalOperatorCase(
        caseId=caseId,
        geometryId=geometryId,
        volumeFeatures=np.ones((volumeCount, 5), dtype=np.float32),
        volumeWeights=np.linspace(1.0, 2.0, volumeCount),
        boundaryFeatures=np.ones((boundaryCount, 7), dtype=np.float32),
        boundaryWeights=np.linspace(0.5, 1.0, boundaryCount),
        globalFeatures=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        queryCoordinates=np.zeros((queryCount, 3), dtype=np.float32),
        queryTemperatures=np.arange(queryCount, dtype=np.float32),
    )


def test_thermal_operator_collate_pads_variable_meshes_and_preserves_masks():
    batch = collateThermalOperatorCases(
        [makeCase('a', 'g1', 3, 4, 5), makeCase('b', 'g2', 7, 2, 3)]
    )

    assert batch['volumeFeatures'].shape == (2, 7, 5)
    assert batch['boundaryFeatures'].shape == (2, 4, 7)
    assert batch['queryCoordinates'].shape == (2, 5, 3)
    assert batch['volumeMask'].sum(dim=1).tolist() == [3, 7]
    assert batch['boundaryMask'].sum(dim=1).tolist() == [4, 2]
    assert batch['queryMask'].sum(dim=1).tolist() == [5, 3]
    assert batch['targets'][1, 3:].abs().sum() == 0.0


def test_thermal_operator_data_split_holds_out_entire_geometry():
    cases = [
        makeCase('a0', 'motorA', 3, 3, 4),
        makeCase('a1', 'motorA', 4, 3, 4),
        makeCase('b0', 'motorB', 5, 4, 4),
    ]
    module = thermalOperatorDataModule(
        cases,
        validationGeometryIds=['motorB'],
        batchSize=2,
    )

    assert {case.geometryId for case in module.trainCases} == {'motorA'}
    assert {case.geometryId for case in module.validationCases} == {'motorB'}
    assert module.featureDimensions == {'volume': 5, 'boundary': 7, 'global': 3, 'query': 3}


def test_thermal_operator_case_rejects_invalid_measure_weights():
    with pytest.raises(ValueError, match='positive finite'):
        thermalOperatorCase(
            caseId='bad',
            geometryId='g',
            volumeFeatures=np.ones((2, 3)),
            volumeWeights=[1.0, 0.0],
            boundaryFeatures=np.ones((2, 3)),
            boundaryWeights=[1.0, 1.0],
            globalFeatures=[1.0],
            queryCoordinates=np.ones((2, 3)),
            queryTemperatures=[1.0, 2.0],
        )
