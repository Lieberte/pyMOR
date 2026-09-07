import json

import numpy as np
import pytest

from mor.nn.data import (
    geometryRecognitionCase,
    loadGeometryRecognitionDatasetRecords,
)
from mor.nn.runtime import (
    extractGeometryDescriptor,
    geometryComponentRecognizer,
    geometryDescriptorConfig,
)


def _boxCase(name, lengths, componentType=None, transform=None):
    x, y, z = np.asarray(lengths, dtype=np.float64) / 2.0
    nodes = np.array(
        [
            [-x, -y, -z],
            [x, -y, -z],
            [x, y, -z],
            [-x, y, -z],
            [-x, -y, z],
            [x, -y, z],
            [x, y, z],
            [-x, y, z],
        ]
    )
    triangles = np.array(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    volumeCoordinates = np.zeros((1, 3))
    volumeWeights = np.array([8.0 * x * y * z])
    if transform is not None:
        nodes, volumeCoordinates, volumeWeights = transform(
            nodes, volumeCoordinates, volumeWeights
        )
    return geometryRecognitionCase(
        sourceId=name,
        geometryId=f'{name}-geometry',
        meshUuid=f'{name}-mesh',
        componentType=componentType,
        componentInstance=name,
        nodesMeters=nodes,
        surfaceTriangles=triangles,
        volumeCoordinatesMeters=volumeCoordinates,
        volumeWeightsM3=volumeWeights,
    )


def _smallConfig():
    return geometryDescriptorConfig(
        surfaceSampleCount=1024,
        pairSampleCount=2048,
        triangleChunkSize=64,
        radialBins=8,
        pairDistanceBins=8,
        normalBins=6,
        axisBins=6,
    )


def test_descriptor_is_translation_scale_and_rotation_invariant():
    base = _boxCase('shaft', (0.4, 0.7, 3.0), 'shaft')
    angle = 0.73
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    def transform(nodes, centers, weights):
        scale = 4.2
        offset = np.array([8.0, -3.0, 2.5])
        return (
            nodes @ rotation.T * scale + offset,
            centers @ rotation.T * scale + offset,
            weights * scale**3,
        )

    changed = _boxCase('changed', (0.4, 0.7, 3.0), 'shaft', transform)
    first = extractGeometryDescriptor(base, _smallConfig())
    second = extractGeometryDescriptor(changed, _smallConfig())
    np.testing.assert_allclose(first.values, second.values, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(second.centerMeters, [8.0, -3.0, 2.5])
    assert second.scaleMeters == pytest.approx(first.scaleMeters * 4.2, rel=1e-12)


def test_default_descriptor_is_128_dimensional():
    descriptor = extractGeometryDescriptor(
        _boxCase('box', (1.0, 2.0, 3.0), 'housing')
    )

    assert descriptor.values.shape == (128,)
    assert len(descriptor.featureNames) == 128


def test_descriptor_encodes_disconnected_surface_components():
    first = _boxCase('first', (1.0, 1.0, 1.0), 'magnets')
    second = _boxCase('second', (1.0, 1.0, 1.0), 'magnets')
    nodeOffset = first.nodesMeters.shape[0]
    combined = geometryRecognitionCase(
        sourceId='combined',
        geometryId='combined',
        meshUuid='combined',
        componentType='magnets',
        componentInstance='magnets',
        nodesMeters=np.concatenate(
            (first.nodesMeters - [2.0, 0.0, 0.0], second.nodesMeters + [2.0, 0.0, 0.0])
        ),
        surfaceTriangles=np.concatenate(
            (first.surfaceTriangles, second.surfaceTriangles + nodeOffset)
        ),
        volumeCoordinatesMeters=np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        volumeWeightsM3=np.array([1.0, 1.0]),
    )

    descriptor = extractGeometryDescriptor(combined, _smallConfig())

    assert descriptor.surfaceComponentCount == 2
    componentCountIndex = descriptor.featureNames.index('surface_component_0')
    assert descriptor.values[componentCountIndex] == pytest.approx(np.log1p(2))


def test_descriptor_is_invariant_to_node_and_triangle_numbering():
    case = _boxCase('box', (1.0, 2.0, 3.0), 'housing')
    permutation = np.array([6, 2, 0, 7, 3, 5, 1, 4])
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(permutation.size)
    reordered = geometryRecognitionCase(
        sourceId='reordered',
        geometryId='reordered',
        meshUuid=None,
        componentType='housing',
        componentInstance='housing',
        nodesMeters=case.nodesMeters[permutation],
        surfaceTriangles=inverse[case.surfaceTriangles[::-1]],
        volumeCoordinatesMeters=case.volumeCoordinatesMeters,
        volumeWeightsM3=case.volumeWeightsM3,
    )
    first = extractGeometryDescriptor(case, _smallConfig())
    second = extractGeometryDescriptor(reordered, _smallConfig())
    np.testing.assert_allclose(first.values, second.values, atol=3e-2, rtol=3e-2)


def test_recognizer_fits_predicts_and_round_trips(tmp_path):
    cases = (
        _boxCase('shaft-a', (0.4, 0.5, 3.0), 'shaft'),
        _boxCase('shaft-b', (0.5, 0.6, 2.8), 'shaft'),
        _boxCase('plate-a', (3.0, 2.5, 0.3), 'magnetic_isolation_plate'),
        _boxCase('housing-a', (2.0, 1.8, 1.6), 'housing'),
    )
    recognizer = geometryComponentRecognizer(
        config=_smallConfig(), minimumMargin=0.0
    ).fit(cases)
    prediction = recognizer.predict(cases[0])
    assert prediction.predictedType == 'shaft'
    assert not prediction.unknown
    assert prediction.candidates[0].componentType == 'shaft'

    modelPath = recognizer.save(tmp_path / 'recognizer.json')
    restored = geometryComponentRecognizer.load(modelPath)
    restoredPrediction = restored.predict(cases[2])
    assert restoredPrediction.predictedType == 'magnetic_isolation_plate'
    assert restoredPrediction.toDict()['geometry']['sourceNodeCount'] == 8


def test_recognizer_rejects_distant_geometry_as_unknown():
    recognizer = geometryComponentRecognizer(
        config=_smallConfig(), minimumMargin=0.0
    ).fit(
        (
            _boxCase('shaft', (0.4, 0.5, 3.0), 'shaft'),
            _boxCase('plate', (3.0, 2.5, 0.3), 'magnetic_isolation_plate'),
            _boxCase('housing', (2.0, 1.8, 1.6), 'housing'),
        )
    )
    irregular = _boxCase('irregular', (9.0, 0.8, 0.15))
    prediction = recognizer.predict(irregular)
    assert prediction.unknown
    assert prediction.predictedType == 'unknown'
    assert 0.0 <= prediction.confidence <= 1.0


def test_dataset_records_are_deduplicated_by_mesh_uuid(tmp_path):
    casesRoot = tmp_path / 'motors' / 'motor' / 'cases'
    entries = []
    for index in range(2):
        caseRoot = casesRoot / f'case-{index}'
        caseRoot.mkdir(parents=True)
        np.savez_compressed(
            caseRoot / 'data.npz',
            nodesMeters=np.zeros((3, 3)),
            surfaceTriangles=np.array([[0, 1, 2]]),
        )
        entries.append(
            {
                'caseId': f'case-{index}',
                'geometryId': 'shaft-mesh-a',
                'meshUuid': 'mesh-a',
                'componentType': 'shaft',
                'componentInstance': 'shaft',
                'path': caseRoot.relative_to(tmp_path).as_posix(),
            }
        )
    (tmp_path / 'dataset.json').write_text(
        json.dumps({'connectivityStored': True, 'cases': entries}), encoding='utf-8'
    )
    records = loadGeometryRecognitionDatasetRecords(tmp_path)
    assert len(records) == 1
    assert records[0].componentType == 'shaft'


def test_dataset_rejects_conflicting_labels_for_one_mesh(tmp_path):
    entries = [
        {
            'caseId': 'one', 'geometryId': 'one', 'meshUuid': 'same',
            'componentType': 'shaft', 'path': 'one',
        },
        {
            'caseId': 'two', 'geometryId': 'two', 'meshUuid': 'same',
            'componentType': 'housing', 'path': 'two',
        },
    ]
    (tmp_path / 'dataset.json').write_text(
        json.dumps({'connectivityStored': True, 'cases': entries}), encoding='utf-8'
    )
    with pytest.raises(ValueError, match='conflicting component types'):
        loadGeometryRecognitionDatasetRecords(tmp_path)


def test_dataset_uses_component_type_as_declared_classification_target(tmp_path):
    caseRoot = tmp_path / 'case'
    caseRoot.mkdir()
    np.savez_compressed(
        caseRoot / 'data.npz',
        nodesMeters=np.zeros((3, 3)),
        surfaceTriangles=np.array([[0, 1, 2]]),
    )
    entry = {
        'caseId': 'case',
        'geometryId': 'shaft-mesh',
        'meshUuid': 'mesh',
        'componentType': 'shaft',
        'path': 'case',
    }
    (tmp_path / 'dataset.json').write_text(
        json.dumps(
            {
                'connectivityStored': True,
                'tasks': {
                    'geometryComponentClassification': {
                        'taskType': 'multiclassClassification',
                        'targetField': 'componentType',
                        'classes': ['shaft'],
                    }
                },
                'cases': [entry],
            }
        ),
        encoding='utf-8',
    )

    records = loadGeometryRecognitionDatasetRecords(tmp_path)

    assert records[0].componentType == 'shaft'


def test_dataset_rejects_wrong_declared_classification_target(tmp_path):
    (tmp_path / 'dataset.json').write_text(
        json.dumps(
            {
                'connectivityStored': True,
                'tasks': {
                    'geometryComponentClassification': {
                        'targetField': 'componentInstance',
                        'classes': ['shaft'],
                    }
                },
                'cases': [{'caseId': 'case'}],
            }
        ),
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='targetField must be componentType'):
        loadGeometryRecognitionDatasetRecords(tmp_path)


def test_descriptor_config_rejects_fractional_and_boolean_counts():
    with pytest.raises(ValueError, match='positive integers'):
        geometryDescriptorConfig(surfaceSampleCount=1.5)
    with pytest.raises(ValueError, match='positive integers'):
        geometryDescriptorConfig(pairSampleCount=True)


def test_geometry_case_rejects_fractional_triangle_indices():
    with pytest.raises(ValueError, match='integer node indices'):
        geometryRecognitionCase(
            sourceId='bad',
            geometryId='bad',
            meshUuid='bad',
            componentType=None,
            componentInstance=None,
            nodesMeters=np.eye(3),
            surfaceTriangles=np.array([[0.0, 1.0, 1.5]]),
        )


def test_recognizer_rejects_conflicting_direct_fit_labels():
    first = _boxCase('shared', (1.0, 2.0, 3.0), 'shaft')
    second = geometryRecognitionCase(
        sourceId='second',
        geometryId='second',
        meshUuid=first.meshUuid,
        componentType='housing',
        componentInstance='second',
        nodesMeters=first.nodesMeters,
        surfaceTriangles=first.surfaceTriangles,
        volumeCoordinatesMeters=first.volumeCoordinatesMeters,
        volumeWeightsM3=first.volumeWeightsM3,
    )

    with pytest.raises(ValueError, match='conflicting component types'):
        geometryComponentRecognizer(config=_smallConfig()).fit((first, second))


def test_recognizer_rejects_invalid_top_k():
    recognizer = geometryComponentRecognizer(config=_smallConfig(), minimumMargin=0.0).fit(
        (
            _boxCase('shaft', (0.4, 0.5, 3.0), 'shaft'),
            _boxCase('housing', (2.0, 1.8, 1.6), 'housing'),
        )
    )

    with pytest.raises(ValueError, match='topK'):
        recognizer.predict(_boxCase('query', (0.4, 0.5, 3.0)), topK=0)
