import numpy as np
import pytest

from mor.nn.data import (
    geometryPointCloudSample,
    geometryRecognitionCase,
    randomRotationMatrix,
    sampleGeometryPointCloud,
    stratifiedGeometryPointCloudSplit,
)


def _tetraCase(name='tetra', scale=1.0, offset=None):
    offset = np.zeros(3) if offset is None else np.asarray(offset)
    nodes = scale * np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ) + offset
    return geometryRecognitionCase(
        sourceId=name,
        geometryId=name,
        meshUuid=name,
        componentType='shaft',
        componentInstance=name,
        nodesMeters=nodes,
        surfaceTriangles=np.array(
            [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64
        ),
        volumeCoordinatesMeters=np.asarray([[0.25, 0.25, 0.25]]) * scale + offset,
        volumeWeightsM3=np.asarray([scale**3 / 6.0]),
    )


def test_geometry_point_cloud_is_translation_and_scale_invariant():
    first = sampleGeometryPointCloud(_tetraCase(), 256, seed=4)
    second = sampleGeometryPointCloud(
        _tetraCase('changed', scale=7.0, offset=[4.0, -2.0, 9.0]), 256, seed=4
    )

    np.testing.assert_allclose(first, second, atol=2e-6, rtol=2e-6)


def test_random_rotation_matrix_is_orthogonal():
    rotation = randomRotationMatrix(np.random.default_rng(8))

    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-6)


def test_stratified_split_excludes_singletons_and_mesh_leakage():
    values = np.zeros((16, 6), dtype=np.float32)
    samples = tuple(
        geometryPointCloudSample(
            geometryId=f'{label}-{index}',
            meshUuid=f'{label}-{index}',
            componentType=label,
            pointsAndNormals=values,
        )
        for label, count in [('shaft', 3), ('gear', 2), ('sleeve', 1)]
        for index in range(count)
    )

    train, validation, excluded = stratifiedGeometryPointCloudSplit(samples, seed=2)

    assert {item.componentType for item in excluded} == {'sleeve'}
    assert {item.componentType for item in validation} == {'shaft', 'gear'}
    assert {item.meshUuid for item in train}.isdisjoint(
        item.meshUuid for item in validation
    )
