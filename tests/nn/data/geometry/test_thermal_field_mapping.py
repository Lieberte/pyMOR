import numpy as np
import pytest

from mor.nn.data.geometry import (
    mapBoundaryFieldByCoordinates,
    mapBoundaryFieldByNodeIds,
    meshIr,
)


def boundaryMesh():
    return meshIr(
        nodes=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        cells=[('tetra', np.array([[0, 1, 2, 3]], dtype=int))],
        boundaryNodeIndices={'cooling': np.array([0, 1, 2], dtype=int)},
        pointData={'originalNodeId': np.array([101, 205, 309, 412])},
    )


def test_node_id_mapping_uses_original_labels_and_boundary_order():
    result = mapBoundaryFieldByNodeIds(
        boundaryMesh(),
        'cooling',
        sourceNodeIds=[309, 101, 205],
        sourceValues=[30.0, 10.0, 20.0],
    )

    np.testing.assert_allclose(result.values, [10.0, 20.0, 30.0])
    assert result.coverage == 1.0
    assert result.asBoundaryField().location == 'node'


def test_coordinate_mapping_supports_units_nearest_and_idw():
    mesh = boundaryMesh()
    sourceMillimeters = mesh.nodes[:3] * 1000.0
    nearest = mapBoundaryFieldByCoordinates(
        mesh,
        'cooling',
        sourceMillimeters,
        [100.0, 200.0, 300.0],
        maxDistance=1e-6,
        sourceLengthScaleToMeters=1e-3,
    )
    idw = mapBoundaryFieldByCoordinates(
        mesh,
        'cooling',
        sourceMillimeters,
        [100.0, 200.0, 300.0],
        maxDistance=1e-6,
        method='idw',
        neighbors=3,
        sourceLengthScaleToMeters=1e-3,
    )

    np.testing.assert_allclose(nearest.values, [100.0, 200.0, 300.0])
    np.testing.assert_allclose(idw.values, nearest.values)
    assert nearest.maximumDistance <= 1e-12


def test_coordinate_mapping_reports_uncovered_nodes_without_filling_values():
    result = mapBoundaryFieldByCoordinates(
        boundaryMesh(),
        'cooling',
        [[0.0, 0.0, 0.0]],
        [100.0],
        maxDistance=0.1,
        requireComplete=False,
    )

    assert np.isclose(result.coverage, 1.0 / 3.0)
    np.testing.assert_array_equal(result.unmappedNodeIndices, [1, 2])
    assert np.isnan(result.values[1:]).all()
    with pytest.raises(ValueError, match='covers 1/3'):
        result.asBoundaryField()
