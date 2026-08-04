import meshio
import numpy as np
import pytest

from mor.nn.data.geometry import (
    assembleMeshThermalOperators,
    convectionBoundary,
    meshIoToIr,
    meshIr,
)
from mor.models import thermalBoundaryField, thermalContactInterface


def unitTetraData():
    nodes = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    cells = np.array([[0, 1, 2, 3]], dtype=int)
    return nodes, cells


def test_meshio_conversion_preserves_indices_and_physics_data():
    nodes, cells = unitTetraData()
    source = meshio.Mesh(
        points=nodes,
        cells=[('tetra', cells)],
        point_data={'initialTemperature': np.array([300.0, 301.0, 302.0, 303.0])},
        cell_data={'conductivity': [np.array([12.0])]},
        cell_sets={'core': [np.array([0], dtype=int)]},
    )

    converted = meshIoToIr(source)

    np.testing.assert_array_equal(converted.boundaryNodeIndices['core'], [0, 1, 2, 3])
    np.testing.assert_allclose(converted.pointData['initialTemperature'], [300.0, 301.0, 302.0, 303.0])
    np.testing.assert_allclose(converted.cellData['conductivity'][0], [12.0])
    np.testing.assert_array_equal(converted.cellSets['core'][0], [0])


def test_mesh_thermal_assembly_resolves_cell_data_and_named_boundaries():
    nodes, cells = unitTetraData()
    mesh = meshIr(
        nodes=nodes,
        cells=[('tetra', cells)],
        boundaryFaces={
            'cooling': [('triangle', np.array([[0, 1, 2]], dtype=int))],
            'heated': [('triangle', np.array([[0, 1, 2]], dtype=int))],
        },
        cellData={
            'conductivity': [np.array([2.0])],
            'density': [np.array([3.0])],
            'specificHeat': [np.array([4.0])],
            'heatSource': [np.array([6.0])],
        },
    )

    result = assembleMeshThermalOperators(
        mesh,
        conductivity='conductivity',
        density='density',
        specificHeat='specificHeat',
        heatSource='heatSource',
        convection={'cooling': convectionBoundary(coefficient=2.0, ambientTemperature=10.0)},
        heatFlux={'heated': 6.0},
    )

    assert result.conductivity.isSparse
    assert result.capacity.isSparse
    assert result.elementCount == 1
    assert np.isclose(result.capacity.toNumpy().sum(), 2.0)
    assert np.isclose(result.load.toNumpy().sum(), 1.0 + 10.0 + 3.0)


def test_mesh_thermal_assembly_resolves_materials_from_cell_sets():
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
    mesh = meshIr(
        nodes=nodes,
        cells=[('tetra', cells)],
        cellSets={
            'copper': [np.array([0], dtype=int)],
            'steel': [np.array([1], dtype=int)],
        },
    )

    result = assembleMeshThermalOperators(
        mesh,
        conductivity={'copper': 400.0, 'steel': 20.0},
        density={'copper': 8900.0, 'steel': 7800.0},
        specificHeat={'copper': 385.0, 'steel': 500.0},
    )

    assert result.elementCount == 2
    assert result.conductivity.isSparse
    assert result.capacity.isSparse


def test_mesh_thermal_assembly_maps_named_nodal_boundary_fields():
    nodes, cells = unitTetraData()
    mesh = meshIr(
        nodes=nodes,
        cells=[('tetra', cells)],
        boundaryFaces={
            'surface': [('triangle', np.array([[0, 1, 2]], dtype=int))],
        },
        boundaryNodeIndices={'surface': np.array([2, 0, 1], dtype=int)},
    )

    base = assembleMeshThermalOperators(mesh, conductivity=1.0)
    result = assembleMeshThermalOperators(
        mesh,
        conductivity=1.0,
        convection={
            'surface': convectionBoundary(
                coefficient=thermalBoundaryField([1.0, 2.0, 3.0], location='node'),
                ambientTemperature=10.0,
            )
        },
        heatFlux={
            'surface': thermalBoundaryField([3.0, 6.0, 9.0, 0.0], location='globalNode')
        },
    )

    expectedBoundary = np.zeros((4, 4), dtype=float)
    expectedBoundary[:3, :3] = np.array(
        [
            [2.0 / 15.0, 3.0 / 40.0, 1.0 / 12.0],
            [3.0 / 40.0, 1.0 / 6.0, 11.0 / 120.0],
            [1.0 / 12.0, 11.0 / 120.0, 1.0 / 5.0],
        ]
    )
    np.testing.assert_allclose(result.K.toNumpy() - base.K.toNumpy(), expectedBoundary)
    assert np.isclose(result.Q.toNumpy().sum(), 10.0 + 3.0)


def test_nearest_contact_mapping_reports_distance_and_rejects_out_of_range_nodes():
    shift = np.array([0.0, 0.0, 1e-4])
    primary = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    nodes = np.vstack([primary, [0.0, 0.0, 1.0], primary + shift, [0.0, 0.0, -1.0]])
    mesh = meshIr(
        nodes=nodes,
        cells=[('tetra', np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=int))],
        boundaryFaces={
            'primary': [('triangle', np.array([[0, 1, 2]], dtype=int))],
            'secondary': [('triangle', np.array([[4, 5, 6]], dtype=int))],
        },
        boundaryNodeIndices={
            'primary': np.array([0, 1, 2]),
            'secondary': np.array([4, 5, 6]),
        },
    )
    interface = thermalContactInterface(
        'primary',
        'secondary',
        coefficient=100.0,
        mapping='nearest',
        maxDistance=1e-3,
    )

    result = assembleMeshThermalOperators(mesh, conductivity=1.0, contacts={'joint': interface})

    assert np.isclose(result.metadata['contacts']['joint']['maximumDistance'], 1e-4)
    with pytest.raises(ValueError, match='outside maxDistance'):
        assembleMeshThermalOperators(
            mesh,
            conductivity=1.0,
            contacts={
                'joint': thermalContactInterface(
                    'primary',
                    'secondary',
                    coefficient=100.0,
                    mapping='nearest',
                    maxDistance=1e-5,
                )
            },
        )
