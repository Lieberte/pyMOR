import numpy as np
from scipy.sparse import csr_matrix, issparse

from mor.operators import (
    applyDirichletConditions,
    assembleLinearTetraThermalOperators,
    matrixOperator,
)


def unitTetra():
    nodes = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    elements = np.array([[0, 1, 2, 3]], dtype=int)
    return nodes, elements


def test_linear_tetra_thermal_volume_operators():
    nodes, elements = unitTetra()
    result = assembleLinearTetraThermalOperators(
        nodes,
        elements,
        conductivity=2.0,
        density=3.0,
        specificHeat=4.0,
        heatSource=6.0,
    )

    gradients = np.array(
        [[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    expectedConductivity = (1.0 / 6.0) * 2.0 * gradients @ gradients.T
    expectedCapacity = (3.0 * 4.0 / 6.0 / 20.0) * (np.ones((4, 4)) + np.eye(4))
    expectedLoad = np.full((4, 1), 6.0 / 6.0 / 4.0)

    assert result.nodeCount == 4
    assert result.elementCount == 1
    assert result.conductivity.isSparse
    assert result.capacity.isSparse
    np.testing.assert_allclose(result.elementVolumes, [1.0 / 6.0])
    np.testing.assert_allclose(result.conductivity.toNumpy(), expectedConductivity)
    np.testing.assert_allclose(result.capacity.toNumpy(), expectedCapacity)
    np.testing.assert_allclose(result.load.toNumpy(), expectedLoad)
    np.testing.assert_allclose(result.conductivity.toNumpy().sum(axis=1), 0.0, atol=1e-14)
    assert np.isclose(result.capacity.toNumpy().sum(), 3.0 * 4.0 / 6.0)
    assert np.isclose(result.load.toNumpy().sum(), 1.0)
    temperature = np.full((4, 1), 10.0)
    np.testing.assert_allclose(result.steadyResidual(temperature), -expectedLoad, atol=1e-14)
    np.testing.assert_allclose(
        result.transientResidual(temperature, np.zeros_like(temperature)),
        -expectedLoad,
        atol=1e-14,
    )


def test_linear_tetra_supports_anisotropic_conductivity_and_lumped_capacity():
    nodes, elements = unitTetra()
    conductivity = np.diag([2.0, 3.0, 4.0])
    result = assembleLinearTetraThermalOperators(
        nodes,
        elements,
        conductivity=conductivity,
        density=2.0,
        specificHeat=5.0,
        capacityMode='lumped',
    )

    expectedDiagonal = np.full(4, 2.0 * 5.0 / 6.0 / 4.0)
    np.testing.assert_allclose(result.capacity.toNumpy(), np.diag(expectedDiagonal))
    np.testing.assert_allclose(result.conductivity.toNumpy(), result.conductivity.toNumpy().T)


def test_convection_and_heat_flux_boundary_terms():
    nodes, elements = unitTetra()
    base = assembleLinearTetraThermalOperators(nodes, elements, conductivity=1.0)
    result = assembleLinearTetraThermalOperators(
        nodes,
        elements,
        conductivity=1.0,
        convectionFaces=np.array([[0, 1, 2]], dtype=int),
        convectionCoefficient=2.0,
        ambientTemperature=10.0,
        heatFluxFaces=np.array([[0, 1, 2]], dtype=int),
        heatFlux=6.0,
    )

    expectedBoundary = np.zeros((4, 4), dtype=float)
    expectedBoundary[:3, :3] = (1.0 / 12.0) * (np.ones((3, 3)) + np.eye(3))
    expectedLoad = np.zeros((4, 1), dtype=float)
    expectedLoad[:3, 0] = 10.0 / 3.0 + 1.0

    np.testing.assert_allclose(
        result.conductivity.toNumpy() - base.conductivity.toNumpy(),
        expectedBoundary,
    )
    np.testing.assert_allclose(result.load.toNumpy(), expectedLoad)


def test_nodal_convection_and_heat_flux_are_integrated_over_boundary_faces():
    nodes, elements = unitTetra()
    base = assembleLinearTetraThermalOperators(nodes, elements, conductivity=1.0)
    result = assembleLinearTetraThermalOperators(
        nodes,
        elements,
        conductivity=1.0,
        convectionFaces=np.array([[0, 1, 2]], dtype=int),
        convectionCoefficient=np.array([[1.0, 2.0, 3.0]]),
        ambientTemperature=10.0,
        heatFluxFaces=np.array([[0, 1, 2]], dtype=int),
        heatFlux=np.array([[3.0, 6.0, 9.0]]),
    )

    expectedBoundary = np.zeros((4, 4), dtype=float)
    expectedBoundary[:3, :3] = np.array(
        [
            [2.0 / 15.0, 3.0 / 40.0, 1.0 / 12.0],
            [3.0 / 40.0, 1.0 / 6.0, 11.0 / 120.0],
            [1.0 / 12.0, 11.0 / 120.0, 1.0 / 5.0],
        ]
    )
    expectedLoad = np.zeros((4, 1), dtype=float)
    expectedLoad[:3, 0] = 10.0 * expectedBoundary[:3, :3].sum(axis=1)
    expectedLoad[:3, 0] += [0.875, 1.0, 1.125]

    np.testing.assert_allclose(
        result.K.toNumpy() - base.K.toNumpy(),
        expectedBoundary,
    )
    np.testing.assert_allclose(result.Q.toNumpy(), expectedLoad)


def test_apply_dirichlet_conditions_preserves_symmetry_and_rhs_shift():
    matrix = csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]]))
    load = np.array([[0.0], [3.0]])

    constrained, shiftedLoad = applyDirichletConditions(
        matrixOperator(matrix, backendName='scipy'),
        load,
        [0],
        10.0,
    )

    assert issparse(constrained.toBackendData())
    np.testing.assert_allclose(constrained.toNumpy(), [[1.0, 0.0], [0.0, 2.0]])
    np.testing.assert_allclose(shiftedLoad.toNumpy(), [[10.0], [13.0]])


def test_apply_dirichlet_conditions_preserves_values_for_unsorted_nodes():
    matrix = csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]]))

    constrained, shiftedLoad = applyDirichletConditions(
        matrix,
        np.zeros((2, 1)),
        [1, 0],
        [20.0, 10.0],
    )

    np.testing.assert_allclose(constrained.toNumpy(), np.eye(2))
    np.testing.assert_allclose(shiftedLoad.toNumpy(), [[10.0], [20.0]])


def test_contact_interface_matrix_is_symmetric_and_energy_conserving():
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
    elements = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=int)
    base = assembleLinearTetraThermalOperators(nodes, elements, conductivity=1.0)
    result = assembleLinearTetraThermalOperators(
        nodes,
        elements,
        conductivity=1.0,
        contactPrimaryFaces=np.array([[0, 1, 2]]),
        contactSecondaryFaces=np.array([[4, 5, 6]]),
        contactCoefficient=12.0,
    )
    contact = result.K.toNumpy() - base.K.toNumpy()

    np.testing.assert_allclose(contact, contact.T)
    np.testing.assert_allclose(contact.sum(axis=1), 0.0, atol=1e-14)
    equalTemperature = np.full(8, 320.0)
    np.testing.assert_allclose(contact @ equalTemperature, 0.0, atol=1e-12)
    splitTemperature = np.array([330.0] * 4 + [300.0] * 4)
    assert float(splitTemperature @ contact @ splitTemperature) > 0.0
    np.testing.assert_allclose(result.metadata['contactAreas'], [0.5])
