import numpy as np
import pytest
import torch
from scipy import linalg
from scipy.sparse import csr_matrix

from mor.assembly import thermalCouplingInterface
from mor.models import coupledThermalModel, thermalModel
from mor.operators import matrixOperator, thermalOperatorSet
from mor.reductors import thermalCouplingReductor


def componentModel(K, M, Q, backend='scipy', **kwargs):
    if backend == 'torch':
        values = [torch.tensor(value, dtype=torch.float64).to_sparse() if np.ndim(value) == 2 and np.shape(value)[1] > 1
                  else torch.tensor(value, dtype=torch.float64) for value in (K, M, Q)]
    else:
        values = [csr_matrix(K), csr_matrix(M), np.asarray(Q)]
    operators = thermalOperatorSet(*(matrixOperator(value, backend) for value in values), np.empty(0))
    return thermalModel(operators, backendName=backend, **kwargs)


def test_closed_two_body_exchange_matches_analytic_solution_and_conserves_energy():
    reductor = thermalCouplingReductor()
    components = [reductor.reduceComponent(name, componentModel([[0.]], [[capacity]], [0.]), basis=[[1.]])
                  for name, capacity in [('gear', 2.), ('oil', 3.)]]
    interface = thermalCouplingInterface.fromNodePairs('gear_oil', 'gear', 'oil', 1, 1, [0], [0], coefficient=4., areas=1.)
    model = reductor.assemble(components, [interface])
    times = np.linspace(0., 2., 2001)
    result = model.solveTransient({'gear': 350., 'oil': 300.}, times)
    equilibrium = 320.
    difference = 50. * np.exp(-4. * (1/2 + 1/3) * times)
    np.testing.assert_allclose(result.temperatures('gear')[:, 0], equilibrium + 3/5 * difference, atol=0.02)
    np.testing.assert_allclose(result.temperatures('oil')[:, 0], equilibrium - 2/5 * difference, atol=0.02)
    assert np.max(np.abs(result.diagnostics['residualW'])) < 1e-8
    np.testing.assert_allclose(2 * result.temperatures('gear')[:, 0] + 3 * result.temperatures('oil')[:, 0], 1600., atol=1e-9)
    assert result.interfacePowers()['gear_oil'][0] == pytest.approx(200.)
    with pytest.raises(ValueError, match='thermal sink'):
        model.solveSteady()


@pytest.mark.parametrize('backend', ['scipy', 'torch'])
def test_component_projection_and_nonmatching_contact_equal_monolithic_solution(backend):
    reductor = thermalCouplingReductor()
    k1, k2 = np.array([[3., -2.], [-2., 3.]]), np.array([[4., -1.], [-1., 2.]])
    m1, m2 = np.array([[2., .2], [.2, 3.]]), np.diag([4., 2.])
    q1, q2 = np.array([2., 1.]), np.array([0., 3.])
    left = reductor.reduceComponent('left', componentModel(k1, m1, q1, backend), basis=np.eye(2), offset=[310., 315.])
    right = reductor.reduceComponent('right', componentModel(k2, m2, q2, backend), basis=np.eye(2), offset=[290., 305.])
    rp, rs = np.array([[1., 0.], [.25, .75]]), np.array([[.5, .5], [0., 1.]])
    interface = thermalCouplingInterface('contact', 'left', 'right', rp, rs, [2., 3.])
    model = coupledThermalModel([left, right], [interface])
    trace = np.hstack((rp, -rs))
    fullK = linalg.block_diag(k1, k2) + trace.T @ np.diag([2., 3.]) @ trace
    fullM, fullQ = linalg.block_diag(m1, m2), np.r_[q1, q2]
    steady = model.reconstruct(model.solveSteady())
    np.testing.assert_allclose(np.r_[steady['left'], steady['right']], np.linalg.solve(fullK, fullQ), atol=1e-10)
    initial = np.array([320., 310., 305., 300.])
    result = model.solveTransient({'left': initial[:2], 'right': initial[2:]}, np.arange(5) * .1)
    expected = initial
    for step in range(1, 5):
        expected = np.linalg.solve(fullM + .1 * fullK, fullM @ expected + .1 * fullQ)
        np.testing.assert_allclose(np.r_[result.temperatures('left')[step], result.temperatures('right')[step]], expected, atol=1e-10)
    assert np.max(np.abs(result.diagnostics['residualW'])) < 1e-8


def test_dirichlet_lifting_and_fixed_reservoir_energy_balance():
    reductor = thermalCouplingReductor()
    solidModel = componentModel([[2., -2.], [-2., 2.]], [[2., .5], [.5, 2.]], [0., 4.], dirichletNodes=[0], dirichletValues=300.)
    solid = reductor.reduceComponent('solid', solidModel, basis=np.eye(2))
    oil = reductor.reduceComponent('oil', componentModel([[1.]], [[3.]], [295.]), basis=[[1.]])
    interface = thermalCouplingInterface('contact', 'solid', 'oil', [[.25, .75]], [[1.]], [2.])
    model = coupledThermalModel([solid, oil], [interface])
    result = model.solveTransient({'solid': [300., 310.], 'oil': [305.]}, [0., .1, .2, .3])
    np.testing.assert_allclose(result.temperatures('solid')[:, 0], 300.)
    trace = np.array([.25, .75, -1.])
    K = linalg.block_diag([[2., -2.], [-2., 2.]], [[1.]]) + 2 * np.outer(trace, trace)
    M, Q = linalg.block_diag([[2., .5], [.5, 2.]], [[3.]]), np.array([0., 4., 295.])
    previous = np.array([300., 310., 305.])
    for step in range(1, 4):
        rhs = M @ previous / .1 + Q
        matrix = M / .1 + K
        temperature = np.r_[300., np.linalg.solve(matrix[1:, 1:], rhs[1:] - matrix[1:, 0] * 300.)]
        np.testing.assert_allclose(np.r_[result.temperatures('solid')[step], result.temperatures('oil')[step]], temperature, atol=1e-10)
        previous = temperature
    assert np.max(np.abs(result.diagnostics['residualW'])) < 1e-8
    with pytest.raises(ValueError, match='fixed-temperature'):
        model.projectTemperatures({'solid': [301., 310.], 'oil': [305.]})


def test_pod_constant_enrichment_time_varying_input_cache_and_serialization(tmp_path):
    reductor = thermalCouplingReductor()
    full = componentModel([[2., -1., 0.], [-1., 2., -1.], [0., -1., 2.]], np.diag([1., 2., 3.]), [0., 0., 0.])
    snapshots = np.array([[300., 310., 320.], [300., 302., 304.], [300., 301., 302.]])
    component = reductor.reduceComponent('solid', full, snapshots=snapshots, rank=2,
                                          inputOperator=matrixOperator([[1.], [0.], [0.]], 'scipy'))
    assert component.order == 2
    np.testing.assert_allclose(component.basis @ component.basis.T @ np.ones(3), 1.)
    model = coupledThermalModel([component])
    times = np.arange(6, dtype=float) * .25
    inputs = lambda time: {'solid': [10 * time]}
    result = model.solveTransient({'solid': 300.}, times, inputs)
    assert model.lastSolveInfo['factorizationCount'] == 1
    assert np.max(np.abs(result.diagnostics['residualW'])) < 1e-8
    path = tmp_path / 'rom.npz'
    model.save(path)
    restored = coupledThermalModel.load(path)
    again = restored.solveTransient({'solid': 300.}, times, inputs)
    np.testing.assert_allclose(again.coordinates, result.coordinates)
    np.testing.assert_allclose(again.temperatures('solid'), result.temperatures('solid'))


def test_interface_serialization_nearest_mapping_and_conservation(tmp_path):
    reductor = thermalCouplingReductor()
    parts = [reductor.reduceComponent(name, componentModel(np.zeros((n, n)), np.eye(n), np.zeros(n)), basis=np.eye(n))
             for name, n in [('left', 2), ('right', 1)]]
    interface = thermalCouplingInterface.fromNearestNodes('contact', 'left', 'right', [[0., 0.], [.01, 0.]], [[0., 0.]],
                    [0, 1], [0], coefficient=[2., 4.], areas=[.5, .5], maxDistance=.02)
    model = coupledThermalModel(parts, [interface])
    path = tmp_path / 'coupled.npz'
    model.save(path)
    restored = coupledThermalModel.load(path)
    result = restored.solveTransient({'left': [310., 320.], 'right': 300.}, [0., .1, .2])
    np.testing.assert_allclose(result.interfacePowers()['contact'][0], 50.)
    assert np.max(np.abs(result.diagnostics['residualW'])) < 1e-9
    with pytest.raises(ValueError, match='maxDistance'):
        thermalCouplingInterface.fromNearestNodes('far', 'left', 'right', [[0., 0.]], [[1., 0.]],
                    [0], [0], coefficient=1., areas=1., maxDistance=.1)


@pytest.mark.parametrize('primary,secondary,g,match', [([[.5]], [[1.]], [1.], 'sum to one'),
    ([[1.]], [[1.]], [-1.], 'non-negative'), ([[1.]], [[1.], [1.]], [1.], 'quadrature')])
def test_invalid_interfaces_rejected(primary, secondary, g, match):
    with pytest.raises(ValueError, match=match):
        thermalCouplingInterface('interface', 'left', 'right', primary, secondary, g)


def test_reduction_rejects_nonlinear_singular_capacity_and_bad_rank():
    reductor = thermalCouplingReductor()
    full = componentModel(np.eye(2), np.eye(2), [0., 0.])
    full.operatorUpdater = lambda state: full.operators
    with pytest.raises(NotImplementedError, match='temperature-independent'):
        reductor.reduceComponent('bad', full, basis=np.eye(2))
    full.operatorUpdater = None
    with pytest.raises(ValueError, match='rank exceeds'):
        reductor.reduceComponent('bad', full, snapshots=np.eye(2), rank=3)
    zeroCapacity = componentModel(np.eye(2), np.zeros((2, 2)), [0., 0.])
    with pytest.raises(linalg.LinAlgError):
        reductor.reduceComponent('bad', zeroCapacity, basis=np.eye(2))


def test_projection_keeps_full_operators_sparse_and_accepts_composed_inputs(monkeypatch):
    reductor = thermalCouplingReductor()
    model = componentModel([[2., -1.], [-1., 2.]], np.eye(2), [0., 1.])
    model.K = model.K * .5 + model.K * .5
    monkeypatch.setattr(model.K, 'toNumpy', lambda: pytest.fail('full K materialized'))
    monkeypatch.setattr(model.M, 'toNumpy', lambda: pytest.fail('full M materialized'))
    reduced = reductor.reduceComponent('solid', model, basis=np.eye(2))
    np.testing.assert_allclose(reduced.capacity, np.eye(2), atol=1e-14)
    assert reduced.fullOrder == 2


def test_centered_pod_result_mean_survives_reduction():
    from mor.reductors import podReductor
    snapshots = np.array([[310., 320., 330.], [300., 305., 310.]])
    basis = podReductor('scipy', centerSnapshots=True).reduce(matrixOperator(snapshots, 'scipy'), rank=1)
    model = componentModel(np.eye(2), np.eye(2), [0., 0.])
    component = thermalCouplingReductor().reduceComponent('solid', model, basis=basis)
    np.testing.assert_allclose(component.offset, snapshots.mean(axis=1))
    np.testing.assert_allclose(component.reconstruct(component.project([315., 302.5])), [315., 302.5])


@pytest.mark.parametrize('times', [[0., 0.], [1., 0.], [0., np.nan], [[0., 1.]]])
def test_bad_time_grids_fail(times):
    part = thermalCouplingReductor().reduceComponent('solid', componentModel([[1.]], [[1.]], [0.]), basis=[[1.]])
    with pytest.raises(ValueError, match='times'):
        coupledThermalModel([part]).solveTransient({'solid': 300.}, times)


def test_unknown_endpoints_inputs_and_duplicate_names_rejected():
    part = thermalCouplingReductor().reduceComponent('solid', componentModel([[1.]], [[1.]], [0.]), basis=[[1.]])
    with pytest.raises(ValueError, match='unique names'):
        coupledThermalModel([part, part])
    interface = thermalCouplingInterface('missing', 'solid', 'unknown', [[1.]], [[1.]], [1.])
    with pytest.raises(ValueError, match='unknown component'):
        coupledThermalModel([part], [interface])
    model = coupledThermalModel([part])
    with pytest.raises(ValueError, match='known component'):
        model.implicitEulerStep([0.], .1, inputs={'unknown': [1.]})
    with pytest.raises(ValueError, match='input size'):
        model.implicitEulerStep([0.], .1, inputs={'solid': [1.]})


def test_tetrahedral_fem_components_couple_without_full_matrix_conversion():
    from mor.assembly import assembleLinearTetraThermalOperators
    nodes = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    operators = assembleLinearTetraThermalOperators(nodes, [[0, 1, 2, 3]], conductivity=2., density=3., specificHeat=4.)
    reductor = thermalCouplingReductor()
    parts = [reductor.reduceComponent(name, thermalModel(operators), basis=np.eye(4)) for name in ['left', 'right']]
    trace = [[1/3, 1/3, 1/3, 0.]]
    interface = thermalCouplingInterface('surface', 'left', 'right', trace, trace, [5.])
    model = coupledThermalModel(parts, [interface])
    result = model.solveTransient({'left': 350., 'right': 300.}, [0., .1, .2])
    assert result.interfacePowers()['surface'][0] == pytest.approx(250.)
    assert np.max(np.abs(result.diagnostics['residualW'])) < 1e-8


def test_coupling_reductor_is_available_through_registry():
    from mor.reductors import reductorRegistry
    assert isinstance(reductorRegistry.get('thermalCoupling'), thermalCouplingReductor)
