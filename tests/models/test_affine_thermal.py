import numpy as np
import pytest
from scipy import linalg, sparse

from mor.models import affineThermalModel
from mor.operators import matrixOperator
from mor.reductors import affineThermalReductor


@pytest.mark.parametrize('operatorForm', ['dense', 'sparse', 'operator'])
def test_reloaded_affine_rom_matches_full_equations_with_dynamic_htc_and_interface_outputs(tmp_path, operatorForm):
    mass = np.array([[2., .2, 0.], [.2, 3., .1], [0., .1, 4.]])
    bulk = np.array([[2., -2., 0.], [-2., 5., -3.], [0., -3., 3.]])
    boundaries = [np.diag([4., 0., 0.]), np.diag([0., 0., 6.])]
    loads = np.column_stack([matrix @ np.ones(3) for matrix in boundaries])
    source = np.array([0., 8., 0.])
    basis, _ = linalg.qr(np.column_stack((np.ones(3), np.eye(3)[:, :2])))
    convert = {'dense': lambda x: x, 'sparse': sparse.csr_matrix,
               'operator': lambda x: matrixOperator(sparse.csr_matrix(x), 'scipy')}[operatorForm]
    model = affineThermalReductor().reduce(convert(mass), convert(bulk), [convert(x) for x in boundaries],
                                           source, loads, basis, referenceTemperature=300., interfaceNames=['left', 'right'])
    path = tmp_path / 'rom.npz'
    model.save(path)
    del model
    model = affineThermalModel.load(path)
    assert model.interfaceNames == ('left', 'right')
    scales = np.array([[1., 1.], [1., 1.], [.5, 2.], [.5, 2.], [0., 0.]])
    refs = np.array([[310., 295.], [315., 290.], [301., 299.], [320., 298.], [300., 300.]])
    heat, dt = np.array([1., .5, 2., 0., 1.]), .2
    result = model.solveTransient(dt, heat, scales, refs, initialRise=2.)
    states = [np.full(3, 2.)]
    expectedPower, expectedStepPower = [], []
    for h, ref, q in zip(scales, refs, heat):
        k = bulk + sum(scale * matrix for scale, matrix in zip(h, boundaries))
        old = states[-1]
        current = linalg.solve(mass / dt + k / 2,
                               (mass / dt - k / 2) @ old + source * q + loads @ (h * (ref - 300.)))
        states.append(current)
        expectedPower.append(h * (loads.T @ current - loads.sum(axis=0) * (ref - 300.)))
        expectedStepPower.append(h * (loads.T @ ((old + current) / 2) - loads.sum(axis=0) * (ref - 300.)))
    states = np.array(states)
    np.testing.assert_allclose(result.coordinates @ basis.T, states, atol=1e-12)
    np.testing.assert_allclose(result.interfaceTemperatures, states @ loads / loads.sum(axis=0) + 300., atol=1e-12)
    np.testing.assert_allclose(result.interfacePowers, expectedPower, atol=1e-12)
    np.testing.assert_allclose(result.stepInterfacePowers, expectedStepPower, atol=1e-12)
    assert max(abs(result.energyResidualW)) < 1e-10
    assert result.factorizationCount == 3
    with np.load(path, allow_pickle=False) as data:
        assert 'basis' not in data and 'nodes' not in data
    with pytest.raises(ValueError, match='nonnegative'):
        model.solveTransient(dt, heat, -scales, refs)


def test_isolated_solid_preserves_uniform_temperature_and_reuses_factorization():
    model = affineThermalReductor().reduce(np.diag([2., 3.]), np.array([[4., -4.], [-4., 4.]]),
        [np.eye(2)], np.zeros(2), np.ones((2, 1)), np.ones((2, 1)) / np.sqrt(2), referenceTemperature=300.)
    result = model.solveTransient(.1, np.zeros(20), np.zeros((20, 1)), np.full((20, 1), 290.), initialRise=7.)
    np.testing.assert_allclose(result.meanTemperature, 307., atol=1e-12)
    np.testing.assert_allclose(result.interfacePowers, 0.)
    assert result.factorizationCount == 1
    with pytest.raises(ValueError, match='finite'):
        model.solveTransient(.1, [np.nan], [[1.]], [[300.]])


def test_rejects_nonconservative_lifting_or_robin_terms():
    reductor = affineThermalReductor()
    with pytest.raises(ValueError, match='constant'):
        reductor.reduce(np.eye(2), np.eye(2), [np.eye(2)], np.ones(2), np.ones((2, 1)), [[1.], [0.]])
    with pytest.raises(ValueError, match='bulk conduction'):
        reductor.reduce(np.eye(2), np.eye(2), [np.eye(2)], np.ones(2), np.ones((2, 1)), np.eye(2))
    with pytest.raises(ValueError, match='Robin'):
        reductor.reduce(np.eye(2), np.zeros((2, 2)), [np.eye(2)], np.ones(2), 2 * np.ones((2, 1)), np.eye(2))
