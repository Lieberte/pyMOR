import torch
import numpy as np
import pytest

from mor.operators import assembleLinearTetraThermalOperators, applyDirichletConditions


def mesh():
    tetra = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    return np.vstack((tetra, tetra + [2., 0., 0.])), np.array([[0, 1, 2, 3], [4, 5, 6, 7]])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('capacityMode', ['consistent', 'lumped'])
def test_torch_sparse_assembly_boundaries_dirichlet_and_residual_match_scipy(device, capacityMode, monkeypatch):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    nodes, elements = mesh()
    options = dict(conductivity=np.diag([2., 3., 4.]), density=3., specificHeat=4., heatSource=[5., 6.],
                   capacityMode=capacityMode, assemblyBatchSize=1, convectionFaces=[[0, 1, 2]],
                   convectionCoefficient=[[1., 2., 3.]], ambientTemperature=300.,
                   heatFluxFaces=[[4, 5, 6]], heatFlux=7.,
                   contactPrimaryFaces=[[0, 1, 2]], contactSecondaryFaces=[[4, 5, 6]], contactCoefficient=2.)
    expected = assembleLinearTetraThermalOperators(nodes, elements, **options)
    expectedK, expectedLoad = applyDirichletConditions(expected.K, expected.Q, [0, 4], [310., 315.])
    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, 'to_dense', lambda *args, **kwargs: pytest.fail('global sparse matrix densified'))
        actual = assembleLinearTetraThermalOperators(torch.tensor(nodes, device=device),
            torch.tensor(elements, device=device), **options, backendName='torch', deviceName=device)
        assert actual.K.isSparse and actual.M.isSparse
        assert actual.K.toBackendData().device.type == device
        assert actual.Q.toBackendData().device.type == device
        assert actual.K.dtype == torch.float64
        actualK, actualLoad = applyDirichletConditions(actual.K, actual.Q, [0, 4], [310., 315.])
        assert actualK.isSparse
        state = torch.arange(8., dtype=torch.float64, device=device).reshape(-1, 1)
        residual = actual.transientResidual(state, state / 10)
    np.testing.assert_allclose(actual.K.toNumpy(), expected.K.toNumpy(), atol=1e-14)
    np.testing.assert_allclose(actual.M.toNumpy(), expected.M.toNumpy(), atol=1e-14)
    np.testing.assert_allclose(actual.Q.toNumpy(), expected.Q.toNumpy(), atol=1e-12)
    np.testing.assert_allclose(actualK.toNumpy(), expectedK.toNumpy(), atol=1e-14)
    np.testing.assert_allclose(actualLoad.toNumpy(), expectedLoad.toNumpy(), atol=1e-12)
    temperature = np.linalg.solve(actualK.toNumpy(), actualLoad.toNumpy())
    np.testing.assert_allclose(temperature[[0, 4], 0], [310., 315.])
    np.testing.assert_allclose(residual.cpu().numpy(), expected.transientResidual(state.cpu().numpy(), state.cpu().numpy()/10), atol=1e-12)


@pytest.mark.parametrize('layout', ['dense', 'coo', 'csr', 'csc'])
def test_torch_complex_dirichlet_preserves_dtype_and_does_not_mutate_inputs(layout):
    from mor.operators import matrixOperator
    dense = torch.tensor([[3., 1.], [1., 2.]], dtype=torch.complex128)
    data = {'dense': lambda: dense, 'coo': dense.to_sparse, 'csr': dense.to_sparse_csr,
            'csc': dense.to_sparse_csc}[layout]()
    rhs = torch.tensor([[1j], [2j]], dtype=torch.complex128)
    constrained, load = applyDirichletConditions(matrixOperator(data, 'torch'), matrixOperator(rhs, 'torch'), [0], 2j)
    np.testing.assert_array_equal(constrained.toNumpy(), [[1., 0.], [0., 2.]])
    np.testing.assert_array_equal(load.toNumpy(), [[2j], [0j]])
    torch.testing.assert_close(rhs, torch.tensor([[1j], [2j]], dtype=torch.complex128))
    np.testing.assert_array_equal(matrixOperator(data, 'torch').toNumpy(), [[3., 1.], [1., 2.]])


@pytest.mark.parametrize('indices,values,match', [([0.5], [1.], 'integers'), ([0], [np.nan], 'finite'),
                                                ([0, 0], [1., 2.], 'conflicting')])
def test_dirichlet_input_validation(indices, values, match):
    with pytest.raises(ValueError, match=match):
        applyDirichletConditions(np.eye(2), np.ones((2, 1)), indices, values)


def test_assembly_rejects_nonfinite_materials_before_sparse_construction():
    nodes, elements = mesh()
    with pytest.raises(ValueError, match='density.*finite'):
        assembleLinearTetraThermalOperators(nodes, elements, conductivity=1., density=np.nan)


def test_scipy_dirichlet_does_not_discard_complex_coefficients():
    data = np.array([[3., 1j], [-1j, 2.]], dtype=complex)
    constrained, load = applyDirichletConditions(data, np.ones((2, 1)), [0], 2.)
    np.testing.assert_array_equal(constrained.toNumpy(), [[1., 0.], [0., 2.]])
    np.testing.assert_array_equal(load.toNumpy(), [[2.], [1. + 2j]])


def test_public_assembly_entry_is_compatible_with_operator_exports():
    from mor.assembly import assembleLinearTetraThermalOperators as implementation
    from mor.assembly import applyDirichletConditions as constrain
    assert implementation is assembleLinearTetraThermalOperators
    assert constrain is applyDirichletConditions
