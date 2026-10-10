import torch
import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix, csr_array, eye

from mor.backends import backendRegistry
from mor.operators import matrixOperator, lowRankOperator


@pytest.mark.parametrize('factory', [coo_matrix, csr_matrix, csr_array])
def test_scipy_sparse_conversion_diagonal_and_factorization(factory):
    backend = backendRegistry.get('scipy')
    data = factory(np.diag([1., 2., 3.]))
    op = matrixOperator(data, 'scipy')
    assert op.isSparse and op.isDiagonal
    np.testing.assert_array_equal(op.diagonal(), [1., 2., 3.])
    solve = backend.linalg.factorize(data)
    rhs = np.ones((3, 2)) * (1 + 2j)
    np.testing.assert_allclose(data @ solve(rhs), rhs)
    assert solve.factorBytes > 0


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('layout', ['coo', 'csr', 'csc', 'bsr', 'bsc'])
def test_torch_sparse_formats_diagonal_conversion_apply_and_transpose(device, layout):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    backend = backendRegistry.get('torch', device=device)
    dense = torch.diag(torch.tensor([1., 2., 3., 4.], dtype=torch.float64, device=device))
    convert = {'coo': lambda x: x.to_sparse(), 'csr': lambda x: x.to_sparse_csr(),
               'csc': lambda x: x.to_sparse_csc(), 'bsr': lambda x: x.to_sparse_bsr((2, 2)),
               'bsc': lambda x: x.to_sparse_bsc((2, 2))}[layout]
    op = matrixOperator(convert(dense), 'torch')
    assert op.isSparse and op.isDiagonal
    torch.testing.assert_close(op.diagonal(), dense.diagonal())
    rhs = torch.ones(4, dtype=torch.float64, device=device)
    torch.testing.assert_close(op.apply(rhs), dense @ rhs)
    torch.testing.assert_close(op.apply(np.ones(4)), dense @ rhs)
    torch.testing.assert_close(op.T.apply(rhs), dense.T @ rhs)
    identity = backend.array.eyeLike(op.toBackendData())
    assert identity.device == dense.device and backend.array.isSparse(identity)
    np.testing.assert_array_equal(op.toNumpy(), dense.cpu().numpy())
    scipyData = backendRegistry.get('scipy').array.toArray(op.toBackendData())
    assert isinstance(scipyData, csr_matrix)
    np.testing.assert_array_equal(scipyData.toarray(), dense.cpu().numpy())


def test_torch_diagonal_detection_sums_duplicates_and_handles_complex_without_host_conversion(monkeypatch):
    backend = backendRegistry.get('torch', device='cpu')
    data = torch.sparse_coo_tensor([[0, 1, 0, 0], [0, 1, 1, 1]],
                                    torch.tensor([1j, 2j, 1j, -1j]), (2, 2))
    monkeypatch.setattr(backend.array, 'toNumpy', lambda *args: pytest.fail('host conversion during detection'))
    assert backend.array.isDiagonal(data)
    torch.testing.assert_close(backend.array.diagonal(data), torch.tensor([1j, 2j]))
    assert backend.array.isDiagonal(torch.diag(torch.tensor([1j, 2j])))
    assert not backend.array.isDiagonal(torch.tensor([[1., 1.], [0., 1.]]))


def test_torch_sparse_identity_never_allocates_dense_square(monkeypatch):
    backend = backendRegistry.get('torch', device='cpu')
    monkeypatch.setattr(torch, 'eye', lambda *args, **kwargs: pytest.fail('dense identity allocated'))
    identity = backend.array.eyeSparse(100000, dtype=torch.float64)
    assert identity._nnz() == 100000


def test_scipy_sparse_qr_bounded_conversion_and_zero_vector():
    backend = backendRegistry.get('scipy')
    data = csr_matrix(np.array([[1., 2.], [0., 1.], [3., 0.]]))
    q = backend.decomposition.qrOrthogonalize(data, backend)
    np.testing.assert_allclose(q.T @ q, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(q @ q.T @ data.toarray(), data.toarray(), atol=1e-14)
    zero = backend.decomposition.qrOrthogonalize(csr_matrix((3, 1)), backend)
    assert np.isfinite(zero).all()
    with pytest.raises(ValueError, match='dense conversion limit'):
        backend.decomposition.qrOrthogonalize(eye(4096, format='csr'), backend)


def test_torch_unsupported_native_decompositions_report_explicitly():
    backend = backendRegistry.get('torch', device='cpu')
    with pytest.raises(NotImplementedError, match='Schur'):
        backend.linalg.schur(torch.eye(2))
    with pytest.raises(NotImplementedError, match='sparse SVD'):
        backend.decomposition.svdSparse(torch.eye(2).to_sparse(), 1)
    with pytest.raises(NotImplementedError, match='dense tensor'):
        backend.linalg.factorize(torch.eye(2).to_sparse_csr())
    with pytest.raises(NotImplementedError, match='direct solve'):
        backend.linalg.solve(torch.eye(2).to_sparse(), torch.ones(2))
    with pytest.raises(NotImplementedError, match='QR requires'):
        backend.decomposition.qrOrthogonalize(torch.eye(2).to_sparse(), backend)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_torch_dense_lu_and_shifted_solve_use_input_device(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from mor.algorithm.linear.shifted import shiftedLinear
    backend = backendRegistry.get('torch', device='cpu')
    data = torch.tensor([[2., 1.], [0., 3.]], dtype=torch.float64, device=device)
    rhs = torch.ones((2, 1), dtype=torch.complex128, device=device) * (1 + 2j)
    solve = backend.linalg.factorize(data)
    torch.testing.assert_close(data.to(rhs.dtype) @ solve(rhs), rhs)
    real = rhs.real
    solver = shiftedLinear('torch')
    result = solver.solve(matrixOperator(data, 'torch'), real, shift=0.3)
    expected = torch.linalg.solve(data + 0.3 * torch.eye(2, dtype=data.dtype, device=device), real)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(solver.factorize(matrixOperator(data, 'torch'), shift=0.3)(real), expected)


def test_lowrank_backend_promotes_factors_without_changing_device():
    op = lowRankOperator(torch.ones((3, 1), dtype=torch.float32),
                         torch.ones((2, 1), dtype=torch.float64), 'torch')
    assert op.dtype == torch.float64
    np.testing.assert_array_equal(op.toNumpy(), np.ones((3, 2)))


def test_torch_sparse_diagonal_weighted_svd_stays_sparse(monkeypatch):
    from mor.algorithm.registry import algorithmRegistry
    data = torch.tensor([[1., 2.], [0., 1.], [2., 1.]], dtype=torch.float64)
    weight = torch.diag(torch.tensor([1., 2., 3.], dtype=torch.float64)).to_sparse_csr()
    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, 'to_dense', lambda *args, **kwargs: pytest.fail('diagonal product densified'))
        u, s, vh = algorithmRegistry.get('svd', 'dual', backendName='torch').decompose(
            matrixOperator(data, 'torch'), product=matrixOperator(weight, 'torch'))
    torch.testing.assert_close((u * s) @ vh, data)
    torch.testing.assert_close(u.mT @ torch.diag(torch.tensor([1., 2., 3.], dtype=torch.float64)) @ u,
                               torch.eye(2, dtype=torch.float64))
