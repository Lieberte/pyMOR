import numpy as np
from scipy.sparse import diags, eye, kron


def matlab_gallery_hilbert(n):
    """MATLAB gallery-style Hilbert matrix: dense, ill-conditioned."""
    i = np.arange(1, n + 1, dtype=float)
    return 1.0 / (i[:, None] + i[None, :] - 1.0)


def matlab_gallery_grcar(n, k=3):
    """MATLAB gallery-style Grcar matrix: dense, nonsymmetric, nonnormal."""
    A = np.zeros((n, n), dtype=float)
    for offset in range(0, k + 1):
        A += np.diag(np.ones(n - offset), k=offset)
    A += np.diag(-np.ones(n - 1), k=-1)
    return A


def matlab_gallery_grcar_rectangular(rows, cols, k=3):
    """Rectangular Grcar-like matrix without materializing a full square matrix."""
    A = np.zeros((rows, cols), dtype=float)
    for offset in range(0, k + 1):
        length = min(rows, cols - offset) if cols - offset > 0 else 0
        if length > 0:
            idx = np.arange(length)
            A[idx, idx + offset] = 1.0
    length = min(rows - 1, cols)
    if length > 0:
        idx = np.arange(length)
        A[idx + 1, idx] = -1.0
    return A


def matrix_depot_poisson_2d(m):
    """MatrixDepot-style 2D Poisson stencil: sparse SPD matrix."""
    T = diags([-np.ones(m - 1), 4.0 * np.ones(m), -np.ones(m - 1)], [-1, 0, 1], format='csr')
    I = eye(m, format='csr')
    off = diags([-np.ones(m)], [0], format='csr')
    return kron(I, T, format='csr') + kron(off, I, format='csr') + kron(off, I, format='csr').T


def matrix_depot_poisson_1d(n):
    """MatrixDepot-style 1D Poisson stencil: sparse SPD tridiagonal."""
    return diags([-np.ones(n - 1), 2.0 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1], format='csr')


def ill_conditioned_diagonal(n, min_exp: float = -12.0):
    """Diagonal matrix with a controlled dynamic range."""
    exponents = np.linspace(0.0, min_exp, n)
    return np.diag(np.power(10.0, exponents))


def singular_diagonal(n, zero_columns: int = 1):
    """Exactly singular diagonal matrix with a trailing zero block."""
    values = np.ones(n, dtype=float)
    values[-zero_columns:] = 0.0
    return np.diag(values)


def geometric_spectrum_matrix(rows, cols, min_exp: float = -16.0, seed: int = 0):
    """Dense matrix with prescribed geometrically decaying singular values."""
    rng = np.random.default_rng(seed)
    left, _ = np.linalg.qr(rng.standard_normal((rows, cols)))
    right, _ = np.linalg.qr(rng.standard_normal((cols, cols)))
    singular_values = np.power(10.0, np.linspace(0.0, min_exp, cols))
    return left @ np.diag(singular_values) @ right.T


def rank_deficient_tall_skinny(rows, cols, effective_rank: int, seed: int = 0):
    """Tall-skinny matrix with exact duplicated-column rank deficiency."""
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((rows, effective_rank))
    coeffs = rng.standard_normal((effective_rank, cols))
    coeffs[:, effective_rank:] = coeffs[:, :1]
    return basis @ coeffs


def nearly_collinear_snapshots(rows, cols, noise_scale: float = 1e-12, seed: int = 0):
    """Snapshot matrix whose columns are almost linearly dependent."""
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal((rows, 1))
    weights = np.linspace(1.0, 2.0, cols)[None, :]
    noise = noise_scale * rng.standard_normal((rows, cols))
    return direction @ weights + noise


def extreme_scaled_diagonal(n, max_exp: float = 150.0):
    """Diagonal matrix spanning values close to double precision stress limits."""
    exponents = np.linspace(max_exp, -max_exp, n)
    return np.diag(np.power(10.0, exponents))


STANDARD_BENCHMARK_SOURCES = {
    'matlab_gallery': {
        'url': 'https://www.mathworks.com/help/matlab/ref/gallery.html',
        'cases': ['hilbert', 'grcar'],
    },
    'julia_matrix_depot': {
        'url': 'https://matrixdepotjl.readthedocs.io/en/latest/',
        'cases': ['poisson_1d', 'poisson_2d'],
    },
    'suitesparse_matrix_collection': {
        'url': 'https://sparse.tamu.edu/',
        'cases': ['external sparse matrices, optional local downloads'],
    },
    'pathological': {
        'url': 'local-generated',
        'cases': [
            'ill_conditioned_diagonal',
            'singular_diagonal',
            'geometric_spectrum_matrix',
            'rank_deficient_tall_skinny',
            'nearly_collinear_snapshots',
            'extreme_scaled_diagonal',
        ],
    },
}
