import pytest
import numpy as np
from scipy import sparse

from mor.algorithm.decompose.svd import staticSVD
from mor.operators import matrixOperator


class TestStaticSVD:

    @pytest.fixture
    def dense_matrix(self):
        """生成稠密矩阵"""
        np.random.seed(42)
        return np.random.randn(10, 5)

    @pytest.fixture
    def low_rank_matrix(self):
        """生成低秩矩阵（rank=3）用于精确测试"""
        np.random.seed(42)
        U = np.random.randn(10, 3)
        S = np.diag([5.0, 3.0, 1.0])
        Vt = np.random.randn(3, 5)
        return U @ S @ Vt

    @pytest.fixture
    def sparse_matrix(self):
        """生成稀疏矩阵"""
        np.random.seed(42)
        data = np.random.randn(10, 5)
        data[data < 0.5] = 0
        return sparse.csr_matrix(data)

    @pytest.fixture
    def svd_algo(self):
        """创建 SVD 算法实例"""
        return staticSVD(backendName='numpy')

    # ========== 基础功能测试 ==========

    def test_decompose_dense_full(self, svd_algo, dense_matrix):
        """测试稠密矩阵完整分解"""
        operator = matrixOperator(dense_matrix)
        U, S, Vt = svd_algo.decompose(operator)

        assert U.shape == (10, 5)
        assert S.shape == (5,)
        assert Vt.shape == (5, 5)

        reconstructed = U @ np.diag(S) @ Vt
        np.testing.assert_allclose(reconstructed, dense_matrix, rtol=1e-10)

    def test_decompose_dense_with_rank(self, svd_algo, low_rank_matrix):
        """测试低秩矩阵指定 rank 分解（精确重构）"""
        operator = matrixOperator(low_rank_matrix)
        rank = 3
        U, S, Vt = svd_algo.decompose(operator, rank=rank)

        assert U.shape == (10, rank)
        assert S.shape == (rank,)
        assert Vt.shape == (rank, 5)

        # 低秩矩阵应该能精确重构
        reconstructed = U @ np.diag(S) @ Vt
        np.testing.assert_allclose(reconstructed, low_rank_matrix, atol=1e-10)

    def test_decompose_sparse(self, svd_algo, sparse_matrix):
        """测试稀疏矩阵分解"""
        operator = matrixOperator(sparse_matrix)
        rank = 3
        U, S, Vt = svd_algo.decompose(operator, rank=rank)

        assert U.shape == (10, rank)
        assert S.shape == (rank,)
        assert Vt.shape == (rank, 5)

    # ========== 边界条件测试 ==========

    def test_rank_larger_than_matrix(self, svd_algo, dense_matrix):
        """测试 rank 大于矩阵秩"""
        operator = matrixOperator(dense_matrix)
        rank = 100
        U, S, Vt = svd_algo.decompose(operator, rank=rank)

        assert U.shape[1] == min(dense_matrix.shape)

    def test_rank_zero(self, svd_algo, dense_matrix):
        """测试 rank=0"""
        operator = matrixOperator(dense_matrix)
        U, S, Vt = svd_algo.decompose(operator, rank=0)

        assert U.shape[1] == 0
        assert len(S) == 0
        assert Vt.shape[0] == 0

    def test_full_matrices_flag(self, svd_algo, dense_matrix):
        """测试 fullMatrices 参数"""
        operator = matrixOperator(dense_matrix)

        U1, S1, Vt1 = svd_algo.decompose(operator, fullMatrices=False)
        assert U1.shape == (10, 5)

        U2, S2, Vt2 = svd_algo.decompose(operator, fullMatrices=True)
        assert U2.shape == (10, 10)
        assert Vt2.shape == (5, 5)

    # ========== 数值精度测试 ==========

    def test_singular_values_sorted(self, svd_algo, dense_matrix):
        """测试奇异值是否降序排列"""
        operator = matrixOperator(dense_matrix)
        _, S, _ = svd_algo.decompose(operator)

        assert np.all(S[:-1] >= S[1:])

    def test_orthogonality(self, svd_algo, dense_matrix):
        """测试 U 和 V 的正交性"""
        operator = matrixOperator(dense_matrix)
        U, _, Vt = svd_algo.decompose(operator)

        UtU = U.T @ U
        np.testing.assert_allclose(UtU, np.eye(U.shape[1]), atol=1e-10)

        VVt = Vt.T @ Vt
        np.testing.assert_allclose(VVt, np.eye(Vt.shape[0]), atol=1e-10)

    # ========== 不同 Backend 测试（修复版）==========

    @pytest.mark.parametrize("backend", ['numpy', 'scipy'])
    def test_different_backends(self, low_rank_matrix, backend):
        """测试不同 Backend 的数值一致性"""
        try:
            algo = staticSVD(backendName=backend)
            operator = matrixOperator(low_rank_matrix)
            U, S, Vt = algo.decompose(operator, rank=3)

            # 验证形状
            assert U.shape == (10, 3)
            assert S.shape == (3,)
            assert Vt.shape == (3, 5)

            # 验证低秩矩阵精确重构
            reconstructed = U @ np.diag(S) @ Vt
            error = np.linalg.norm(low_rank_matrix - reconstructed, 'fro')
            assert error < 1e-8  # 数值精度误差

            # 验证正交性
            np.testing.assert_allclose(U.T @ U, np.eye(3), atol=1e-10)

        except ImportError:
            pytest.skip(f"{backend} backend not available")

    # ========== 性能测试 ==========

    def test_large_matrix_performance(self, svd_algo):
        """测试大矩阵性能"""
        large_matrix = np.random.randn(1000, 500)
        operator = matrixOperator(large_matrix)

        import time
        start = time.time()
        U, S, Vt = svd_algo.decompose(operator, rank=50)
        elapsed = time.time() - start

        assert elapsed < 5.0
        assert U.shape == (1000, 50)