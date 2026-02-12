"""
staticSVD 性能和正确性测试
测试工厂模式和直接实例化两种方式
"""

import sys
import os
import numpy as np
import pytest
import time

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from temp.utils.svd import svdFactory, svdMethod
from temp.utils.svd import staticSVD


class TestStaticSVDCorrectness:
    """测试 staticSVD 的正确性"""

    def setup_method(self):
        """每个测试前的准备"""
        np.random.seed(42)
        self.small_matrix = np.random.randn(10, 8)
        self.medium_matrix = np.random.randn(100, 50)
        self.tall_matrix = np.random.randn(200, 50)
        self.wide_matrix = np.random.randn(50, 200)

    def test_factory_creation(self):
        """测试工厂模式创建"""
        svd = svdFactory.create(svdMethod.static)
        assert svd is not None
        assert isinstance(svd, staticSVD)
        assert svd.method == svdMethod.static

    def test_direct_creation(self):
        """测试直接实例化创建"""
        svd = staticSVD(method=svdMethod.static)
        assert svd is not None
        assert isinstance(svd, staticSVD)
        assert svd.method == svdMethod.static

    def test_basic_decomposition(self):
        """测试基本 SVD 分解"""
        A = self.medium_matrix
        svd = svdFactory.create(svdMethod.static)
        svd.fit(A, r=30)

        # 检查维度
        assert svd.U.shape == (100, 30), f"U shape wrong: {svd.U.shape}"
        assert svd.S.shape == (30,), f"S shape wrong: {svd.S.shape}"
        assert svd.Vt.shape == (30, 50), f"Vt shape wrong: {svd.Vt.shape}"

        # 检查奇异值递减
        assert np.all(svd.S[:-1] >= svd.S[1:]), "Singular values should be decreasing"

        # 检查奇异值非负
        assert np.all(svd.S >= 0), "Singular values should be non-negative"

    def test_full_rank_decomposition(self):
        """测试全秩分解（r = min(m, n)）"""
        A = self.small_matrix  # 10x8
        svd = svdFactory.create(svdMethod.static)
        svd.fit(A, r=8)

        assert svd.U.shape == (10, 8)
        assert svd.S.shape == (8,)
        assert svd.Vt.shape == (8, 8)

    def test_reconstruction_full_rank(self):
        """测试全秩重构（应该几乎完美）"""
        A = self.small_matrix
        svd = svdFactory.create(svdMethod.static)
        svd.fit(A, r=8)

        A_reconstructed = svd.reconstruct()

        # 检查维度
        assert A_reconstructed.shape == A.shape

        # 检查重构误差（应该非常小）
        error = np.linalg.norm(A - A_reconstructed, 'fro')
        relative_error = error / np.linalg.norm(A, 'fro')

        print(f"\n全秩重构相对误差: {relative_error:.2e}")
        assert relative_error < 1e-10, f"Reconstruction error too large: {relative_error}"

    def test_reconstruction_low_rank(self):
        """测试低秩重构"""
        A = self.medium_matrix
        svd = svdFactory.create(svdMethod.static)

        # 使用不同的秩进行测试
        ranks = [5, 10, 20, 30]
        errors = []

        for r in ranks:
            svd.fit(A, r=r)
            A_reconstructed = svd.reconstruct()
            error = np.linalg.norm(A - A_reconstructed, 'fro')
            errors.append(error)
            print(f"Rank {r:2d}: Reconstruction error = {error:.4f}")

        # 检查误差随秩增加而减小
        assert all(errors[i] >= errors[i+1] for i in range(len(errors)-1)), \
            "Error should decrease as rank increases"

    def test_orthogonality_U(self):
        """测试 U 的正交性: U^T U = I"""
        A = self.medium_matrix
        svd = svdFactory.create(svdMethod.static)
        svd.fit(A, r=30)

        UtU = svd.U.T @ svd.U
        I = np.eye(30)

        error = np.linalg.norm(UtU - I, 'fro')
        print(f"\nU 正交性误差: {error:.2e}")
        assert error < 1e-10, f"U is not orthogonal: {error}"

    def test_orthogonality_V(self):
        """测试 V 的正交性: V^T V = I"""
        A = self.medium_matrix
        svd = svdFactory.create(svdMethod.static)
        svd.fit(A, r=30)

        VVt = svd.Vt @ svd.Vt.T
        I = np.eye(30)

        error = np.linalg.norm(VVt - I, 'fro')
        print(f"\nV 正交性误差: {error:.2e}")
        assert error < 1e-10, f"V is not orthogonal: {error}"

    def test_tall_matrix(self):
        """测试高瘦矩阵 (m >> n)"""
        A = self.tall_matrix  # 200x50
        svd = svdFactory.create(svdMethod.static)
        svd.fit(A, r=30)

        assert svd.U.shape == (200, 30)
        assert svd.S.shape == (30,)
        assert svd.Vt.shape == (30, 50)

        A_reconstructed = svd.reconstruct()
        assert A_reconstructed.shape == A.shape

    def test_wide_matrix(self):
        """测试矮胖矩阵 (m << n)"""
        A = self.wide_matrix  # 50x200
        svd = svdFactory.create(svdMethod.static)
        svd.fit(A, r=30)

        assert svd.U.shape == (50, 30)
        assert svd.S.shape == (30,)
        assert svd.Vt.shape == (30, 200)

        A_reconstructed = svd.reconstruct()
        assert A_reconstructed.shape == A.shape

    def test_fullMatrices_parameter(self):
        """测试 fullMatrices 参数"""
        A = self.small_matrix  # 10x8

        # fullMatrices=False (默认)
        svd1 = svdFactory.create(svdMethod.static, fullMatrices=False)
        svd1.fit(A, r=5)

        # fullMatrices=True
        svd2 = svdFactory.create(svdMethod.static, fullMatrices=True)
        svd2.fit(A, r=5)

        # 两者的前 r 个分量应该相同
        assert np.allclose(svd1.U, svd2.U[:, :5])
        assert np.allclose(svd1.S, svd2.S[:5])
        assert np.allclose(svd1.Vt, svd2.Vt[:5, :])

    def test_singular_values_magnitude(self):
        """测试奇异值的量级"""
        A = self.medium_matrix
        svd = svdFactory.create(svdMethod.static)
        svd.fit(A, r=30)

        # 最大奇异值应该接近矩阵的谱范数
        matrix_norm = np.linalg.norm(A, 2)
        max_singular_value = svd.S[0]

        print(f"\n矩阵谱范数: {matrix_norm:.4f}")
        print(f"最大奇异值: {max_singular_value:.4f}")

        assert np.abs(max_singular_value - matrix_norm) < 1e-10

    def test_rank_deficient_matrix(self):
        """测试秩亏矩阵"""
        # 创建一个秩为 5 的矩阵
        U_true = np.random.randn(50, 5)
        S_true = np.diag(np.array([10, 8, 6, 4, 2]))
        Vt_true = np.random.randn(5, 30)
        A = U_true @ S_true @ Vt_true

        svd = svdFactory.create(svdMethod.static)
        svd.fit(A, r=10)

        # 前 5 个奇异值应该显著大于后面的
        print(f"\n奇异值: {svd.S}")
        assert svd.S[4] > 1.0  # 第5个奇异值应该还比较大
        assert svd.S[5] < 1e-10  # 第6个奇异值应该接近0


class TestStaticSVDPerformance:
    """测试 staticSVD 的性能"""

    def setup_method(self):
        """每个测试前的准备"""
        np.random.seed(42)

    def test_factory_vs_direct_creation_speed(self):
        """对比工厂模式和直接实例化的速度"""
        iterations = 1000

        # 测试工厂模式
        start = time.perf_counter()
        for _ in range(iterations):
            svd = svdFactory.create(svdMethod.static)
        factory_time = time.perf_counter() - start

        # 测试直接实例化
        start = time.perf_counter()
        for _ in range(iterations):
            svd = staticSVD(method=svdMethod.static)
        direct_time = time.perf_counter() - start

        print(f"\n创建 {iterations} 个实例:")
        print(f"  工厂模式:     {factory_time*1000:.3f} ms ({factory_time/iterations*1e6:.2f} μs/次)")
        print(f"  直接实例化:   {direct_time*1000:.3f} ms ({direct_time/iterations*1e6:.2f} μs/次)")
        print(f"  差异:        {abs(factory_time - direct_time)*1000:.3f} ms")

        # 性能差异应该很小
        assert abs(factory_time - direct_time) < 0.01, "创建方式性能差异过大"

    def test_computation_performance(self):
        """测试不同矩阵大小的计算性能"""
        test_cases = [
            (100, 50, 20, "小矩阵"),
            (500, 300, 50, "中矩阵"),
            (1000, 500, 100, "大矩阵"),
        ]

        print("\n" + "="*60)
        print("SVD 计算性能测试")
        print("="*60)

        for m, n, r, desc in test_cases:
            A = np.random.randn(m, n)

            # 工厂模式
            svd1 = svdFactory.create(svdMethod.static)
            start1 = time.perf_counter()
            svd1.fit(A, r=r)
            time1 = time.perf_counter() - start1

            # 直接实例化
            svd2 = staticSVD(method=svdMethod.static)
            start2 = time.perf_counter()
            svd2.fit(A, r=r)
            time2 = time.perf_counter() - start2

            print(f"\n{desc} ({m}×{n}, r={r}):")
            print(f"  工厂模式:     {time1*1000:.2f} ms")
            print(f"  直接实例化:   {time2*1000:.2f} ms")
            print(f"  差异:        {abs(time1-time2)*1000:.3f} ms ({abs(time1-time2)/time1*100:.4f}%)")


            assert abs(time1 - time2) < 0.1, f"时间差异过大: {abs(time1-time2)*1000:.2f} ms"

    @pytest.mark.slow
    def test_large_matrix_performance(self):
        """测试大矩阵性能（标记为慢速测试）"""
        A = np.random.randn(2000, 1000)

        svd = svdFactory.create(svdMethod.static)

        start = time.perf_counter()
        svd.fit(A, r=100)
        elapsed = time.perf_counter() - start

        print(f"\n大矩阵 (2000×1000, r=100):")
        print(f"  计算时间: {elapsed:.2f} 秒")

        # 确保在合理时间内完成（根据你的机器调整）
        assert elapsed < 10.0, f"计算时间过长: {elapsed:.2f}s"


class TestStaticSVDEdgeCases:
    """测试边界情况"""

    def test_single_column_matrix(self):
        """测试单列矩阵"""
        A = np.random.randn(100, 1)
        svd = svdFactory.create(svdMethod.static)
        svd.fit(A, r=1)

        assert svd.U.shape == (100, 1)
        assert svd.S.shape == (1,)
        assert svd.Vt.shape == (1, 1)

    def test_single_row_matrix(self):
        """测试单行矩阵"""
        A = np.random.randn(1, 100)
        svd = svdFactory.create(svdMethod.static)
        svd.fit(A, r=1)

        assert svd.U.shape == (1, 1)
        assert svd.S.shape == (1,)
        assert svd.Vt.shape == (1, 100)

    def test_square_matrix(self):
        """测试方阵"""
        A = np.random.randn(50, 50)
        svd = svdFactory.create(svdMethod.static)
        svd.fit(A, r=25)

        assert svd.U.shape == (50, 25)
        assert svd.S.shape == (25,)
        assert svd.Vt.shape == (25, 50)

    def test_rank_one_matrix(self):
        """测试秩1矩阵"""
        u = np.random.randn(50, 1)
        v = np.random.randn(1, 30)
        A = u @ v  # 秩为1

        svd = svdFactory.create(svdMethod.static)
        svd.fit(A, r=5)

        # 第一个奇异值应该远大于其他的
        print(f"\n秩1矩阵的奇异值: {svd.S}")
        assert svd.S[0] > 100 * svd.S[1]

    def test_zero_matrix(self):
        """测试零矩阵"""
        A = np.zeros((50, 30))
        svd = svdFactory.create(svdMethod.static)
        svd.fit(A, r=10)

        # 所有奇异值应该为0
        assert np.allclose(svd.S, 0)

    def test_identity_matrix(self):
        """测试单位矩阵"""
        A = np.eye(50)
        svd = svdFactory.create(svdMethod.static)
        svd.fit(A, r=30)

        # 所有奇异值应该为1
        assert np.allclose(svd.S, 1.0)


class TestStaticSVDComparison:
    """对比工厂模式和直接实例化的结果"""

    def setup_method(self):
        np.random.seed(42)
        self.A = np.random.randn(100, 50)

    def test_results_identical(self):
        """测试两种方式结果完全相同"""
        # 工厂模式
        svd1 = svdFactory.create(svdMethod.static)
        svd1.fit(self.A, r=20)

        # 直接实例化
        svd2 = staticSVD(method=svdMethod.static)
        svd2.fit(self.A, r=20)

        # 结果应该相同
        assert np.allclose(svd1.U, svd2.U), "U matrices differ"
        assert np.allclose(svd1.S, svd2.S), "Singular values differ"
        assert np.allclose(svd1.Vt, svd2.Vt), "Vt matrices differ"

        # 重构结果应该相同
        recon1 = svd1.reconstruct()
        recon2 = svd2.reconstruct()
        assert np.allclose(recon1, recon2), "Reconstructions differ"


if __name__ == '__main__':
    # 运行所有测试
    pytest.main([__file__, '-v', '-s'])

    # 或者只运行特定测试
    # pytest.main([__file__, '-v', '-s', '-k', 'test_basic'])

    # 跳过慢速测试
    # pytest.main([__file__, '-v', '-s', '-m', 'not slow'])