"""optim.sensitivity — Sobol 方差分解感度分析测试.

覆盖：Saltelli 矩阵构造边界、一阶 Si / 总效应 STi 数值正确性（纯加法函数
解析可验证）、报告格式化、空方差（常数函数）、非法参数拒绝。
"""

from __future__ import annotations

import numpy as np
import pytest

from zylab.optim import (
    OptimError,
    SaltelliSample,
    SobolIndices,
    build_saltelli_sample,
    sobol_analysis,
    sobol_indices,
)

__all__ = []


class TestBuildSaltelliSample:
    """Saltelli-Amat 设计矩阵构造."""

    def test_shapes_and_count(self) -> None:
        d, N = 3, 64
        s = build_saltelli_sample(d, N, seed=7)
        assert isinstance(s, SaltelliSample)
        assert s.A.shape == (N, d)
        assert s.B.shape == (N, d)
        assert s.C_A.shape == (d, N, d)
        assert s.C_B.shape == (d, N, d)
        assert s.d == d
        assert s.N == N
        assert s.total_evals == (2 * d + 2) * N
        # 全部落在 [0,1]
        for arr in (s.A, s.B, s.C_A, s.C_B):
            assert arr.min() >= 0.0 - 1e-9
            assert arr.max() <= 1.0 + 1e-9

    def test_column_swap_property(self) -> None:
        """C_A[i, :, i] 应等于 B[:, i]，其他列来自 A."""
        s = build_saltelli_sample(d=4, N=128, seed=1)
        for i in range(4):
            np.testing.assert_allclose(s.C_A[i, :, i], s.B[:, i], atol=1e-12)
            np.testing.assert_allclose(s.C_B[i, :, i], s.A[:, i], atol=1e-12)
            # 非 i 列的 C_A 来自 A
            for j in range(4):
                if j != i:
                    np.testing.assert_allclose(s.C_A[i, :, j], s.A[:, j], atol=1e-12)

    def test_seed_reproducible(self) -> None:
        s1 = build_saltelli_sample(2, 16, seed=42)
        s2 = build_saltelli_sample(2, 16, seed=42)
        np.testing.assert_array_equal(s1.A, s2.A)
        np.testing.assert_array_equal(s1.B, s2.B)

    def test_d_too_small(self) -> None:
        with pytest.raises(OptimError, match="至少需 2 个参数"):
            build_saltelli_sample(1, 64)

    def test_N_too_small(self) -> None:
        with pytest.raises(OptimError, match="最低基样本量"):
            build_saltelli_sample(2, 4)


class TestSobolIndices:
    """纯数值 sobol_indices —— 不依赖 build_saltelli_sample."""

    def test_pure_additive_first_order(self) -> None:
        """纯加法函数 y = x0 + 2*x1 → STi = Si（无交互，系数平方比 1:4）.

        Saltelli 一阶估计器在有限 N 下方差较大（理论上需 N ≥ 2^13 稳定），
        因此用 N=4096 + rtol=0.5 放宽；Jansen ST 估计器更稳——
        重点验证 ST 方向和比例正确。
        """
        N = 4096
        rng = np.random.default_rng(0)
        d = 2
        A = rng.random((N, d))
        B = rng.random((N, d))
        C_A = np.array([A.copy(), A.copy()])
        C_A[0, :, 0] = B[:, 0]
        C_A[1, :, 1] = B[:, 1]

        def f(X):
            return X[:, 0] + 2 * X[:, 1]

        Y_A = f(A)
        Y_B = f(B)
        Y_CA = np.array([f(C_A[0]), f(C_A[1])])

        Si, STi, _, _ = sobol_indices(Y_A, Y_B, Y_CA)
        # 总效应应覆盖全部方差（STi sum ≈ 1）
        assert STi.sum() >= 1.0 - 0.1
        # 一阶 ≈ 总效应（加法函数无交互，所以 STi - Si 应接近 0）
        np.testing.assert_allclose(Si, STi, atol=0.2)
        # ST 的系数平方比应稳定：ST1/ST0 ≈ 4
        np.testing.assert_allclose(STi[1] / STi[0], 4.0, rtol=0.35)

    def test_constant_function_all_zero(self) -> None:
        """常数函数 → 所有指数应为 0，防止除零."""
        Y_A = np.full(64, 5.0)
        Y_B = np.full(64, 5.0)
        Y_CA = np.zeros((2, 64))
        Si, STi, var, mean = sobol_indices(Y_A, Y_B, Y_CA)
        assert np.allclose(Si, 0.0)
        assert np.allclose(STi, 0.0)
        assert var == 0.0
        assert mean == 5.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(OptimError):
            sobol_indices(np.zeros(10), np.zeros(11), np.zeros((2, 10)))

    def test_Y_CA_wrong_dim_raises(self) -> None:
        """Y_CA 必须是二维 (d, N) —— 三维或一维都应报错."""
        with pytest.raises(OptimError, match="Y_CA 须为二维"):
            sobol_indices(np.zeros(8), np.zeros(8), np.zeros((2, 2, 8)))
        with pytest.raises(OptimError, match="Y_CA 须为二维"):
            sobol_indices(np.zeros(8), np.zeros(8), np.zeros(8))

    def test_jansen_fallback_without_YCB(self) -> None:
        """不传 Y_CB 时走 Jansen 估计器."""
        N = 512
        rng = np.random.default_rng(3)
        d = 2
        A = rng.random((N, d))
        B = rng.random((N, d))
        C_A = np.array([A.copy(), A.copy()])
        C_A[0, :, 0] = B[:, 0]
        C_A[1, :, 1] = B[:, 1]

        def f(x):
            return x[:, 0] + x[:, 1]

        Y_A, Y_B = f(A), f(B)
        Y_CA_vals = np.array([f(C_A[0]), f(C_A[1])])
        _, STi, _, _ = sobol_indices(Y_A, Y_B, Y_CA_vals, Y_CB=None)
        assert STi.sum() >= 1.0 - 0.1


class TestSobolAnalysis:
    """sobol_analysis 端到端入口."""

    def test_names_and_report(self) -> None:
        def f(x):
            return x[0] ** 2 + 3 * x[1]

        result = sobol_analysis(f, d=2, N=512, names=["alpha", "beta"])
        assert isinstance(result, SobolIndices)
        assert result.mean != 0.0
        assert result.variance > 0
        assert result.Si.shape == (2,)
        assert result.STi.shape == (2,)
        # report 含参数名
        report = result.report()
        assert "alpha" in report
        assert "beta" in report
        # sort_by
        order = result.sort_by("STi")
        assert len(order) == 2

    def test_func_accepts_batch_input(self) -> None:
        """向量化 func（接受 (n, d) 批量输入）应正常工作."""

        def f_batch(X):
            return X[:, 0] + X[:, 1]

        result = sobol_analysis(f_batch, d=2, N=512)
        assert result.STi.sum() >= 1.0 - 0.1

    def test_func_accepts_scalar(self) -> None:
        """仅接受单条 (d,) 输入的 func 也应工作（回退到 for 循环）."""

        def f_single(x):
            return x[0] + x[1]

        result = sobol_analysis(f_single, d=2, N=256)
        assert result.STi.sum() >= 1.0 - 0.1

    def test_names_length_mismatch_raises(self) -> None:
        def f(x):
            return x[0]

        with pytest.raises(OptimError, match="names 长度"):
            sobol_analysis(f, d=2, N=64, names=["a"])

    def test_func_raises_typeerror_uses_loop_fallback(self) -> None:
        """func 显式抛 TypeError → 强制走 for 循环分支."""

        def f_strict(x):
            if np.ndim(x) != 1:
                raise TypeError("expect 1D vector, got batch")
            return float(x[0] + 2 * x[1])

        result = sobol_analysis(f_strict, d=2, N=256)
        assert result.STi.sum() >= 1.0 - 0.1
