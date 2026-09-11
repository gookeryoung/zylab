"""DOE 模块单元测试."""

from __future__ import annotations

import numpy as np
import pytest

from zylab.doe import (
    DesignSpace,
    DesignVariable,
    DoeError,
    SamplingMethod,
    sample,
    sample_unit,
)

# ====================================================================== 1. DesignVariable 构造与校验


class TestDesignVariable:
    def test_continuous_normal(self) -> None:
        v = DesignVariable.continuous("x", 0.0, 10.0)
        assert v.name == "x"
        assert v.lower == 0.0
        assert v.upper == 10.0
        assert v.is_continuous
        assert not v.is_discrete

    def test_continuous_bad_bounds(self) -> None:
        with pytest.raises(DoeError, match="上界须严格大于下界"):
            DesignVariable.continuous("x", 5.0, 5.0)
        with pytest.raises(DoeError, match="上界须严格大于下界"):
            DesignVariable.continuous("x", 10.0, 0.0)

    def test_continuous_non_finite(self) -> None:
        v = DesignVariable(name="x", lower=0.0, upper=np.inf)
        with pytest.raises(DoeError):
            v.validate()

    def test_discrete_normal(self) -> None:
        v = DesignVariable.discrete("y", [1, 2, 3])
        assert v.is_discrete
        assert not v.is_continuous
        assert len(v.levels) == 3

    def test_discrete_too_few_levels(self) -> None:
        with pytest.raises(DoeError, match="至少需 2 个水平"):
            DesignVariable.discrete("y", [1])
        with pytest.raises(DoeError, match="至少需 2 个水平"):
            DesignVariable.discrete("y", [])

    def test_discrete_non_finite_level(self) -> None:
        v = DesignVariable(name="y", levels=(1.0, np.nan, 3.0))
        with pytest.raises(DoeError, match="水平值须为有限数"):
            v.validate()

    def test_default_construct_frozen(self) -> None:
        """连续变量默认构造 lower=0, upper=1 不报错."""
        v = DesignVariable(name="x")
        v.validate()  # default [0, 1] 合法


# ====================================================================== 2. normalize / denormalize 互逆


class TestDesignVariableNormalize:
    def test_continuous_roundtrip(self) -> None:
        v = DesignVariable.continuous("x", 1.0, 5.0)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        unit = v.normalize(values)
        assert np.allclose(unit, [0.0, 0.25, 0.5, 0.75, 1.0])
        restored = v.denormalize(unit)
        assert np.allclose(restored, values)

    def test_discrete_roundtrip(self) -> None:
        v = DesignVariable.discrete("y", [10.0, 20.0, 30.0, 40.0])
        unit = v.normalize(np.array([10.0, 20.0, 30.0, 40.0]))
        assert np.allclose(unit, [0.0, 1.0 / 3, 2.0 / 3, 1.0])
        restored = v.denormalize(unit)
        assert np.allclose(restored, [10.0, 20.0, 30.0, 40.0])

    def test_discrete_denormalize_nearest(self) -> None:
        """unit 非精确点时映射到最近离散水平."""
        v = DesignVariable.discrete("y", [10.0, 20.0, 30.0])
        result = v.denormalize(np.array([0.1, 0.45, 0.9]))
        np.testing.assert_array_equal(result, [10.0, 20.0, 30.0])

    def test_continuous_normalize_clipped(self) -> None:
        """normalize/denormalize 处理标量."""
        v = DesignVariable.continuous("x", 0.0, 10.0)
        assert abs(v.normalize(5.0) - 0.5) < 1e-12


# ====================================================================== 3. 四种采样方法基本性质


class TestSamplingBasic:
    @pytest.fixture
    def variables(self):
        return [
            DesignVariable.continuous("a", 0.0, 10.0),
            DesignVariable.continuous("b", -1.0, 1.0),
            DesignVariable.discrete("c", [1.0, 2.0, 3.0]),
        ]

    def test_lhc_shape_and_bounds(self, variables) -> None:
        X = sample(variables, SamplingMethod.LATIN_HYPERCUBE, n_samples=50, seed=0)
        assert X.shape == (50, 3)
        assert np.all(X[:, 0] >= 0.0) and np.all(X[:, 0] <= 10.0)
        assert np.all(X[:, 1] >= -1.0) and np.all(X[:, 1] <= 1.0)
        # 离散列只取 levels 值
        assert set(np.unique(X[:, 2])).issubset({1.0, 2.0, 3.0})

    def test_lhc_stratification(self, variables) -> None:
        """LHC 每层每维恰好一个样本."""
        n = 10
        unit = sample_unit(variables[:2], SamplingMethod.LATIN_HYPERCUBE, n_samples=n, seed=7)
        for col in range(2):
            bin_idx = np.floor(unit[:, col] * n).astype(int)
            np.testing.assert_array_equal(np.sort(bin_idx), np.arange(n))

    def test_sobol_shape(self, variables) -> None:
        X = sample(variables, SamplingMethod.SOBOL, n_samples=64, seed=1)
        assert X.shape == (64, 3)
        assert np.all(X[:, 0] >= 0.0) and np.all(X[:, 0] <= 10.0)

    def test_random_shape_and_bounds(self, variables) -> None:
        X = sample(variables, SamplingMethod.RANDOM, n_samples=30, seed=99)
        assert X.shape == (30, 3)
        assert np.all(X[:, 0] >= 0.0) and np.all(X[:, 0] <= 10.0)
        assert np.all(X[:, 1] >= -1.0) and np.all(X[:, 1] <= 1.0)
        assert set(np.unique(X[:, 2])).issubset({1.0, 2.0, 3.0})

    def test_factorial_count(self) -> None:
        """全因子点数 = n_per_dim ^ d，且点均匀分布."""
        vars_ = [
            DesignVariable.continuous("x", 0.0, 1.0),
            DesignVariable.continuous("y", 0.0, 1.0),
        ]
        X = sample(vars_, SamplingMethod.FULL_FACTORIAL, n_per_dim=5)
        assert X.shape == (25, 2)
        # 5 个水平值（中心点法）：0.1, 0.3, 0.5, 0.7, 0.9
        grid = np.round(np.linspace(0.1, 0.9, 5), 1)
        for col in [X[:, 0], X[:, 1]]:
            col_rounded = np.round(col, 1)
            np.testing.assert_array_equal(np.sort(col_rounded), np.repeat(grid, 5))

    def test_factorial_few_vars(self) -> None:
        """变量数多到 n_per_dim^d 超 1e6 时报错."""
        big_vars = [DesignVariable.continuous(f"x{i}", 0.0, 1.0) for i in range(50)]
        with pytest.raises(DoeError, match="超过安全上限"):
            sample(big_vars, SamplingMethod.FULL_FACTORIAL, n_per_dim=10)

    def test_reproducible_with_seed(self) -> None:
        """同 seed → 同结果（LHC / Sobol / Random）."""
        vars_ = [DesignVariable.continuous("x", 0.0, 1.0) for _ in range(3)]
        for m in [SamplingMethod.LATIN_HYPERCUBE, SamplingMethod.RANDOM]:
            X1 = sample(vars_, m, n_samples=20, seed=42)
            X2 = sample(vars_, m, n_samples=20, seed=42)
            np.testing.assert_array_equal(X1, X2)
            X3 = sample(vars_, m, n_samples=20, seed=43)
            assert not np.allclose(X1, X3)

    def test_sample_unit_range(self) -> None:
        """sample_unit 输出必须在 [0, 1] 内."""
        vars_ = [DesignVariable.continuous("x", 0.0, 1.0) for _ in range(4)]
        for m in [SamplingMethod.LATIN_HYPERCUBE, SamplingMethod.SOBOL, SamplingMethod.RANDOM]:
            U = sample_unit(vars_, m, n_samples=50, seed=5)
            assert U.shape == (50, 4)
            assert np.all(U >= 0.0) and np.all(U <= 1.0)


# ====================================================================== 4. 参数校验


class TestSamplingValidation:
    def test_no_variables(self) -> None:
        with pytest.raises(DoeError, match="至少需 1 个"):
            sample([], SamplingMethod.RANDOM, n_samples=5)

    def test_zero_samples(self) -> None:
        vars_ = [DesignVariable.continuous("x", 0.0, 1.0)]
        with pytest.raises(DoeError):
            sample(vars_, SamplingMethod.RANDOM, n_samples=0)

    def test_unknown_method(self) -> None:
        vars_ = [DesignVariable.continuous("x", 0.0, 1.0)]
        with pytest.raises(DoeError):
            sample(vars_, "not_a_method", n_samples=5)


# ====================================================================== 5. DesignSpace 容器


class TestDesignSpace:
    def test_from_variables(self) -> None:
        vars_ = [DesignVariable.continuous("x", 0.0, 1.0), DesignVariable.discrete("y", [1, 2])]
        ds = DesignSpace.from_variables(vars_)
        assert ds.n_vars == 2
        assert len(ds) == 2
        assert ds[0].name == "x"

    def test_duplicate_names(self) -> None:
        vars_ = [
            DesignVariable.continuous("x", 0.0, 1.0),
            DesignVariable.continuous("x", 0.0, 1.0),
        ]
        with pytest.raises(DoeError, match="重复"):
            DesignSpace.from_variables(vars_)

    def test_empty_variables(self) -> None:
        with pytest.raises(DoeError, match="至少需 1 个"):
            DesignSpace.from_variables([])

    def test_space_sample_shortcut(self) -> None:
        vars_ = [DesignVariable.continuous(f"x{i}", 0.0, 1.0) for i in range(3)]
        ds = DesignSpace.from_variables(vars_)
        X = ds.sample(SamplingMethod.RANDOM, n_samples=10, seed=0)
        assert X.shape == (10, 3)

    def test_to_input_rows(self) -> None:
        ds = DesignSpace.from_variables(
            [
                DesignVariable.continuous("beam.length", 1.0, 5.0),
                DesignVariable.discrete("mesh.level", [1.0, 2.0]),
            ]
        )
        X = np.array([[3.0, 1.0], [2.5, 2.0], [4.9, 1.0]])
        rows = ds.to_input_rows(X)
        assert len(rows) == 3
        assert rows[0] == {"beam.length": 3.0, "mesh.level": 1.0}
        assert rows[1] == {"beam.length": 2.5, "mesh.level": 2.0}
        assert rows[2] == {"beam.length": 4.9, "mesh.level": 1.0}

    def test_to_input_rows_shape_mismatch(self) -> None:
        ds = DesignSpace.from_variables([DesignVariable.continuous("x", 0.0, 1.0)])
        X = np.array([[0.0, 0.0], [1.0, 1.0]])  # 2 columns but 1 var
        with pytest.raises(DoeError):
            ds.to_input_rows(X)

    def test_to_input_rows_1d_rejected(self) -> None:
        ds = DesignSpace.from_variables([DesignVariable.continuous("x", 0.0, 1.0)])
        with pytest.raises(DoeError):
            ds.to_input_rows(np.array([1.0, 2.0, 3.0]))


# ====================================================================== 6. 集成：DOE 采样 → 构造 ParameterStore 输入 → 单次 workflow


class TestDoeParameterStoreIntegration:
    def test_rows_align_with_param_store_keys(self) -> None:
        """采样矩阵的键名格式 "node_id.param_key" 与 ParameterStore.inputs 对齐."""
        ds = DesignSpace.from_variables(
            [
                DesignVariable.continuous("cantilever.length", 1.0, 5.0),
                DesignVariable.continuous("cantilever.height", 0.1, 0.5),
                DesignVariable.continuous("load.magnitude", 100.0, 1000.0),
            ]
        )
        X = ds.sample(SamplingMethod.LATIN_HYPERCUBE, n_samples=7, seed=11)
        rows = ds.to_input_rows(X)

        # 每行的键名必须是 "node.param" 格式，且值在各自 bounds 内
        for row in rows:
            assert "cantilever.length" in row
            assert "cantilever.height" in row
            assert "load.magnitude" in row
            assert 1.0 <= row["cantilever.length"] <= 5.0
            assert 0.1 <= row["cantilever.height"] <= 0.5
            assert 100.0 <= row["load.magnitude"] <= 1000.0
