"""优化模块（代理模型 + 代理上的 scipy 全局优化）单元测试."""

from __future__ import annotations

import numpy as np
import pytest

from zylab.doe import DesignVariable
from zylab.optim import (
    GprSurrogate,
    OptimError,
    Optimizer,
    OptimResult,
    RbfSurrogate,
    SurrogateError,
    optimize,
)


@pytest.fixture
def quadratic_1d():
    """一维凸函数训练样本：y = x^2，全局最小 x=0."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-5, 5, size=(40, 1))
    y = (X**2).ravel()
    return X, y


@pytest.fixture
def quadratic_2d():
    """二维凸函数训练样本：y = x0^2 + x1^2，全局最小 (0,0)."""
    rng = np.random.default_rng(0)
    X = rng.uniform([-5, -5], [5, 5], size=(30, 2))
    y = (X**2).sum(axis=1)
    return X, y


# ====================================================================== 1. 代理模型基本 fit/predict


class TestSurrogateFitPredict:
    def test_rbf_fit_predict(self, quadratic_2d) -> None:
        X, y = quadratic_2d
        rbf = RbfSurrogate().fit(X, y)
        preds = rbf.predict(X)
        np.testing.assert_allclose(preds, y, rtol=1e-10, atol=1e-10)

    def test_rbf_predict_shape(self, quadratic_2d) -> None:
        X, y = quadratic_2d
        rbf = RbfSurrogate().fit(X, y)
        preds = rbf.predict(X[:3])
        assert preds.shape == (3,)

    def test_rbf_not_fit(self) -> None:
        rbf = RbfSurrogate()
        with pytest.raises(SurrogateError):
            rbf.predict(np.array([[0.0, 0.0]]))

    def test_gpr_fit_predict(self, quadratic_2d) -> None:
        X, y = quadratic_2d
        gpr = GprSurrogate().fit(X, y)
        preds = gpr.predict(X)
        assert preds.shape == y.shape

    def test_gpr_predict_std(self, quadratic_2d) -> None:
        X, y = quadratic_2d
        gpr = GprSurrogate().fit(X, y)
        std = gpr.predict_std(X)
        assert std.shape == (len(X),)
        assert np.all(std >= 0)

    def test_base_predict_std_default(self, quadratic_2d) -> None:
        X, y = quadratic_2d
        rbf = RbfSurrogate().fit(X, y)
        assert rbf.predict_std(X) is None


# ====================================================================== 2. 形状校验


class TestSurrogateShapes:
    def test_X_wrong_dim(self) -> None:
        rbf = RbfSurrogate()
        with pytest.raises(SurrogateError):
            rbf.fit(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))

    def test_y_wrong_dim(self) -> None:
        rbf = RbfSurrogate()
        with pytest.raises(SurrogateError):
            rbf.fit(np.array([[1.0], [2.0]]), np.array([[1.0], [2.0]]))

    def test_mismatch(self) -> None:
        rbf = RbfSurrogate()
        with pytest.raises(SurrogateError):
            rbf.fit(np.array([[1.0], [2.0]]), np.array([1.0]))


# ====================================================================== 3. 优化器基本功能


class TestOptimizeBasic:
    def test_de_optimizer_works(self, quadratic_2d) -> None:
        """差分进化能跑完并返回合法 OptimResult，且解不越界."""
        X, y = quadratic_2d
        surrogate = RbfSurrogate().fit(X, y)
        vars_ = [
            DesignVariable.continuous("x0", -5.0, 5.0),
            DesignVariable.continuous("x1", -5.0, 5.0),
        ]
        result = optimize(surrogate, vars_, optimizer=Optimizer.DIFFERENTIAL_EVOLUTION, n_iter=30, seed=0)
        assert isinstance(result, OptimResult)
        assert result.optimizer == "differential_evolution"
        assert result.best_x.shape == (2,)
        # 优化器可能触边界
        assert np.all(np.abs(result.best_x) <= 5.0 + 1e-6)

    def test_dual_annealing_optimizer(self, quadratic_2d) -> None:
        X, y = quadratic_2d
        surrogate = RbfSurrogate().fit(X, y)
        vars_ = [
            DesignVariable.continuous("x0", -5.0, 5.0),
            DesignVariable.continuous("x1", -5.0, 5.0),
        ]
        result = optimize(surrogate, vars_, optimizer=Optimizer.DUAL_ANNEALING, n_iter=20, seed=0)
        assert result.best_x.shape == (2,)
        assert isinstance(result.best_y, float)

    def test_basinhopping_optimizer(self, quadratic_1d) -> None:
        X, y = quadratic_1d
        surrogate = RbfSurrogate().fit(X, y)
        vars_ = [DesignVariable.continuous("x0", -5.0, 5.0)]
        result = optimize(surrogate, vars_, optimizer=Optimizer.BASIN_HOPPING, n_iter=10, seed=0)
        assert result.best_x.shape == (1,)

    def test_string_optimizer_name(self, quadratic_1d) -> None:
        X, y = quadratic_1d
        surrogate = RbfSurrogate().fit(X, y)
        vars_ = [DesignVariable.continuous("x0", -5.0, 5.0)]
        result = optimize(surrogate, vars_, optimizer="shgo", n_iter=5, seed=0)
        assert result.optimizer == "shgo"

    def test_discrete_variable_rounding(self) -> None:
        """离散变量：优化器给的实数会 round 到最近 levels."""
        X = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 5.0], [10.0, 5.0], [5.0, 2.0]])
        y = np.array([0.0, 0.0, 0.0, 0.0, 100.0])
        surrogate = RbfSurrogate(smoothing=1e-4).fit(X, y)
        vars_ = [
            DesignVariable.continuous("x0", 0.0, 10.0),
            DesignVariable.discrete("x1", [0.0, 2.5, 5.0]),
        ]
        result = optimize(surrogate, vars_, optimizer=Optimizer.DUAL_ANNEALING, n_iter=15, seed=0)
        assert result.best_x[1] in (0.0, 2.5, 5.0)

    def test_callback_receives_each_eval(self, quadratic_1d) -> None:
        X, y = quadratic_1d
        surrogate = RbfSurrogate().fit(X, y)
        vars_ = [DesignVariable.continuous("x0", -5.0, 5.0)]
        calls: list[tuple[float, float]] = []

        def cb(x, f) -> None:
            calls.append((float(x[0]), float(f)))

        optimize(surrogate, vars_, optimizer=Optimizer.DUAL_ANNEALING, n_iter=5, seed=0, callback=cb)
        assert len(calls) > 0


# ====================================================================== 4. 参数校验


class TestOptimizeValidation:
    def test_no_surrogate(self) -> None:
        vars_ = [DesignVariable.continuous("x", 0.0, 1.0)]
        with pytest.raises(OptimError):
            optimize("not a surrogate", vars_)

    def test_surrogate_not_fit(self) -> None:
        vars_ = [DesignVariable.continuous("x", 0.0, 1.0)]
        rbf = RbfSurrogate()
        with pytest.raises(OptimError):
            optimize(rbf, vars_)

    def test_unknown_optimizer(self, quadratic_1d) -> None:
        X, y = quadratic_1d
        surrogate = RbfSurrogate().fit(X, y)
        vars_ = [DesignVariable.continuous("x0", -5.0, 5.0)]
        with pytest.raises(OptimError):
            optimize(surrogate, vars_, optimizer="foo_bar")


# ====================================================================== 5. OptimResult 结构 & x_dict


class TestOptimResult:
    def test_x_dict(self) -> None:
        rng = np.random.default_rng(7)
        X = rng.uniform([1.0, 0.1], [5.0, 0.5], size=(20, 2))
        y = ((X[:, 0] - 3.0) ** 2 + (X[:, 1] - 0.3) ** 2) * 100
        surrogate = RbfSurrogate().fit(X, y)
        vars_ = [
            DesignVariable.continuous("length", 1.0, 5.0),
            DesignVariable.continuous("height", 0.1, 0.5),
        ]
        result = optimize(surrogate, vars_, optimizer=Optimizer.DUAL_ANNEALING, n_iter=20, seed=5)
        d = result.x_dict(vars_)
        assert "length" in d
        assert "height" in d
        assert d["length"] == float(result.best_x[0])
        assert d["height"] == float(result.best_x[1])


# ====================================================================== 6. 完整链路：DOE → 训练代理 → 代理优化 → 真函数验证


class TestFullPipeline:
    def test_doe_surrogate_optimize_stress(self) -> None:
        """模拟真实 workflow 闭环：DOE 抽样 → 代理训练 → 代理优化 → 真函数验证."""
        from zylab.doe import DesignSpace, DesignVariable, SamplingMethod

        ds = DesignSpace.from_variables(
            [
                DesignVariable.continuous("beam.length", 1.0, 5.0),
                DesignVariable.continuous("beam.height", 0.1, 0.5),
                DesignVariable.discrete("mesh.level", [1.0, 2.0, 3.0]),
            ]
        )

        X_train = ds.sample(SamplingMethod.LATIN_HYPERCUBE, n_samples=30, seed=0)

        def true_model(X: np.ndarray) -> np.ndarray:
            return X[:, 0] / (X[:, 1] ** 2)

        y_train = true_model(X_train)

        surrogate = RbfSurrogate().fit(X_train, y_train)

        result = optimize(surrogate, ds.variables, optimizer=Optimizer.DIFFERENTIAL_EVOLUTION, n_iter=30, seed=1)

        true_value = float(true_model(result.best_x.reshape(1, -1))[0])
        rows = ds.to_input_rows(result.best_x.reshape(1, -1))
        param_dict = rows[0]

        assert result.best_x.shape == (3,)
        assert result.best_x[2] in (1.0, 2.0, 3.0)
        assert true_value > 0
        assert set(param_dict.keys()) == {"beam.length", "beam.height", "mesh.level"}

    def test_gpr_on_smooth_function(self) -> None:
        """GPR 在平滑函数上的代理误差 < 随机噪声（量级 O(1)）."""
        rng = np.random.default_rng(99)

        def f(X):
            return np.sin(X[:, 0]) * np.cos(X[:, 1])

        X = rng.uniform(-np.pi, np.pi, size=(40, 2))
        y = f(X)

        gpr = GprSurrogate().fit(X, y)
        X_test = rng.uniform(-np.pi, np.pi, size=(100, 2))
        y_true = f(X_test)
        y_pred = gpr.predict(X_test)
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        assert rmse < 0.5, f"GPR RMSE={rmse:.3f} 过大"


class TestSurrogateEdgeCases:
    def test_gpr_predict_not_fit(self) -> None:
        gpr = GprSurrogate()
        with pytest.raises(SurrogateError):
            gpr.predict(np.array([[0.0, 0.0]]))

    def test_gpr_predict_wrong_shape(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.uniform([-5, -5], [5, 5], size=(10, 2))
        y = (X**2).sum(axis=1)
        gpr = GprSurrogate().fit(X, y)
        with pytest.raises(SurrogateError):
            gpr.predict(X[0])  # 1D 行向量不是 2D

    def test_gpr_predict_std_not_fit(self) -> None:
        gpr = GprSurrogate()
        with pytest.raises(SurrogateError):
            gpr.predict_std(np.array([[0.0, 0.0]]))

    def test_gpr_kernel_not_fit(self) -> None:
        gpr = GprSurrogate()
        with pytest.raises(SurrogateError):
            _ = gpr.kernel_

    def test_rbf_predict_wrong_shape(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.uniform([-5, -5], [5, 5], size=(10, 2))
        y = (X**2).sum(axis=1)
        rbf = RbfSurrogate().fit(X, y)
        with pytest.raises(SurrogateError):
            rbf.predict(X[0])
