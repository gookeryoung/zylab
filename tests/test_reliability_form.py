"""FORM / SORM 可靠性分析测试."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from zylab.reliability import (
    Distribution,
    RandomVariable,
    ReliabilityError,
    form_analysis,
    sorm_analysis,
)

# ---------- 解析解基准：正态 R-S ----------


class TestFORMAnalytical:
    """G = R - S，R~N(100,10)，S~N(60,8)，解析解 β=(100-60)/√(10²+8²)=3.1235."""

    R = RandomVariable("R", Distribution.NORMAL, {"loc": 100.0, "scale": 10.0})
    S = RandomVariable("S", Distribution.NORMAL, {"loc": 60.0, "scale": 8.0})

    @pytest.fixture()
    def vars(self) -> list[RandomVariable]:
        return [self.R, self.S]

    @staticmethod
    def g(x: np.ndarray) -> float:
        return float(x[0] - x[1])

    @staticmethod
    def grad_g(_x: np.ndarray) -> np.ndarray:
        return np.array([1.0, -1.0])

    def test_beta_analytical(self, vars: list[RandomVariable]) -> None:
        """β 必须与解析解精确匹配（4 位小数）."""
        res = form_analysis(self.g, vars, grad=self.grad_g)
        assert res.converged
        assert res.beta == pytest.approx(3.1235, abs=1e-4)
        # pf_form 是 Φ(-beta) 的结果，用相对容差
        assert res.pf_form == pytest.approx(stats.norm.cdf(-res.beta), rel=1e-4)

    def test_beta_finite_diff(self, vars: list[RandomVariable]) -> None:
        """不传解析梯度，用中心差分也应收敛到同一 β."""
        res = form_analysis(self.g, vars)
        assert res.beta == pytest.approx(3.1235, abs=1e-3)

    def test_design_point_on_limit_state(self, vars: list[RandomVariable]) -> None:
        """设计点物理空间 G 值应 ≈ 0."""
        res = form_analysis(self.g, vars, grad=self.grad_g)
        assert abs(res.g_star) < 1e-6

    def test_design_point_magnitude(self, vars: list[RandomVariable]) -> None:
        """设计点标准化向量 β = ||u*||."""
        res = form_analysis(self.g, vars, grad=self.grad_g)
        assert float(np.linalg.norm(res.u_star)) == pytest.approx(res.beta)

    def test_start_std_custom(self, vars: list[RandomVariable]) -> None:
        """自定义起点（覆盖 start_std 参数分支）."""
        res = form_analysis(self.g, vars, grad=self.grad_g, start_std=np.array([1.0, -1.0]))
        assert res.converged
        assert res.beta == pytest.approx(3.1235, abs=1e-3)


# ---------- 非正态分布 ----------


class TestFORMSkew:
    """对数正态抗力 + Gumbel 载荷，HLRF 处理非正态."""

    def test_lognormal_gumbel(self) -> None:
        R = RandomVariable("R", Distribution.LOGNORMAL, {"loc": 120.0, "scale": 15.0})
        P = RandomVariable("P", Distribution.GUMBEL, {"loc": 50.0, "scale": 10.0})

        def g(x: np.ndarray) -> float:
            return float(x[0] - x[1])

        res = form_analysis(g, [R, P], grad=lambda _x: np.array([1.0, -1.0]))
        assert res.converged
        assert 1.0 < res.beta < 6.0
        assert res.pf_form > 0
        assert res.pf_form < 0.5

    def test_weibull_uniform(self) -> None:
        """Weibull 抗力 + Uniform 载荷."""
        R = RandomVariable("R", Distribution.WEIBULL, {"shape": 5.0, "scale": 100.0})
        P = RandomVariable("P", Distribution.UNIFORM, {"lo": 40.0, "hi": 80.0})

        res = form_analysis(
            lambda x: float(x[0] - x[1]),
            [R, P],
            grad=lambda _x: np.array([1.0, -1.0]),
        )
        assert res.converged
        assert res.beta > 0

    def test_exponential(self) -> None:
        """单变量指数分布."""
        R = RandomVariable("R", Distribution.EXPONENTIAL, {"loc": 30.0, "scale": 20.0})
        res = form_analysis(lambda x: float(x[0] - 60.0), [R], grad=lambda _x: np.array([1.0]))
        assert res.beta > 0
        assert res.pf_form > 0


class TestFORMErrorCases:
    """FORM 错误路径."""

    def test_empty_variables(self) -> None:
        with pytest.raises(ReliabilityError, match="不能为空"):
            form_analysis(lambda _x: 0.0, [])

    def test_infinite_limit_state(self) -> None:
        R = RandomVariable("R", Distribution.NORMAL, {"loc": 100.0, "scale": 10.0})
        with pytest.raises(ReliabilityError, match="非有限值"):
            form_analysis(lambda x: 1.0 / (x[0] - 100.0), [R], max_iter=3)

    def test_zero_gradient(self) -> None:
        """设计点处梯度为零向量 → ReliabilityError."""
        R = RandomVariable("R", Distribution.NORMAL, {"loc": 100.0, "scale": 10.0})
        # G = -(X-100)^2，在 X=100 处 G=0 且梯度=0
        with pytest.raises(ReliabilityError, match="梯度为零向量"):
            form_analysis(lambda x: float(-((x[0] - 100.0) ** 2)), [R], max_iter=1)

    def test_custom_distribution(self) -> None:
        """CUSTOM 分布接受 (cdf, ppf, pdf) callable."""
        R = RandomVariable(
            "R",
            Distribution.CUSTOM,
            {
                "cdf": lambda x: float(stats.norm.cdf(x, loc=100.0, scale=10.0)),
                "ppf": lambda p: float(stats.norm.ppf(p, loc=100.0, scale=10.0)),
                "pdf": lambda x: float(stats.norm.pdf(x, loc=100.0, scale=10.0)),
            },
        )
        res = form_analysis(lambda x: float(x[0] - 60.0), [R], grad=lambda _x: np.array([1.0]))
        assert res.converged


# ---------- SORM ----------


class TestSORM:
    """Breitung / Hohenbichler 曲率修正."""

    R = RandomVariable("R", Distribution.NORMAL, {"loc": 100.0, "scale": 10.0})
    S = RandomVariable("S", Distribution.NORMAL, {"loc": 60.0, "scale": 8.0})

    def test_linear_limit_state_curvature_zero(self) -> None:
        """G = R - S 是线性函数，曲率应为 0，SORM ≈ FORM."""

        def g(x: np.ndarray) -> float:
            return float(x[0] - x[1])

        def grad(_x: np.ndarray) -> np.ndarray:
            return np.array([1.0, -1.0])

        res = sorm_analysis(g, [self.R, self.S], grad=grad)
        assert res.pf_breitung == pytest.approx(res.form.pf_form, abs=1e-8)
        assert np.allclose(res.kappa, 0.0, atol=1e-6)

    def test_3d_von_mises(self) -> None:
        """三变量 von Mises 屈服面（非线性）."""
        sig_y = RandomVariable("sig_y", Distribution.NORMAL, {"loc": 250.0, "scale": 20.0})
        s1 = RandomVariable("s1", Distribution.NORMAL, {"loc": 120.0, "scale": 15.0})
        s2 = RandomVariable("s2", Distribution.NORMAL, {"loc": 80.0, "scale": 12.0})

        def g(x: np.ndarray) -> float:
            return float(x[0] - np.sqrt(x[1] ** 2 - x[1] * x[2] + x[2] ** 2))

        res = sorm_analysis(g, [sig_y, s1, s2])
        assert res.form.converged
        assert len(res.kappa) == 2
        assert math.isfinite(res.pf_breitung)

    def test_single_var_sorm(self) -> None:
        """单变量 SORM（曲率数组为空、Breitung = FORM）."""
        W = RandomVariable("W", Distribution.WEIBULL, {"shape": 3.0, "scale": 50.0})
        res = sorm_analysis(
            lambda x: float(x[0] - 35.0),
            [W],
            grad=lambda _x: np.array([1.0]),
        )
        assert res.form.converged
        assert len(res.kappa) == 0
        assert math.isfinite(res.pf_breitung)


# ---------- 结果数据类 ----------


class TestResultDataclass:
    def test_form_result_frozen(self) -> None:
        R = RandomVariable("R", Distribution.NORMAL, {"loc": 100.0, "scale": 10.0})
        S = RandomVariable("S", Distribution.NORMAL, {"loc": 60.0, "scale": 8.0})
        res = form_analysis(lambda x: float(x[0] - x[1]), [R, S])
        with pytest.raises(AttributeError):
            res.beta = 0.0  # type: ignore[misc]
