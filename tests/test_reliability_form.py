"""FORM / SORM 可靠性分析测试."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from zylab.reliability import (
    Distribution,
    MCResult,
    RandomVariable,
    ReliabilityError,
    form_analysis,
    mc_analysis,
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


# ---------- Monte Carlo ----------


class TestMC:
    def test_mc_crude_pf_close_to_form(self) -> None:
        R = RandomVariable("R", Distribution.NORMAL, {"loc": 100.0, "scale": 10.0})
        S = RandomVariable("S", Distribution.NORMAL, {"loc": 60.0, "scale": 8.0})

        def g(x):
            return float(x[0] - x[1])

        res_f = form_analysis(g, [R, S], grad=lambda _x: np.array([1.0, -1.0]))
        res_mc = mc_analysis(g, [R, S], n_samples=500_000, seed=42)
        assert res_mc.pf_ci_95[0] <= res_f.pf_form <= res_mc.pf_ci_95[1]
        assert res_f.beta == pytest.approx(res_mc.beta, abs=0.05)
        assert isinstance(res_mc, MCResult)

    def test_mc_lhs_method(self) -> None:
        R = RandomVariable("R", Distribution.NORMAL, {"loc": 100.0, "scale": 10.0})
        S = RandomVariable("S", Distribution.NORMAL, {"loc": 60.0, "scale": 8.0})
        res = mc_analysis(lambda x: float(x[0] - x[1]), [R, S], n_samples=100_000, method="lhc", seed=42)
        assert res.method == "lhc"
        assert 0 < res.pf < 1
        assert res.n_fail > 0

    def test_mc_sobol_method(self) -> None:
        R = RandomVariable("R", Distribution.NORMAL, {"loc": 100.0, "scale": 10.0})
        S = RandomVariable("S", Distribution.NORMAL, {"loc": 60.0, "scale": 8.0})
        res = mc_analysis(lambda x: float(x[0] - x[1]), [R, S], n_samples=100_000, method="sobol", seed=42)
        assert res.method == "sobol"
        assert 0 < res.pf < 1

    def test_mc_stats_fields(self) -> None:
        R = RandomVariable("R", Distribution.NORMAL, {"loc": 100.0, "scale": 10.0})
        S = RandomVariable("S", Distribution.NORMAL, {"loc": 60.0, "scale": 8.0})
        res = mc_analysis(lambda x: float(x[0] - x[1]), [R, S], n_samples=500_000, seed=42)
        assert res.cov > 0
        assert res.pf_ci_95[0] < res.pf < res.pf_ci_95[1]
        assert res.beta > 0

    def test_mc_bad_method_raises(self) -> None:
        R = RandomVariable("R", Distribution.NORMAL, {"loc": 100.0, "scale": 10.0})
        with pytest.raises(ValueError, match="unsupported method"):
            mc_analysis(lambda x: float(x[0]), [R], method="bogus")

    def test_mc_empty_variables_raises(self) -> None:
        with pytest.raises(ValueError, match="variables must not be empty"):
            mc_analysis(lambda x: float(x[0]), [])

    def test_mc_zero_samples_raises(self) -> None:
        R = RandomVariable("R", Distribution.NORMAL, {"loc": 100.0, "scale": 10.0})
        with pytest.raises(ValueError, match="must be positive"):
            mc_analysis(lambda x: float(x[0]), [R], n_samples=0)

    def test_mc_always_safe_pf_zero(self) -> None:
        """Always-safe LSF: Pf=0, beta=inf, CI degenerates."""
        R = RandomVariable("R", Distribution.NORMAL, {"loc": 100.0, "scale": 10.0})
        res = mc_analysis(lambda x: float(1000.0 - x[0]), [R], n_samples=10_000, seed=42)
        assert res.pf == 0.0
        assert res.beta == float("inf")
        assert res.n_fail == 0
        assert res.pf_ci_95[0] == 0.0

    def test_mc_always_fail_pf_one(self) -> None:
        """Always-failing LSF: Pf=1, beta=-inf, CI degenerates."""
        R = RandomVariable("R", Distribution.NORMAL, {"loc": 100.0, "scale": 10.0})
        res = mc_analysis(lambda _x: -1.0, [R], n_samples=10_000, seed=42)
        assert res.pf == 1.0
        assert res.beta == float("-inf")
        assert res.n_fail == 10_000
        assert res.pf_ci_95[1] == 1.0


# ---------------------------------------------------------------------------
# 内部辅助函数边界条件（不支持的分布、CUSTOM fallback）
# ---------------------------------------------------------------------------


def test_standard_to_physical_unsupported_dist() -> None:
    """_standard_to_physical 在遇到不支持的 dist 时抛 ReliabilityError."""
    # 构造一个伪造的 rv，dist 是无效枚举值
    import types

    from zylab.reliability.form import _standard_to_physical

    rv = types.SimpleNamespace(dist="NOT_A_DIST", params={})
    with pytest.raises(ReliabilityError, match="不支持的分布"):
        _standard_to_physical(0.0, rv)


def test_physical_to_standard_unsupported_dist() -> None:
    """_physical_to_standard 在遇到不支持的 dist 时抛 ReliabilityError."""
    import types

    from zylab.reliability.form import _physical_to_standard

    rv = types.SimpleNamespace(dist="NOT_A_DIST", params={})
    with pytest.raises(ReliabilityError, match="不支持的分布"):
        _physical_to_standard(0.0, rv)


def test_pdf_physical_custom_no_pdf_fallback() -> None:
    """_pdf_physical 对 CUSTOM 分布无 pdf 时返回标准正态 pdf."""
    from zylab.reliability.form import _pdf_physical

    # CUSTOM 分布但不传 pdf → fallback 到 norm.pdf
    rv = RandomVariable("c", Distribution.CUSTOM, {"ppf": lambda p: p})
    result = _pdf_physical(0.0, rv)
    # 标准正态在 0 点的 pdf ≈ 0.3989
    assert abs(result - stats.norm.pdf(0.0)) < 1e-10


def test_pdf_physical_unknown_dist_returns_zero() -> None:
    """_pdf_physical 对未知 dist 返回 0."""
    import types

    from zylab.reliability.form import _pdf_physical

    rv = types.SimpleNamespace(dist="NOT_A_DIST", params={})
    assert _pdf_physical(0.0, rv) == 0.0
