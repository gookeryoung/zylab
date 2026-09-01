"""zylab.reliability.analysis 统计分析测试：手算精确例 + 蒙特卡洛一致性验证."""

from __future__ import annotations

import numpy as np
import pytest

from zylab.reliability.analysis import (
    SensitivityTestResult,
    dixon_mood,
    karber,
    mle_estimate,
    run_sensitivity_test,
)
from zylab.reliability.errors import ReliabilityError
from zylab.reliability.methods import METHOD_NAMES

# ------------------------------------------------ 手算精确例


def test_karber_hand_computed_example() -> None:
    """Kärber 手算例：水平 [1,2,3,4]、命中 [0,3,7,10]/10 发.

    频率 p=[0,.3,.7,1]，边界 x_0=0（p=0）、x_5=5（p=1）：
    ∫p ≈ Σ½(p_i+p_{i+1})Δx = 0+0.15+0.5+0.85+1.0 = 2.5，
    μ̂ = b − ∫p = 5 − 2.5 = 2.5；
    ∫2x·p ≈ Σ½(p_i+p_{i+1})(x_{i+1}²−x_i²) = 17.9，
    σ̂² = b² − ∫2x·p − μ̂² = 25 − 17.9 − 6.25 = 0.85。
    """
    estimate = karber(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 3, 7, 10]), 10)
    assert estimate.estimator == "Kärber"
    assert estimate.mu == pytest.approx(2.5)
    assert estimate.sigma == pytest.approx(np.sqrt(0.85))


def test_karber_rejects_degenerate_input() -> None:
    """水平不足或发数非法报 ReliabilityError."""
    with pytest.raises(ReliabilityError, match="至少需要 2 个水平"):
        karber(np.array([1.0]), np.array([3]), 10)
    with pytest.raises(ReliabilityError, match="至少需要 2 个水平"):
        karber(np.array([1.0, 2.0]), np.array([3, 4]), 0)


def test_dixon_mood_symmetric_data() -> None:
    """Dixon-Mood 对称构造例：响应恰在 μ 上方一级.

    水平 [9(0),10(0),11(1),12(1)] 步长 1：响应计数较少（2 vs 2 取响应），
    响应重心 i=(2+3)/2=2.5，μ̂ = 9 + (2.5 − 0.5) = 11？按对称数据真实中心 10.5。
    """
    x = np.array([9.0, 10.0, 11.0, 12.0])
    y = np.array([0, 0, 1, 1])
    estimate = dixon_mood(x, y, 1.0)
    # 响应重心 = (11+12)/2 = 11.5，Dixon-Mood 修正 −0.5 步长 → 11.0
    assert estimate.mu == pytest.approx(11.0)
    assert estimate.sigma > 0.0


def test_dixon_mood_rejects_unmixed_data() -> None:
    """全响应数据无混合：Dixon-Mood 拒绝分析."""
    with pytest.raises(ReliabilityError, match="无混合响应"):
        dixon_mood(np.array([9.0, 10.0]), np.array([1, 1]), 1.0)


def test_mle_recovers_parameters_from_large_sample() -> None:
    """MLE 大样本恢复真值（500 点网格化二项数据，容差 5%）."""
    rng = np.random.default_rng(11)
    x = rng.uniform(7.0, 13.0, 500)
    p = 1.0 / (1.0 + np.exp(-(x - 10.0) / 1.2))
    y = (rng.random(500) < p).astype(int)
    estimate = mle_estimate("logistic", x, y)
    assert estimate.converged
    assert estimate.mu == pytest.approx(10.0, abs=0.25)
    assert estimate.sigma == pytest.approx(1.2, rel=0.1)
    assert 0.0 < estimate.se_mu < 0.2
    assert 0.0 < estimate.se_sigma < 0.2


def test_mle_separated_data_returns_unconverged() -> None:
    """完全分离数据：MLE 不存在，返回收敛失败与区间中点."""
    estimate = mle_estimate("logistic", np.array([8.0, 9.0, 11.0, 12.0]), np.array([0, 0, 1, 1]))
    assert not estimate.converged
    assert estimate.mu == pytest.approx(10.0)


def test_mle_rejects_insufficient_points() -> None:
    """试验点不足报 ReliabilityError."""
    with pytest.raises(ReliabilityError, match="至少需要 2 个试验点"):
        mle_estimate("logistic", np.array([10.0]), np.array([1]))


# ------------------------------------------------ 蒙特卡洛一致性验证（计算示例）


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("langlie", {}),  # 方法101 兰利法
        ("ostr", {}),  # 方法102 OSTR法
        ("updown", {}),  # 方法103 升降法
        ("doptimal", {}),  # 方法104 D优化法
        ("probit", {}),  # 方法201 概率单位法
        ("stepstress", {}),  # 方法202 完全步进法
    ],
)
def test_run_sensitivity_test_recovers_truth(method: str, kwargs: dict) -> None:
    """六方法固定种子模拟：估计 μ 落在真值 1.5σ 容差内、σ 量级正确.

    发数取 30（兰利法等序贯法标准推荐规模；发数过大时兰利区间塌缩
    到单点，σ 不可估）。
    """
    result = run_sensitivity_test(
        method=method,
        model="logistic",
        mu=10.0,
        sigma=1.0,
        n_total=30,
        x_low=6.0,
        x_high=14.0,
        step=0.8,
        n_per_level=10,
        n_levels=7,
        seed=7,
        **kwargs,
    )
    assert isinstance(result, SensitivityTestResult)
    assert result.method == method
    assert result.levels.size == result.responses.size
    assert set(np.unique(result.responses)) <= {0, 1}
    # 混合响应（有响应有未响应），估计落在真值邻域
    assert 0 < result.responses.sum() < result.responses.size
    assert result.mu_hat == pytest.approx(10.0, abs=1.5)
    assert 0.3 < result.sigma_hat < 3.0
    # 拟合曲线单调非降且落在 [0, 1]
    assert np.all(np.diff(result.curve_p) >= -1.0e-9)
    assert np.all((result.curve_p >= 0.0) & (result.curve_p <= 1.0))
    assert result.curve_x.size == result.curve_p.size


def test_run_probit_uses_normal_model() -> None:
    """概率单位法（Probit 模型）：正态响应 MLE 恢复真值."""
    result = run_sensitivity_test(
        method="probit", model="normal", mu=10.0, sigma=1.0, x_low=6.0, x_high=14.0, n_levels=7, n_per_level=12, seed=5
    )
    assert result.model == "normal"
    assert result.mu_hat == pytest.approx(10.0, abs=1.0)
    assert 0.4 < result.sigma_hat < 2.5


def test_run_updown_uses_dixon_mood() -> None:
    """升降法分析走 Dixon-Mood 公式（非 MLE）."""
    result = run_sensitivity_test(
        method="updown", mu=10.0, sigma=1.0, n_total=60, x_low=9.0, x_high=11.0, step=0.8, seed=3
    )
    assert result.estimate.estimator == "Dixon-Mood"
    assert result.mu_hat == pytest.approx(10.0, abs=1.5)


def test_run_stepstress_uses_karber() -> None:
    """完全步进法分析走 Kärber 公式（非参数）."""
    result = run_sensitivity_test(
        method="stepstress", mu=10.0, sigma=1.0, x_low=6.0, x_high=14.0, step=0.8, n_per_level=10, seed=9
    )
    assert result.estimate.estimator == "Kärber"
    assert result.mu_hat == pytest.approx(10.0, abs=1.5)


def test_run_sensitivity_test_reproducible_with_same_seed() -> None:
    """同种子完全可重复（DSL 模板示例验证的基础）."""
    kwargs = {"method": "langlie", "mu": 10.0, "sigma": 1.0, "n_total": 30, "seed": 42}
    first = run_sensitivity_test(**kwargs)
    second = run_sensitivity_test(**kwargs)
    np.testing.assert_array_equal(first.levels, second.levels)
    np.testing.assert_array_equal(first.responses, second.responses)
    assert first.mu_hat == second.mu_hat


def test_run_sensitivity_test_rejects_bad_arguments() -> None:
    """非法方法/模型/参数配置报 ReliabilityError."""
    with pytest.raises(ReliabilityError, match="方法"):
        run_sensitivity_test(method="ghost", seed=1)
    with pytest.raises(ReliabilityError, match="模型"):
        run_sensitivity_test(method="langlie", model="cauchy", seed=1)
    with pytest.raises(ReliabilityError, match="σ 须为正"):
        run_sensitivity_test(method="langlie", sigma=0.0, seed=1)


def test_all_methods_covered() -> None:
    """方法表完整（六方法与测试参数化一致）."""
    assert METHOD_NAMES == ("langlie", "ostr", "updown", "doptimal", "probit", "stepstress")
