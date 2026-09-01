"""zylab.reliability.updown 测试：Excel 金标准对照 + G/H 系数标定验证."""

from __future__ import annotations

import numpy as np
import pytest

from zylab.reliability.analysis import (
    analyze_updown_records,
    dixon_mood,
    parse_trial_records,
    response_points,
    run_sensitivity_test,
)
from zylab.reliability.errors import ReliabilityError
from zylab.reliability.updown import dixon_mood_core, gh_factors

# ------------------------------------------------ Excel 金标准数据
# 来源 ref/升降法试验数据_20250330.xlsx「试验结果」表：24 发固定步长
# 升降试验，步长 d=0.05（表内 b=0.5 为笔误），O=响应、X=不响应。
# 表内中间参数：n=11、A=0（以 3.20 为基准级）、B=10、M=0.91、ρ=1.52118、
# G=0.961、H=1.735；估计 μ=3.225、σ=0.07606。
_X = np.array(
    [
        3.20,
        3.15,
        3.20,
        3.25,
        3.30,
        3.35,
        3.30,
        3.25,
        3.20,
        3.15,
        3.20,
        3.25,
        3.30,
        3.25,
        3.20,
        3.15,
        3.10,
        3.15,
        3.20,
        3.25,
        3.20,
        3.15,
        3.20,
        3.25,
    ]
)
_Y = np.array([1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1])
_STEP = 0.05

#: 金标准记录的序列化文本（TrialRecordEdit 控件同款格式）
_RECORDS = ", ".join(f"{level:.2f} {'O' if hit else 'X'}" for level, hit in zip(_X, _Y))


def test_dixon_mood_core_excel_golden() -> None:
    """金标准点估计与中间参数：μ̂/σ̂/M/ρ 与表值一致.

    不响应为较少类别（11 vs 13），以最低试验水平 3.10 为 0 级计数：
    n=11、A=22、B=54、M=110/121≈0.9091（表值 0.91），与表内以 3.20
    为基准的 A=0、B=10 平移等价。σ̂ = 1.620·d·(M+0.029) 精确值
    0.075985（表值 0.07606 因表内 M 取舍入值 0.91 而略有出入）。
    """
    mu, sigma, detail = dixon_mood_core(_X, _Y, _STEP)
    assert mu == pytest.approx(3.225, abs=1.0e-9)
    assert sigma == pytest.approx(0.0759854, rel=1.0e-4)
    assert detail.n_used == 11
    assert detail.a_value == pytest.approx(22.0)
    assert detail.b_value == pytest.approx(54.0)
    assert detail.m_value == pytest.approx(110.0 / 121.0, rel=1.0e-6)
    assert detail.rho == pytest.approx(1.620 * (110.0 / 121.0 + 0.029), rel=1.0e-6)


def test_dixon_mood_excel_gh_factors() -> None:
    """G/H 查表系数：标定 G 与表值 0.961 一致；H 为线性式自洽口径.

    表值 H=1.735 为原 Dixon-Mood 表（开方式 σ̂ 的 σ̂² 标准误）口径，
    线性式（GB/T 24176）下沿用会低估 σ̂ 不确定度，本实现按线性式
    标定 H≈1.58（se(σ̂)=H·σ̂/√n）。
    """
    _, _, detail = dixon_mood_core(_X, _Y, _STEP)
    assert detail.g_factor == pytest.approx(0.960, abs=5.0e-3)
    assert detail.h_factor == pytest.approx(1.582, abs=1.0e-2)


def test_dixon_mood_standard_errors_follow_gh() -> None:
    """标准误按 G/H 系数式：se(μ̂)=G·σ̂/√n、se(σ̂)=H·σ̂/√n."""
    estimate = dixon_mood(_X, _Y, _STEP)
    assert estimate.detail is not None
    se_mu, se_sigma = estimate.detail.standard_errors(estimate.sigma)
    assert estimate.se_mu == pytest.approx(se_mu)
    assert estimate.se_sigma == pytest.approx(se_sigma)
    assert 0.0 < estimate.se_mu < estimate.sigma
    assert estimate.se_sigma > 0.0


def test_response_points_match_excel() -> None:
    """响应点估计与表值对照（正态模型，容差 1e-3 覆盖表内舍入）.

    表值：0.1% 点 2.98996、0.01% 点 2.94213、99.9% 点 3.46004、
    99.999% 点 3.54939（表以舍入后 ρ 计算，偏差 ~3e-4）。
    """
    estimate = dixon_mood(_X, _Y, _STEP)
    points = response_points("normal", estimate)
    expected = {0.0001: 2.94213, 0.001: 2.98996, 0.999: 3.46004, 0.9999: 3.50758, 0.99999: 3.54939}
    assert points.probs.size == points.x.size == points.se.size
    for prob, value in expected.items():
        index = int(np.argmin(np.abs(points.probs - prob)))
        assert points.probs[index] == pytest.approx(prob)
        assert points.x[index] == pytest.approx(value, abs=1.0e-3)
    # delta 法区间对称包住点估计
    assert np.all(points.x_low < points.x)
    assert np.all(points.x < points.x_high)
    assert np.allclose(points.x_high - points.x, points.x - points.x_low)


def test_response_points_weibull_and_missing_se() -> None:
    """weibull 参数化与缺失标准误（Kärber）：点估计有效、SE 置 NaN."""
    from zylab.reliability.analysis import karber

    estimate = karber(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 3, 7, 10]), 10)
    points = response_points("normal", estimate)
    assert np.all(np.isfinite(points.x))
    assert np.all(np.isnan(points.se))
    assert np.all(np.isnan(points.x_low)) and np.all(np.isnan(points.x_high))
    # weibull：(η, k) 参数化不适用 delta 法
    mle = run_sensitivity_test(
        method="probit", model="weibull", mu=10.0, sigma=1.5, x_low=2.0, x_high=25.0, n_levels=7, n_per_level=15, seed=4
    )
    assert mle.points is not None
    assert np.all(np.isfinite(mle.points.x))
    assert np.all(np.isnan(mle.points.se))


def test_response_points_rejects_invalid_probabilities() -> None:
    """响应概率越界报 ReliabilityError."""
    estimate = dixon_mood(_X, _Y, _STEP)
    with pytest.raises(ReliabilityError, match="响应概率"):
        response_points("normal", estimate, probs=(0.5, 1.5))
    with pytest.raises(ReliabilityError, match="响应概率"):
        response_points("normal", estimate, probs=(0.0, 0.5))


def test_gh_factors_interpolation_and_clamping() -> None:
    """G/H 系数插值：网格点取表值、网格外取端点、全程量级合理."""
    g_grid, h_grid = gh_factors(1.0)
    assert g_grid == pytest.approx(1.000, abs=1.0e-3)
    assert h_grid == pytest.approx(1.372, abs=1.0e-3)
    # 网格外端点截断
    assert gh_factors(0.3) == gh_factors(0.5)
    assert gh_factors(5.0) == gh_factors(3.0)
    for rho in np.linspace(0.5, 3.0, 26):
        g_factor, h_factor = gh_factors(float(rho))
        assert 0.3 < g_factor < 1.2
        assert 0.3 < h_factor < 2.1


def test_dixon_mood_degenerate_branch_sigma() -> None:
    """M < 0.3 退化分支：σ̂ = 0.53·d、ρ = 0.53（GB/T 24176）."""
    # 水平 0/1 各两发、响应集中在低级：M = 0.25 < 0.3
    x = np.array([9.0, 10.0, 11.0, 12.0])
    y = np.array([0, 0, 1, 1])
    _mu, sigma, detail = dixon_mood_core(x, y, 1.0)
    assert detail.m_value == pytest.approx(0.25)
    assert detail.rho == pytest.approx(0.53)
    assert sigma == pytest.approx(0.53)


def test_run_updown_carries_detail_and_points() -> None:
    """总装结果：升降法携带中间参数与响应点表，非升降法 detail 为 None."""
    result = run_sensitivity_test(
        method="updown", mu=10.0, sigma=1.0, n_total=60, x_low=9.0, x_high=11.0, step=0.8, seed=3
    )
    assert result.detail is not None
    assert result.detail.n_used > 0
    assert result.points is not None
    # 响应点随 p 单调不减，区间包住点估计
    assert np.all(np.diff(result.points.x) >= -1.0e-9)
    assert np.all(result.points.x_low <= result.points.x)
    assert np.all(result.points.x <= result.points.x_high)
    # 响应点与拟合曲线一致：p(x̂_p) ≈ p
    from zylab.reliability.model import response_prob

    for prob, x_value in zip(result.points.probs, result.points.x, strict=True):
        assert float(response_prob("logistic", x_value, result.mu_hat, result.sigma_hat)) == pytest.approx(
            float(prob), abs=1.0e-6
        )
    # 非升降法（MLE）：无中间参数，响应点表仍可用（MLE 收敛时 SE 有限）
    langlie = run_sensitivity_test(method="langlie", mu=10.0, sigma=1.0, n_total=30, seed=42)
    assert langlie.detail is None
    assert langlie.points is not None
    probit = run_sensitivity_test(
        method="probit", model="normal", mu=10.0, sigma=1.0, x_low=6.0, x_high=14.0, n_levels=7, n_per_level=12, seed=5
    )
    assert probit.detail is None
    assert probit.points is not None
    assert np.all(np.isfinite(probit.points.se))


# ------------------------------------------------ 实测记录分析


def test_parse_trial_records_formats() -> None:
    """记录解析：逗号/分号/冒号/无分隔与 1/0 记号均收敛同一结果."""
    for text in (
        "3.20 O, 3.15 X, 3.20 X",
        "3.20O 3.15X 3.20X",
        "3.2:o;3.15:x;3.2:0",
        "3.20 1, 3.15 0, 3.20 0",
    ):
        levels, responses = parse_trial_records(text)
        assert levels == pytest.approx([3.20, 3.15, 3.20])
        assert responses.tolist() == [1, 0, 0]


def test_parse_trial_records_rejects_invalid_text() -> None:
    """空文本与含无法解析残余的文本报 ReliabilityError."""
    for text in ("", "   ", "3.20 A, 3.15 X", "3.20 O abc", "O X O"):
        with pytest.raises(ReliabilityError, match="试验记录格式非法"):
            parse_trial_records(text)


def test_analyze_updown_records_excel_golden() -> None:
    """实测记录分析全链路：与 Excel 金标准 μ̂/σ̂/中间参数/响应点对齐.

    结果载荷 method=updown，估计器 Dixon-Mood，与逐发数组直算一致
    （曲线、响应点、试验记录齐全）。
    """
    result = analyze_updown_records(_RECORDS, _STEP, model="normal")
    assert result.method == "updown"
    assert result.estimate.estimator == "Dixon-Mood"
    assert result.levels == pytest.approx(_X)
    assert result.responses.tolist() == _Y.tolist()
    assert result.mu_hat == pytest.approx(3.225, abs=1.0e-9)
    assert result.sigma_hat == pytest.approx(0.0759854, rel=1.0e-4)
    assert result.detail is not None
    assert result.detail.n_used == 11
    assert result.detail.m_value == pytest.approx(110.0 / 121.0, rel=1.0e-6)
    assert result.points is not None
    assert result.points.x[2] == pytest.approx(3.46004, abs=1.0e-3)
    assert result.curve_x.size == result.curve_p.size > 0


def test_analyze_updown_records_rejects_bad_inputs() -> None:
    """模型名/步长非法与无混合响应数据报 ReliabilityError."""
    with pytest.raises(ReliabilityError, match="不受支持"):
        analyze_updown_records(_RECORDS, _STEP, model="unknown")
    with pytest.raises(ReliabilityError, match="步长须为正"):
        analyze_updown_records(_RECORDS, 0.0)
    with pytest.raises(ReliabilityError, match="无混合响应"):
        analyze_updown_records("3.20 O, 3.25 O, 3.30 O", _STEP)
