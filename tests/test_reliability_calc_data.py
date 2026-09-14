"""可靠性感度分析计算正确性数据驱动校核（测试数据见 tests/data/*.json）.

数据集按典型 + 极端情形手工设计，期望值全部由标准闭式公式独立核算：

- ``reliability_dixon_mood_cases.json``：Dixon-Mood 升降法 n/A/B/M、μ̂/σ̂、
  M<0.3 退化分支、G/H 查表插值与越界截断、负水平/尺度缩放/乱序/网格抖动；
- ``reliability_karber_cases.json``：Spearman-Kärber 梯形积分、方差负值截断、
  spacing/2 下限、不等间隔、非单调频率、尺度缩放；
- ``reliability_mle_cases.json``：对称频数下 μ̂ 严格居中与概率互补、
  完全分离 MLE 不存在、weibull 正值域；
- ``reliability_designer_cases.json``：六种序贯/固定设计的下一水平规则。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from zylab.reliability.analysis import dixon_mood, karber, mle_estimate
from zylab.reliability.errors import ReliabilityError
from zylab.reliability.methods import (
    doptimal_next,
    langlie_next,
    neyer_next,
    ostr_next,
    probit_levels,
    stepstress_levels,
    updown_adaptive_next,
    updown_next,
)
from zylab.reliability.model import response_prob
from zylab.reliability.updown import dixon_mood_core

_DATA_DIR = Path(__file__).parent / "data"


def _load(name: str) -> dict:
    """读取 tests/data 下的 JSON 校核数据集."""
    with (_DATA_DIR / name).open(encoding="utf-8") as fh:
        return json.load(fh)


def _ids(cases: list[dict]) -> list[str]:
    return [case["name"] for case in cases]


_dm_data = _load("reliability_dixon_mood_cases.json")
_karber_data = _load("reliability_karber_cases.json")
_mle_data = _load("reliability_mle_cases.json")
_designer_data = _load("reliability_designer_cases.json")


# ---------------------------------------------------------------------------
# Dixon-Mood 升降法
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _dm_data["cases"], ids=_ids(_dm_data["cases"]))
def test_dixon_mood_data_cases(case: dict) -> None:
    """逐数据集校核 Dixon-Mood 中间参数/点估计/G-H 系数或错误路径."""
    x, y, step = np.asarray(case["x"], dtype=float), np.asarray(case["y"], dtype=int), float(case["step"])
    if "raises" in case:
        if case.get("target") == "dixon_mood":
            with pytest.raises(ReliabilityError, match=case["raises"]):
                dixon_mood(x, y, step)
        else:
            with pytest.raises(ReliabilityError, match=case["raises"]):
                dixon_mood_core(x, y, step)
        return

    mu, sigma, detail = dixon_mood_core(x, y, step)
    expected = case["expected"]
    rtol = case.get("rtol", 1.0e-9)
    assert mu == pytest.approx(expected["mu"], rel=rtol, abs=1.0e-12)
    assert sigma == pytest.approx(expected["sigma"], rel=rtol, abs=1.0e-12)
    assert detail.n_used == expected["n_used"]
    assert detail.a_value == pytest.approx(expected["a"], rel=rtol)
    assert detail.b_value == pytest.approx(expected["b"], rel=rtol)
    assert detail.m_value == pytest.approx(expected["m"], rel=rtol)
    assert detail.rho == pytest.approx(expected["rho"], rel=rtol)
    if "g" in expected:
        assert detail.g_factor == pytest.approx(expected["g"], rel=1.0e-9)
        assert detail.h_factor == pytest.approx(expected["h"], rel=1.0e-9)
    # G/H 系数始终在标定表量级范围内
    assert 0.3 < detail.g_factor < 1.2
    assert 0.3 < detail.h_factor < 2.1


# ---------------------------------------------------------------------------
# Spearman-Kärber 完全步进法
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _karber_data["cases"], ids=_ids(_karber_data["cases"]))
def test_karber_data_cases(case: dict) -> None:
    """逐数据集校核 Kärber 均值/标准差或非法输入."""
    x = np.asarray(case["x_levels"], dtype=float)
    hits = np.asarray(case["hits"], dtype=int)
    n_per_level = int(case["n_per_level"])
    if "raises" in case:
        with pytest.raises(ReliabilityError, match=case["raises"]):
            karber(x, hits, n_per_level)
        return
    estimate = karber(x, hits, n_per_level)
    expected = case["expected"]
    assert estimate.mu == pytest.approx(expected["mu"], rel=1.0e-10, abs=1.0e-12)
    assert estimate.sigma == pytest.approx(expected["sigma"], rel=1.0e-10, abs=1.0e-12)
    assert estimate.estimator == "Kärber"
    assert np.isfinite(estimate.mu) and estimate.sigma > 0.0


# ---------------------------------------------------------------------------
# 极大似然估计
# ---------------------------------------------------------------------------


def _expand_grouped(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """分组频数数据展开为逐发 (水平, 0/1) 数组."""
    levels: list[float] = []
    responses: list[int] = []
    n_per_level = int(spec["n_per_level"])
    for level, raw_hits in zip(spec["levels"], spec["hits"], strict=True):
        hit_count = int(raw_hits)
        levels.extend([float(level)] * n_per_level)
        responses.extend([1] * hit_count + [0] * (n_per_level - hit_count))
    return np.asarray(levels), np.asarray(responses, dtype=int)


@pytest.mark.parametrize("case", _mle_data["cases"], ids=_ids(_mle_data["cases"]))
def test_mle_data_cases(case: dict) -> None:
    """逐数据集校核 MLE：对称中心/互补概率/分离识别/参数域错误."""
    if "grouped" in case:
        x, y = _expand_grouped(case["grouped"])
    else:
        x = np.asarray(case["raw"]["x"], dtype=float)
        y = np.asarray(case["raw"]["y"], dtype=int)
    model = case["model"]

    if "raises" in case:
        with pytest.raises(ReliabilityError, match=case["raises"]):
            mle_estimate(model, x, y)
        return

    estimate = mle_estimate(model, x, y)
    expected = case["expected"]
    assert estimate.converged is expected["converged"]

    if expected.get("converged"):
        if "mu" in expected:
            assert estimate.mu == pytest.approx(expected["mu"], abs=expected.get("mu_abs_tol", 1.0e-8))
        if "mu_about" in expected:
            assert estimate.mu == pytest.approx(expected["mu_about"], rel=expected.get("param_rtol", 0.05))
        assert estimate.sigma == pytest.approx(expected["sigma_about"], rel=expected.get("sigma_rtol", 0.05))
        if expected.get("se_positive"):
            assert np.isfinite(estimate.se_mu) and estimate.se_mu > 0.0
            assert np.isfinite(estimate.se_sigma) and estimate.se_sigma > 0.0
        if expected.get("complement"):
            # 对称中心处拟合：p(x)+p(-x)=1
            for level in np.unique(x):
                p_plus = float(response_prob(model, float(level), estimate.mu, estimate.sigma))
                p_minus = float(response_prob(model, float(-level), estimate.mu, estimate.sigma))
                assert p_plus + p_minus == pytest.approx(1.0, abs=1.0e-8)
        if expected.get("monotone"):
            probs = [float(response_prob(model, float(level), estimate.mu, estimate.sigma)) for level in np.unique(x)]
            assert np.all(np.diff(probs) > 0.0)
    else:
        assert estimate.mu == pytest.approx(expected["mu"], rel=1.0e-12)
        assert estimate.sigma == pytest.approx(expected["sigma"], rel=1.0e-12)
        if expected.get("se_nan"):
            assert np.isnan(estimate.se_mu) and np.isnan(estimate.se_sigma)


# ---------------------------------------------------------------------------
# 序贯/固定设计器规则
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _designer_data["updown"], ids=_ids(_designer_data["updown"]))
def test_updown_next_data_cases(case: dict) -> None:
    levels, responses = np.asarray(case["levels"], dtype=float), np.asarray(case["responses"], dtype=int)
    kwargs = {"x_low": case["x_low"], "x_high": case["x_high"], "step": case["step"]}
    if "raises" in case:
        with pytest.raises(ReliabilityError, match=case["raises"]):
            updown_next(levels, responses, **kwargs)
    else:
        assert updown_next(levels, responses, **kwargs) == pytest.approx(case["expected"])


@pytest.mark.parametrize("case", _designer_data["updown_adaptive"], ids=_ids(_designer_data["updown_adaptive"]))
def test_updown_adaptive_next_data_cases(case: dict) -> None:
    levels, responses = np.asarray(case["levels"], dtype=float), np.asarray(case["responses"], dtype=int)
    kwargs = {
        "x_low": case["x_low"],
        "x_high": case["x_high"],
        "step": case["step"],
        "sigma": case.get("sigma", 1.0),
        "floor_factor": case.get("floor_factor", 0.2),
    }
    if "raises" in case:
        with pytest.raises(ReliabilityError, match=case["raises"]):
            updown_adaptive_next(levels, responses, **kwargs)
    else:
        assert updown_adaptive_next(levels, responses, **kwargs) == pytest.approx(case["expected"])


@pytest.mark.parametrize("case", _designer_data["langlie"], ids=_ids(_designer_data["langlie"]))
def test_langlie_next_data_cases(case: dict) -> None:
    levels, responses = np.asarray(case["levels"], dtype=float), np.asarray(case["responses"], dtype=int)
    assert langlie_next(levels, responses, case["x_low"], case["x_high"]) == pytest.approx(case["expected"])


@pytest.mark.parametrize("case", _designer_data["ostr"], ids=_ids(_designer_data["ostr"]))
def test_ostr_next_data_cases(case: dict) -> None:
    levels, responses = np.asarray(case["levels"], dtype=float), np.asarray(case["responses"], dtype=int)
    next_level = ostr_next(levels, responses, case["x_low"], case["x_high"], case["model"], case["mu"], case["sigma"])
    if "expected" in case:
        assert next_level == pytest.approx(case["expected"])
    else:
        low, high = case["expected_between"]
        assert low - 1.0e-12 <= next_level <= high + 1.0e-12


@pytest.mark.parametrize("case", _designer_data["neyer"], ids=_ids(_designer_data["neyer"]))
def test_neyer_next_data_cases(case: dict) -> None:
    levels, responses = np.asarray(case["levels"], dtype=float), np.asarray(case["responses"], dtype=int)
    next_level = neyer_next(levels, responses, case["x_low"], case["x_high"], case["model"], case["mu"], case["sigma"])
    if "expected" in case:
        assert next_level == pytest.approx(case["expected"])
    else:
        low, high = case["expected_between"]
        assert low - 1.0e-12 <= next_level <= high + 1.0e-12


def test_doptimal_next_within_bounds() -> None:
    """方法104 D-优化法：任意候选均 clip 在初始界内（补充固定全区间设计路径）."""
    levels = np.array([5.0, 3.0, 7.0])
    responses = np.array([0, 0, 1])
    next_level = doptimal_next(levels, responses, 0.0, 10.0, "normal", 5.0, 1.5)
    assert 0.0 <= next_level <= 10.0


@pytest.mark.parametrize("case", _designer_data["probit"], ids=_ids(_designer_data["probit"]))
def test_probit_levels_data_cases(case: dict) -> None:
    if "raises" in case:
        with pytest.raises(ReliabilityError, match=case["raises"]):
            probit_levels(case["x_low"], case["x_high"], case["n_levels"])
    else:
        np.testing.assert_allclose(probit_levels(case["x_low"], case["x_high"], case["n_levels"]), case["expected"])


@pytest.mark.parametrize("case", _designer_data["stepstress"], ids=_ids(_designer_data["stepstress"]))
def test_stepstress_levels_data_cases(case: dict) -> None:
    counts = None
    if case["counts"] is not None:
        table = {float(level): int(hits) for level, hits in case["counts"].items()}

        def counts(x: float, table: dict[float, int] = table) -> int:
            return table[float(x)]

    kwargs = {
        "x_low": case["x_low"],
        "x_high": case["x_high"],
        "step": case["step"],
        "n_per_level": case["n_per_level"],
    }
    if "raises" in case:
        with pytest.raises(ReliabilityError, match=case["raises"]):
            stepstress_levels(**kwargs, counts=counts)
        return
    levels, hits = stepstress_levels(**kwargs, counts=counts)
    np.testing.assert_allclose(levels, case["expected_levels"])
    np.testing.assert_array_equal(hits, case["expected_hits"])
    # 停止规则：全响应出现后恰好多试一级
    n_per_level = case["n_per_level"]
    full = [i for i, value in enumerate(hits) if value >= n_per_level]
    if full and counts is not None:
        assert full[-1] == len(hits) - 2
