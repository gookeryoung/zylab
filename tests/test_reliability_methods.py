"""zylab.reliability.methods 试验设计器测试：下一水平规则与设计表."""

from __future__ import annotations

import numpy as np
import pytest

from zylab.reliability.errors import ReliabilityError
from zylab.reliability.methods import (
    doptimal_next,
    langlie_next,
    neyer_next,
    probit_levels,
    stepstress_levels,
    updown_next,
)


def test_langlie_first_level_is_midpoint() -> None:
    """无历史记录：兰利法取初始区间中点."""
    x = langlie_next(np.asarray([]), np.asarray([], dtype=int), 6.0, 14.0)
    assert x == pytest.approx(10.0)


def test_langlie_midpoint_of_mixed_interval() -> None:
    """有记录：取最大全不响应水平与最小全响应水平的中点.

    历史试验：x=[8(0), 12(1)]，则 D0=8（≤8 全不响应）、D1=12（≥12 全响应），
    下一水平 = 10。
    """
    x = langlie_next(np.asarray([8.0, 12.0]), np.asarray([0, 1]), 6.0, 14.0)
    assert x == pytest.approx(10.0)


def test_langlie_partial_records_fall_back_to_bounds() -> None:
    """无全响应水平：D1 取初始上界（区间继续上探）."""
    # 全部不响应 → D0 = 10（最大水平，≤10 全不响应），D1 = 14（初始上界）
    x = langlie_next(np.asarray([10.0, 9.0]), np.asarray([0, 0]), 6.0, 14.0)
    assert x == pytest.approx(12.0)


def test_updown_step_direction() -> None:
    """升降法：响应降级、不响应升级，初始取区间中点."""
    empty = (np.asarray([]), np.asarray([], dtype=int))
    assert updown_next(*empty, 6.0, 14.0, step=1.0) == pytest.approx(10.0)
    assert updown_next(np.asarray([10.0]), np.asarray([1]), 6.0, 14.0, step=1.5) == pytest.approx(8.5)
    assert updown_next(np.asarray([10.0]), np.asarray([0]), 6.0, 14.0, step=1.5) == pytest.approx(11.5)


def test_updown_rejects_nonpositive_step() -> None:
    """非正步长报 ReliabilityError."""
    with pytest.raises(ReliabilityError, match="步长须为正"):
        updown_next(np.asarray([10.0]), np.asarray([1]), 6.0, 14.0, step=0.0)


def test_neyer_first_level_is_midpoint() -> None:
    """无历史记录：Neyer 法取初始区间中点."""
    x = neyer_next(np.asarray([]), np.asarray([], dtype=int), 6.0, 14.0, "logistic", 10.0, 1.0)
    assert x == pytest.approx(10.0)


def test_neyer_exponential_probe_upward() -> None:
    """全不响应：向对侧以 σ·2^(m/2) 指数步长上探（m=2 → 步长 2σ）."""
    levels = np.asarray([10.0, 11.9])
    responses = np.asarray([0, 0])
    x = neyer_next(levels, responses, 6.0, 14.0, "logistic", 10.0, 1.3)
    assert x == pytest.approx(11.9 + 1.3 * 2.0)


def test_neyer_exponential_probe_downward() -> None:
    """全响应：向对侧下探（m=1 → 步长 σ·√2）."""
    levels = np.asarray([10.0])
    responses = np.asarray([1])
    x = neyer_next(levels, responses, 6.0, 14.0, "logistic", 10.0, 2.0)
    assert x == pytest.approx(10.0 - 2.0 * np.sqrt(2.0))


def test_neyer_mixed_matches_doptimal() -> None:
    """响应已翻转：与 D-优化法同一 D-最优选点规则."""
    levels = np.asarray([9.0, 11.0])
    responses = np.asarray([0, 1])
    a = neyer_next(levels, responses, 6.0, 14.0, "logistic", 10.0, 1.0)
    b = doptimal_next(levels, responses, 6.0, 14.0, "logistic", 10.0, 1.0)
    assert a == pytest.approx(b)


def test_probit_levels_uniform_grid() -> None:
    """概率单位法水平表：等间隔覆盖区间."""
    levels = probit_levels(6.0, 14.0, 5)
    np.testing.assert_allclose(levels, [6.0, 8.0, 10.0, 12.0, 14.0])


def test_probit_rejects_single_level() -> None:
    """水平数 < 2 报 ReliabilityError."""
    with pytest.raises(ReliabilityError, match="水平数须"):
        probit_levels(6.0, 14.0, 1)


def test_stepstress_stops_one_level_after_full_response() -> None:
    """完全步进法：首个全响应水平（计数=发数）后再补一级确认停止.

    响应回调：x=11 首次全响应（10/10）→ 试验水平为 [6,7,8,9,10,11,12]。
    """
    hits = {11.0: 10}
    levels, counts = stepstress_levels(6.0, 20.0, 1.0, 10, lambda x: hits.get(x, 0))
    np.testing.assert_allclose(levels, [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    assert counts.tolist() == [0, 0, 0, 0, 0, 10, 0]


def test_stepstress_without_hits_runs_to_upper_bound() -> None:
    """始终无响应：步进试至上界为止."""
    levels, counts = stepstress_levels(6.0, 9.0, 1.0, 5, lambda _x: 0)
    np.testing.assert_allclose(levels, [6.0, 7.0, 8.0, 9.0])
    assert counts.tolist() == [0, 0, 0, 0]


def test_stepstress_preview_without_callback() -> None:
    """缺省回调（设计预览）：全 0 计数试到上界."""
    levels, counts = stepstress_levels(0.0, 3.0, 1.0, 4)
    np.testing.assert_allclose(levels, [0.0, 1.0, 2.0, 3.0])
    assert counts.tolist() == [0, 0, 0, 0]


def test_stepstress_rejects_bad_step() -> None:
    """非正步长/发数报 ReliabilityError."""
    with pytest.raises(ReliabilityError, match="步长须为正"):
        stepstress_levels(6.0, 14.0, 0.0, 10)
    with pytest.raises(ReliabilityError, match="发数须"):
        stepstress_levels(6.0, 14.0, 1.0, 0)
