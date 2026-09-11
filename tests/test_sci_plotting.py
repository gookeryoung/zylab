"""sci.plotting 绘图事件与 matplotlib rcParams 默认配置测试."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from zylab.core import EventBus
from zylab.sci import (
    TOPIC_PLOT_REQUESTED,
    PlotRequest,
    apply_matplotlib_defaults,
    make_plot_function,
)


def test_plot_publishes_event() -> None:
    """plot 应发布 PlotRequest 事件并携带全部参数."""
    bus = EventBus()
    received: list[PlotRequest] = []
    bus.subscribe(TOPIC_PLOT_REQUESTED, received.append)
    plot = make_plot_function(bus)
    plot([1, 2, 3], [4, 5, 6], title="标题", xlabel="x轴", ylabel="y轴", clear=True, label="曲线A")
    assert len(received) == 1
    req = received[0]
    assert list(req.x) == [1, 2, 3]
    assert list(req.y) == [4, 5, 6]
    assert req.title == "标题"
    assert req.xlabel == "x轴"
    assert req.ylabel == "y轴"
    assert req.clear is True
    assert req.extra == {"label": "曲线A"}


def test_plot_y_only_generates_x() -> None:
    """单参数调用时 x 自动生成 0..n-1."""
    bus = EventBus()
    received: list[PlotRequest] = []
    bus.subscribe(TOPIC_PLOT_REQUESTED, received.append)
    make_plot_function(bus)([10, 20, 30])
    assert list(received[0].x) == [0, 1, 2]
    assert list(received[0].y) == [10, 20, 30]


def test_plot_without_subscriber_is_noop() -> None:
    """无订阅者时 plot 静默通过（CLI/worker 场景）."""
    make_plot_function(EventBus())([1.0, 2.0])  # 不抛异常


# ---------------------------------------------------------------------------
# matplotlib rcParams 全局默认配置


@pytest.fixture(autouse=True)
def _reset_mpl_rc() -> None:
    """每个测试前恢复 matplotlib rcParams 默认值，测试后原样还原."""
    plt.rcParams.update(plt.rcParamsDefault)
    yield
    plt.rcParams.update(plt.rcParamsDefault)


def test_apply_defaults_sets_grid_on() -> None:
    """应用默认后 axes.grid 应开启（matplotlib 默认 False）."""
    assert plt.rcParams["axes.grid"] is False  # 前置条件确认
    apply_matplotlib_defaults()
    assert plt.rcParams["axes.grid"] is True


def test_apply_defaults_sets_linewidth() -> None:
    """应用默认后 lines.linewidth 应为 2.0（科学计算更醒目）."""
    apply_matplotlib_defaults()
    assert plt.rcParams["lines.linewidth"] == 2.0


def test_apply_defaults_sets_font_size() -> None:
    """应用默认后 font.size 应为 11.0."""
    apply_matplotlib_defaults()
    assert plt.rcParams["font.size"] == 11.0


def test_apply_defaults_sets_dpi() -> None:
    """应用默认后 figure.dpi 应为 120."""
    apply_matplotlib_defaults()
    assert plt.rcParams["figure.dpi"] == 120.0


def test_apply_defaults_sets_unicode_minus_false() -> None:
    """应用默认后 axes.unicode_minus 应为 False（避免中文环境负号变方块）."""
    apply_matplotlib_defaults()
    assert plt.rcParams["axes.unicode_minus"] is False


def test_apply_defaults_injects_ns_vars() -> None:
    """传入命名空间时应注入 available_fonts 与 cn_font_candidates."""
    ns: dict[str, object] = {}
    apply_matplotlib_defaults(ns)
    assert "available_fonts" in ns
    assert "cn_font_candidates" in ns
    assert isinstance(ns["available_fonts"], frozenset)
    assert isinstance(ns["cn_font_candidates"], list)


def test_apply_defaults_idempotent() -> None:
    """幂等性：重复调用不产生额外副作用."""
    apply_matplotlib_defaults()
    rc_after_first = dict(plt.rcParams)
    apply_matplotlib_defaults()
    for key in ("axes.grid", "lines.linewidth", "font.size", "figure.dpi"):
        assert plt.rcParams[key] == rc_after_first[key]


def test_apply_defaults_returns_fonts() -> None:
    """返回值应为系统已安装字体集合（matplotlib 可用时）."""
    result = apply_matplotlib_defaults()
    assert result is not None
    assert isinstance(result, frozenset)
    assert len(result) > 0


def test_apply_defaults_updates_grid_visuals() -> None:
    """grid 相关视觉参数应合理（alpha/color 兼顾可读性与不抢眼）."""
    apply_matplotlib_defaults()
    assert plt.rcParams["grid.alpha"] == pytest.approx(0.3)
    assert plt.rcParams["grid.linewidth"] == 0.8
    assert plt.rcParams["grid.color"] == "#cccccc"
