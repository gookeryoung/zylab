"""sci.plotting 绘图事件测试."""

from __future__ import annotations

from zylab.core import EventBus
from zylab.sci import TOPIC_PLOT_REQUESTED, PlotRequest, make_plot_function


def test_plot_publishes_event() -> None:
    """plot 应发布 PlotRequest 事件并携带全部参数."""
    bus = EventBus()
    received: list[PlotRequest] = []
    bus.subscribe(TOPIC_PLOT_REQUESTED, received.append)
    plot = make_plot_function(bus)
    plot([1, 2, 3], [4, 5, 6], title="标题", xlabel="x轴", ylabel="y轴", clear=True)
    assert len(received) == 1
    req = received[0]
    assert list(req.x) == [1, 2, 3]
    assert list(req.y) == [4, 5, 6]
    assert req.title == "标题"
    assert req.xlabel == "x轴"
    assert req.ylabel == "y轴"
    assert req.clear is True


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
