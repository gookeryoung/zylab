"""gui.pages.plot_page 绘图页测试."""

from __future__ import annotations

import numpy as np
import pytest

from zylab.core import EventBus
from zylab.gui.pages.plot_page import PlotPage
from zylab.sci import TOPIC_PLOT_REQUESTED, PlotRequest


@pytest.mark.gui
def test_plot_page_renders_on_event(qtbot) -> None:
    """发布绘图事件应渲染曲线并发出 plot_shown 信号."""
    bus = EventBus()
    page = PlotPage(bus)
    qtbot.addWidget(page)
    with qtbot.waitSignal(page.plot_shown, timeout=1000):
        bus.publish(
            TOPIC_PLOT_REQUESTED,
            PlotRequest(x=np.array([0.0, 1.0]), y=np.array([1.0, 0.0]), title="标题", xlabel="x轴", ylabel="y轴"),
        )
    assert len(page._plot.listDataItems()) == 1


@pytest.mark.gui
def test_plot_page_clear(qtbot) -> None:
    """clear=True 应先清空已有曲线."""
    bus = EventBus()
    page = PlotPage(bus)
    qtbot.addWidget(page)
    req = PlotRequest(x=np.array([0.0, 1.0]), y=np.array([1.0, 2.0]))
    bus.publish(TOPIC_PLOT_REQUESTED, req)
    bus.publish(TOPIC_PLOT_REQUESTED, req)
    assert len(page._plot.listDataItems()) == 2
    bus.publish(
        TOPIC_PLOT_REQUESTED,
        PlotRequest(x=np.array([0.0, 1.0]), y=np.array([2.0, 1.0]), clear=True),
    )
    assert len(page._plot.listDataItems()) == 1
