"""zylab.sci 绘图接口（Qt-free）.

``plot`` 不直接操作任何 GUI：把绘图请求作为事件发布到总线（``sci.plot.requested``），
由 GUI 层订阅渲染（pyqtgraph），CLI/worker 场景无订阅者时静默通过。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from zylab.core.events import EventBus

__all__ = ["TOPIC_PLOT_REQUESTED", "PlotRequest", "make_plot_function"]

TOPIC_PLOT_REQUESTED = "sci.plot.requested"


@dataclass(frozen=True)
class PlotRequest:
    """绘图请求（同进程事件载荷，持有数组引用不拷贝）.

    :param x: 横轴数据。
    :param y: 纵轴数据（与 x 等长）。
    :param title: 图标题。
    :param xlabel: 横轴标签。
    :param ylabel: 纵轴标签。
    :param clear: True 时清空已有曲线，False 时叠加。
    """

    x: Any
    y: Any
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    clear: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def make_plot_function(bus: EventBus) -> Any:
    """构建绑定事件总线的 ``plot`` 函数（注入 REPL 命名空间用）.

    用法（控制台内）::

        plot(x, y)
        plot(y)                  # x 自动取 0..n-1
        plot(x, y, title="正弦", xlabel="t", ylabel="v", clear=True)
    """

    def plot(  # noqa: PLR0913
        x: Any, y: Any = None, *, title: str = "", xlabel: str = "", ylabel: str = "", clear: bool = False
    ) -> None:
        """发布绘图请求事件（无订阅者时静默）."""
        if y is None:
            y = x
            x = np.arange(len(y))
        bus.publish(
            TOPIC_PLOT_REQUESTED,
            PlotRequest(x=np.asarray(x), y=np.asarray(y), title=title, xlabel=xlabel, ylabel=ylabel, clear=clear),
        )

    return plot
