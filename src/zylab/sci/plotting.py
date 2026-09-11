"""zylab.sci 绘图接口（Qt-free）.

两层绘图管线：

- **matplotlib rcParams 全局配置**（:func:`apply_matplotlib_defaults`）：
  为笔记本 REPL 环境内置合理的科学计算默认样式，通过 seaborn
  ``whitegrid`` 主题 + 定制色环一次性设置网格、线条、字号、DPI、
  中文字体与曲线循环色，用户 ``import matplotlib.pyplot as plt``
  后直接即可获得较好的曲线显示效果。
- **plot 事件管线**（``plot()`` → ``sci.plot.requested`` → GUI 渲染）：
  ``plot`` 不直接操作任何 GUI，把绘图请求发布到事件总线，
  由 GUI 层订阅渲染（pyqtgraph），CLI/worker 场景无订阅者时静默通过。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from zylab.core.events import EventBus
from zylab.sci.palettes import CURVE_PALETTE

__all__ = [
    "TOPIC_PLOT_REQUESTED",
    "PlotRequest",
    "apply_matplotlib_defaults",
    "make_plot_function",
]

logger = logging.getLogger(__name__)

TOPIC_PLOT_REQUESTED = "sci.plot.requested"

#: 跨平台中文字体候选列表（按优先级排序，前置到 matplotlib 默认 sans-serif 链）
_CN_FONT_CANDIDATES: list[str] = [
    "DengXian",
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "SimHei",
    "KaiTi",
    "STKaiti",
    "FangSong",
    "STFangsong",
    "PingFang SC",
    "Heiti SC",
    "Hiragino Sans GB",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
]

#: seaborn set_theme 之后叠加的精简 rcParams（zylab 特有保障项）.
#:
#: seaborn whitegrid 已接管 grid / font / axes.linewidth，
#: 这里仅保留 zylab 必须覆盖的项：线条粗细、DPI、负号显示、图例.
_MPL_RC_OVERRIDES: dict[str, Any] = {
    # --- 线条 ---
    "lines.linewidth": 2.0,
    "lines.markersize": 6.0,
    # --- DPI ---
    "figure.dpi": 120,
    "savefig.dpi": 150,
    # --- 图例 ---
    "legend.fontsize": "medium",
    "legend.frameon": True,
    "legend.framealpha": 0.85,
    # --- 负号 ---
    "axes.unicode_minus": False,  # 关键：set_theme 会重置为 True
}


def apply_matplotlib_defaults(ns: dict[str, Any] | None = None) -> frozenset[str] | None:
    """应用 zylab 内置 matplotlib rcParams 全局默认值.

    在笔记本 REPL 场景下由内核启动时自动调用；CLI/worker 场景用户可手动
    调用。本函数幂等，重复调用不产生额外副作用。

    应用顺序（关键，顺序错会导致覆盖）：

    1. seaborn ``set_theme(style="whitegrid", context="notebook",
       palette=CURVE_PALETTE, color_codes=False)`` —— 接管网格/字号/色环.
    2. 叠加 ``_MPL_RC_OVERRIDES`` —— 覆盖 set_theme 重置的
       ``unicode_minus`` 并保障 zylab 特有线条/DPI.
    3. 前置中文字体候选链 —— set_theme 也会重写 ``font.sans-serif``，
       必须在其后合并，否则中文仍会乱码.

    :param ns: 可选的命名空间字典，若提供则注入
        ``available_fonts``（系统已安装字体名集合）、
        ``cn_font_candidates``（候选列表）与 ``curve_palette``.
    :returns: matplotlib 不可用时返回 ``None``；可用时返回系统已安装字体集.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError:
        logger.info("matplotlib 未安装，跳过 rcParams 配置")
        return None
    # --- 1. seaborn 主题 ---
    try:
        import seaborn as sns
    except ImportError:
        logger.info("seaborn 未安装，回退到基础 rcParams")
        plt.rcParams.update(_MPL_RC_OVERRIDES)
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.alpha"] = 0.3
        plt.rcParams["grid.color"] = "#cccccc"
        plt.rcParams["grid.linewidth"] = 0.8
    else:
        sns.set_theme(
            style="whitegrid",
            context="notebook",
            palette=list(CURVE_PALETTE),
            color_codes=False,  # 保持 matplotlib 单字母色码经典行为
        )
        # --- 2. 叠加 zylab 保障项（含 unicode_minus） ---
        plt.rcParams.update(_MPL_RC_OVERRIDES)
    # --- 3. 中文字体（最后合并，set_theme 会重写 font.sans-serif） ---
    available: set[str] = {f.name for f in font_manager.fontManager.ttflist}
    selected_cn = [name for name in _CN_FONT_CANDIDATES if name in available]
    if selected_cn:
        default_sans = list(plt.rcParams.get("font.sans-serif", []))
        merged: list[str] = []
        seen: set[str] = set()
        for name in selected_cn + default_sans:
            if name not in seen:
                merged.append(name)
                seen.add(name)
        plt.rcParams["font.sans-serif"] = merged
    # --- 4. 注入命名空间 ---
    if ns is not None:
        ns["available_fonts"] = frozenset(available)
        ns["cn_font_candidates"] = list(_CN_FONT_CANDIDATES)
        ns["curve_palette"] = list(CURVE_PALETTE)
    return frozenset(available)


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
        plot(x, y, title="正弦", xlabel="t", ylabel="v", clear=True, label="sin")
    """

    def plot(  # noqa: PLR0913
        x: Any,
        y: Any = None,
        *,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        clear: bool = False,
        label: str = "",
    ) -> None:
        """发布绘图请求事件（无订阅者时静默）.

        :param label: 曲线图例名（同一单元多次 ``plot`` 合并为多曲线单图时用于区分）.
        """
        if y is None:
            y = x
            x = np.arange(len(y))
        bus.publish(
            TOPIC_PLOT_REQUESTED,
            PlotRequest(
                x=np.asarray(x),
                y=np.asarray(y),
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                clear=clear,
                extra={"label": label},
            ),
        )

    return plot
