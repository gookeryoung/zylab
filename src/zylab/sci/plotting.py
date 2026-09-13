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
from pathlib import Path
from typing import Any

import numpy as np

from zylab.core.events import EventBus
from zylab.sci.palettes import CURVE_PALETTE

__all__ = [
    "TOPIC_PLOT_REQUESTED",
    "PlotRequest",
    "apply_matplotlib_defaults",
    "fig_to_png_bytes",
    "make_plot_function",
    "plot_band",
    "plot_reg",
    "save_figure",
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

#: seaborn set_theme 的 rc 覆盖项（统一的 whitegrid 视觉规范）.
#:
#: 包含网格样式（虚线 + alpha）、去除上右边框、字号层级、图例外观。
#: 这些项放进 set_theme 的 rc 参数而非 plt.rcParams.update，
#: 避免 set_theme 在后续步骤里再次重置。
_SEABORN_THEME_RC: dict[str, Any] = {
    # --- 网格 ---
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.35,
    # --- 边框（whitegrid 标志性：去上右边框） ---
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    # --- 字号（context=notebook 默认 12pt，细化层级） ---
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    # --- 图例（半透明白底 + 细边框） ---
    "legend.frameon": True,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "#DDDDDD",
}

#: seaborn set_theme 之后叠加的精简 rcParams（zylab 特有保障项）.
#:
#: seaborn whitegrid + _SEABORN_THEME_RC 已接管 grid / font / legend / spines，
#: 这里仅保留 zylab 必须覆盖的项：线条粗细、DPI、负号显示（set_theme 会重置）.
_MPL_RC_OVERRIDES: dict[str, Any] = {
    # --- 线条 ---
    "lines.linewidth": 2.0,
    "lines.markersize": 6.0,
    # --- DPI ---
    "figure.dpi": 120,
    "savefig.dpi": 150,
    # --- 负号（关键：set_theme 会重置为 True，必须在其后再次覆盖） ---
    "axes.unicode_minus": False,
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
        import matplotlib.pyplot as plt  # pragma: no cover - 硬依赖
        from matplotlib import font_manager  # pragma: no cover - 硬依赖
    except ImportError:  # pragma: no cover - 硬依赖永不触发
        logger.info("matplotlib 未安装，跳过 rcParams 配置")
        return None
    # --- 1. seaborn 主题（硬依赖，pyproject.toml 已声明） ---
    import seaborn as sns

    sns.set_theme(
        style="whitegrid",
        context="notebook",
        palette=list(CURVE_PALETTE),
        color_codes=False,  # 保持 matplotlib 单字母色码经典行为
        rc=_SEABORN_THEME_RC,  # 统一的 whitegrid 视觉规范：网格/边框/字号/图例
    )
    # --- 2. 叠加 zylab 保障项（含 unicode_minus + 线条/DPI，set_theme 会重置这些） ---
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


def plot_band(  # noqa: PLR0913  # 用户 API：参数多为方便一次性调整
    ys: Any,
    x: Any | None = None,
    *,
    label: str = "",
    color: str | None = None,
    ci: str = "sd",
    ax: Any | None = None,
) -> tuple[Any, Any, Any]:
    """多组 y 值的均值曲线 ± 置信带（seaborn lineplot 风格）.

    工程仿真中多次试验/扰动分析的常用快捷函数：传入形状 ``(n_trials, n_points)``
    的二维数组或等长列表列表，自动计算均值 + 标准差（或 95% 置信区间）带。

    :param ys: 多组纵轴数据，形状 ``(n_trials, n_points)``。一维数组视为单组
        退化为普通折线（等同 ``plt.plot``）。
    :param x: 横轴数据，长度须等于 ``ys`` 的列数（或单组时长度）。缺省取
        ``0..n_points-1``。
    :param label: 图例名（空字符串不画图例标签）。
    :param color: 显式曲线颜色（hex 或语义色名），缺省按 CURVE_PALETTE 循环。
    :param ci: 置信带类型，``"sd"``（均值 ± 标准差，默认）或 ``"95"``
        （均值 ± 95% t 分布置信区间）。
    :param ax: matplotlib Axes 对象，缺省取 ``plt.gca()``。
    :returns: ``(mean_line, lower_band, upper_band)`` —— 均值线 + 带的两条边界
        （seaborn lineplot 返回值风格，便于事后调样式）。
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    ys_arr = np.asarray(ys, dtype=float)
    if ys_arr.ndim == 1:
        # 单组：退化为普通折线（seaborn lineplot 只画一条线，无带）
        if x is None:
            x_arr = np.arange(len(ys_arr))
        else:
            x_arr = np.asarray(x, dtype=float)
        (mean_line,) = sns.lineplot(x=x_arr, y=ys_arr, ax=ax, label=label, color=color).get_lines()
        return mean_line, None, None
    # 多组：按列聚合
    n_trials, n_points = ys_arr.shape
    if x is None:
        x_arr = np.arange(n_points)
    else:
        x_arr = np.asarray(x, dtype=float)
        if len(x_arr) != n_points:
            raise ValueError(f"x 长度 {len(x_arr)} 与 ys 列数 {n_points} 不匹配")
    if ci == "sd":
        y_mean = ys_arr.mean(axis=0)
        y_std = ys_arr.std(axis=0, ddof=1)
        y_lo, y_hi = y_mean - y_std, y_mean + y_std
    elif ci == "95":
        y_mean = ys_arr.mean(axis=0)
        from scipy import stats  # 懒加载，不增加硬依赖

        se = stats.sem(ys_arr, axis=0)
        h = se * stats.t.ppf(0.975, df=n_trials - 1)
        y_lo, y_hi = y_mean - h, y_mean + h
    else:
        raise ValueError(f"ci 须为 'sd' 或 '95'，收到 {ci!r}")
    ax_target = ax if ax is not None else plt.gca()
    (mean_line,) = ax_target.plot(x_arr, y_mean, label=label, color=color)
    band_color = color if color is not None else mean_line.get_color()
    fill = ax_target.fill_between(x_arr, y_lo, y_hi, color=band_color, alpha=0.15)
    return mean_line, fill, None


def plot_reg(  # noqa: PLR0913  # 用户 API：参数多为方便一次性调整
    x: Any,
    y: Any,
    *,
    order: int = 1,
    ci: int = 95,
    color: str | None = None,
    label: str = "",
    ax: Any | None = None,
) -> Any:
    """散点 + 回归线 + 置信带（seaborn regplot 风格）.

    :param x: 横轴数据。
    :param y: 纵轴数据（与 x 等长）。
    :param order: 多项式阶数（1=线性，2=二次）。
    :param ci: 置信区间百分比（默认 95）。
    :param color: 回归线颜色（hex 或语义色名），缺省按 CURVE_PALETTE 循环。
    :param label: 图例名。
    :param ax: matplotlib Axes 对象，缺省取 ``plt.gca()``。
    :returns: seaborn Axes 对象（可继续叠加元素）。
    """
    import seaborn as sns

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    return sns.regplot(
        x=x_arr,
        y=y_arr,
        order=order,
        ci=ci,
        color=color,
        label=label or None,
        ax=ax,
    )


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


# ---------------------------------------------------------------- 报告层导出


def save_figure(
    fig: Any,
    path: str | Path,
    *,
    dpi: int = 150,
    bbox_inches: str | None = "tight",
    **kwargs: Any,
) -> Path:
    """把 matplotlib Figure 导出为图像文件（PNG / PDF / SVG，按后缀自动识别）.

    复用已配置的中文字体（调用方须先跑 :func:pply_matplotlib_defaults，
    通常 notebook 启动时已自动调用）。导出 PNG 时通过临时 BytesIO 中转，
    避免磁盘中间态残留。

    Args:
        fig: matplotlib Figure 对象（非 Figure 则 TypeError）。
        path: 输出路径，后缀决定格式：.png / .pdf / .svg。
        dpi: 图像分辨率，默认 150（屏幕报告 150 足够，印刷建议 300）。
        bbox_inches: 裁剪方式，默认 "tight" 紧贴内容；None 保留完整画布。
        **kwargs: 透传给 `fig.savefig`（如 facecolor / transparent / pad_inches）。

    Returns:
        写入完成的 `path`（Path 类型，与入参类型一致）。

    Raises:
        TypeError: `fig` 非 matplotlib Figure。
        OSError: 目标目录不可写。
    """
    import matplotlib.figure  # 懒加载

    if not isinstance(fig, matplotlib.figure.Figure):
        raise TypeError(f"save_figure: fig 必须是 matplotlib Figure，收到 {type(fig).__name__}")

    from pathlib import Path as _Path

    p = _Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p), dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    logger.info("matplotlib figure 已导出: %s (%ddpi)", p, dpi)
    return p


def fig_to_png_bytes(fig: Any, *, dpi: int = 150) -> bytes:
    """把 matplotlib Figure 渲染为 PNG 字节串（零磁盘 I/O，供报告嵌入 / clipboard）.

    Args:
        fig: matplotlib Figure 对象。
        dpi: 渲染分辨率，默认 150。

    Returns:
        PNG 图像字节串（可直接写入文件或转 QImage）。

    Raises:
        TypeError: `fig` 非 matplotlib Figure。
    """
    import io

    import matplotlib.figure  # 懒加载

    if not isinstance(fig, matplotlib.figure.Figure):
        raise TypeError(f"fig_to_png_bytes: fig 必须是 matplotlib Figure，收到 {type(fig).__name__}")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()
