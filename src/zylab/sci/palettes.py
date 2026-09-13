"""zylab 统一曲线色板（Qt-free、零重依赖）.

全链路唯一取色源：matplotlib/seaborn 主题（sci.plotting）、pyqtgraph 曲线
（gui.pages.notebook_page / gui.widgets.dsl_result_view）与报告 SVG 曲线
（flowchart.report）共用同一循环色与语义色，保证三条渲染路径视觉一致。

本模块只含纯常量与纯函数，不 import seaborn/matplotlib/Qt —— flowchart 报告
构建路径必须保持轻量。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CURVE_PALETTE",
    "PG_CURVE_DEFAULTS",
    "SEMANTIC_CURVE_COLORS",
    "resolve_curve_color",
]

#: 曲线循环色环（6 色，首色为 GUI 语义 primary 靛蓝族）
#:
#: 与 GUI Palette 语义色协调（primary/success/warning/danger 系）并补青、紫
#: 两色扩展循环长度；中间明度选取，浅色/深色主题背景与白底报告均可读。
CURVE_PALETTE: tuple[str, ...] = (
    "#3C2ECA",  # 靛蓝（primary 族）
    "#10B981",  # 绿（success 族）
    "#F59E0B",  # 琥珀（warning 族）
    "#EF4444",  # 红（danger 族）
    "#06B6D4",  # 青
    "#8B5CF6",  # 紫
)

#: 语义色名 → 十六进制（DSL ``series_styles`` 的 ``color`` 字段取值域）
SEMANTIC_CURVE_COLORS: dict[str, str] = {
    "primary": "#3C2ECA",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "info": "#3B82F6",
}

#: pyqtgraph 曲线视觉默认值（对齐 seaborn whitegrid 风格，纯常量无 Qt 依赖）.
#:
#: notebook_page / dsl_result_view 共享此配置，保证 GUI 内嵌曲线视觉一致。
#: 与 matplotlib seaborn whitegrid 的对应关系见 :mod:`zylab.sci.plotting`.
PG_CURVE_DEFAULTS: dict[str, Any] = {
    "grid_color": "#E8E8E8",  # seaborn whitegrid grid.alpha=0.35 的等效灰色
    "grid_alpha": 0.35,  # 网格透明度（比默认 0.3 稍亮）
    "axis_width": 0.8,  # 轴线宽度（seaborn axes.linewidth）
    "curve_width": 2.0,  # 曲线宽度（seaborn lines.linewidth）
    "legend_bg_alpha": 200,  # 图例背景 alpha（0-255，半透明白）
    "legend_border": "#DDDDDD",  # 图例边框色
    "legend_offset": (8, 8),  # 图例右上偏移
}


def resolve_curve_color(spec: str | None, index: int) -> str:
    """解析曲线颜色：语义名 → hex，``#`` 开头直通，缺省按索引循环取色.

    :param spec: 颜色规格（语义色名或 ``#RRGGBB``），None/空表示未指定。
    :param index: 序列索引（未指定颜色时用于色环循环）。
    :returns: 十六进制颜色字符串。
    """
    if spec:
        if spec.startswith("#"):
            return spec
        return SEMANTIC_CURVE_COLORS.get(spec, spec)
    return CURVE_PALETTE[index % len(CURVE_PALETTE)]
