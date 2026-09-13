"""DSL 结果视图：按标准化视图数据（Curve/Table/Text）渲染单页结果.

与 :class:`~zylab.gui.widgets.result_view.ResultView`（解对象 -> 云图/
振型等类型化渲染）互补：本视图消费 DSL ``results`` 声明解析出的
:data:`~zylab.flowchart.results.ViewData`（曲线/表格/文本），供参数化计算应用页
（P6）按结果页签组装；云图声明（CloudData）由参数化计算应用页路由到既有
ResultView，本视图仅显示占位说明。

同组多结果（DSL ``group`` 声明）由 :class:`DslGroupedResultView` 合并
为单页分块渲染（曲线定高/表格限高滚动/文本自动换行），页内滚动查看，
避免内容少的结果各占一页。

本模块曲线视图使用统一组件 :class:`ZyPlotWidget`（预配置主题背景/网格/
轴色/图例 + 可组合中文右键菜单），并提供 ``build_curve_widget`` 工厂函数
为单页/分组页共同调用，保证视觉一致性。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pyqtgraph as pg

from zylab.flowchart.results import CloudData, CurveData, TableData, TextData, ViewData
from zylab.sci.palettes import CURVE_PALETTE, PG_CURVE_DEFAULTS, resolve_curve_color

from .. import theme
from ..qt_compat import (
    QGroupBox,
    QHeaderView,
    QLabel,
    QScrollArea,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from .plot_widget import PlotMenuConfig, ZyPlotWidget

__all__ = ["DslGroupedResultView", "DslResultView"]

#: 分组页内曲线块高度（px，同页多块时固定高度避免挤占）
_GROUPED_CURVE_HEIGHT = 300

#: 分组页内表格块高度上限（px，超出内部滚动）
_GROUPED_TABLE_MAX_HEIGHT = 320


class DslResultView(QWidget):  # pragma: no cover
    """DSL 结果单页视图（curve/table/text 分发渲染）."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化：标题行 + 占位正文（set_data 时按类型重建）."""
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD)
        self._layout.setSpacing(theme.SPACING_SM)
        self._title = QLabel("", objectName="resultTitle")
        self._layout.addWidget(self._title)
        self._body: QWidget | None = None
        self._show_placeholder("尚未运行")

    def set_data(self, data: ViewData) -> None:
        """按视图数据类型重建正文."""
        self._title.setText(data.title)
        self._clear_body()
        if isinstance(data, CurveData):
            self._body = build_curve_widget(data)
        elif isinstance(data, TableData):
            self._body = build_table_widget(data)
        elif isinstance(data, TextData):
            self._body = build_text_widget(data)
        else:
            self._show_placeholder(_cloud_hint(data))
        if self._body is not None:
            self._layout.addWidget(self._body, stretch=1)

    def set_error(self, message: str) -> None:
        """显示结果解析错误（引用未运行/路径不存在等，标题保持页签名）."""
        self._clear_body()
        self._body = QLabel(message, objectName="errorText")
        self._body.setWordWrap(True)
        self._layout.addWidget(self._body, stretch=1)

    # ------------------------------------------------------------------ 内部

    def _clear_body(self) -> None:
        """移除旧正文控件."""
        if self._body is not None:
            self._layout.removeWidget(self._body)
            self._body.deleteLater()
            self._body = None

    def _show_placeholder(self, text: str) -> None:
        """显示占位正文."""
        self._clear_body()
        self._body = QLabel(text, objectName="secondaryText")
        self._body.setWordWrap(True)
        self._layout.addWidget(self._body, stretch=1)


class DslGroupedResultView(QWidget):  # pragma: no cover
    """DSL 分组结果页：同组多结果按类别分块纵向排列（单页紧凑视图）.

    每块为 QGroupBox（组名 = 结果声明 title），正文按视图数据类型
    渲染；块级解析失败显示错误文本块，不影响其余块。整页置于
    滚动区中，块自顶向下堆叠。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化：滚动区 + 分块容器（set_data 时重建）."""
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.NoFrame)
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD)
        self._layout.setSpacing(theme.SPACING_MD)
        self._scroll.setWidget(self._container)
        root.addWidget(self._scroll)
        self._placeholder = QLabel("尚未运行", objectName="secondaryText")
        self._placeholder.setWordWrap(True)
        self._layout.addWidget(self._placeholder)

    def set_data(self, blocks: list[tuple[str, ViewData | str]]) -> None:
        """按 ``(块标题, 视图数据或错误消息)`` 序列重建分块正文.

        :param blocks: 块声明序列；错误消息（str）渲染为错误文本块。
        """
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for title, payload in blocks:
            self._layout.addWidget(_build_block(title, payload))
        self._layout.addStretch()


# ------------------------------------------------------------------ 块渲染


def _build_block(title: str, payload: ViewData | str) -> QGroupBox:  # pragma: no cover
    """单个结果块：QGroupBox 标题 + 按类型紧凑渲染的正文."""
    box = QGroupBox(title)
    layout = QVBoxLayout(box)
    layout.setContentsMargins(theme.SPACING_SM, theme.SPACING_SM, theme.SPACING_SM, theme.SPACING_SM)
    layout.setSpacing(0)
    if isinstance(payload, str):  # 块级解析失败
        body: QWidget = QLabel(payload, objectName="errorText")
        body.setWordWrap(True)
    elif isinstance(payload, CurveData):
        body = build_curve_widget(payload)
        body.setFixedHeight(_GROUPED_CURVE_HEIGHT)
    elif isinstance(payload, TableData):
        body = build_table_widget(payload)
        body.setMaximumHeight(_GROUPED_TABLE_MAX_HEIGHT)
    elif isinstance(payload, TextData):
        body = build_text_widget(payload)
    else:  # CloudData 不参与分组（参数化计算应用页路由到解算视图）
        body = QLabel(_cloud_hint(payload), objectName="secondaryText")
        body.setWordWrap(True)
    layout.addWidget(body)
    return box


def build_curve_widget(data: CurveData) -> ZyPlotWidget:
    """DSL 曲线视图（统一组件 + 多序列图例 + 轴标签 + 对数轴 + 峰值标注 +
    系列样式覆盖 + seaborn whitegrid 对齐 + 完整中文右键菜单）."""
    defaults = PG_CURVE_DEFAULTS
    # 使用统一绘图组件（构造时传入对数轴参数）
    plot = ZyPlotWidget(log_x=data.log_x, log_y=data.log_y)
    if data.series:
        plot.addLegend(offset=defaults["legend_offset"])
    if data.x_label:
        plot.setLabel("bottom", data.x_label)
    if data.y_label:
        plot.setLabel("left", data.y_label)
    for index, series in enumerate(data.series):
        # 系列样式覆盖：color / dash / width
        style = data.series_styles[index] if index < len(data.series_styles) else {}
        color = style.get("color")
        width = float(style.get("width", defaults["curve_width"]))
        dash = style.get("dash")  # "solid"/"dashed"/"dotted"/"-"
        if color:
            pen = pg.mkPen(_resolve_color(color), width=width)
            if dash == "dashed":
                pen.setDashPattern([4, 2])
            elif dash == "dotted":
                pen.setDashPattern([1, 2])
        else:
            pen = pg.mkPen(CURVE_PALETTE[index % len(CURVE_PALETTE)], width=width)
        plot.plot(
            list(series.x),
            list(series.y),
            name=series.name,
            pen=pen,
        )
        # 峰值标注（该序列的极值点）
        if data.mark_peak and len(series.y) > 0:
            peak_idx = _peak_index(series.y)
            plot.plot(
                [series.x[peak_idx]],
                [series.y[peak_idx]],
                name=None,
                pen=None,
                symbol="o",
                symbolBrush="#EF4444",
                symbolSize=8,
            )
    # 完整中文右键菜单（DSL 参数化计算曲线的用户体验 bug 修复点）
    plot.setup_context_menu(
        PlotMenuConfig(
            export_csv_fn=lambda path: _export_curve_csv(data, path),
        )
    )
    return plot


def _export_curve_csv(data: CurveData, path: str) -> None:
    """将 CurveData.series 导出为多列 CSV（第一个序列的 x 列作为公共索引）."""
    target = Path(path)
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        # 表头：x 列 + 各序列 y 列（用序列名，缺省用 Series_N）
        headers = ["x"]
        for i, series in enumerate(data.series):
            headers.append(series.name or f"Series_{i + 1}")
        writer.writerow(headers)
        # 取各序列最短长度（避免越界）
        n = min(len(s.x) for s in data.series) if data.series else 0
        for i in range(n):
            row: list[Any] = [data.series[0].x[i]]
            for series in data.series:
                row.append(series.y[i])
            writer.writerow(row)


def _resolve_color(color: str) -> str:  # pragma: no cover
    """解析语义色名到十六进制（委托 sci.palettes.resolve_curve_color）."""
    return resolve_curve_color(color, 0)


def _peak_index(values: tuple) -> int:  # pragma: no cover
    """返回数值序列中极值的索引."""
    if not values:
        return 0
    # 取绝对极值（正负皆显著）
    abs_vals = [abs(v) for v in values]
    return max(range(len(abs_vals)), key=lambda i: abs_vals[i])


def build_table_widget(data: TableData) -> QWidget:  # pragma: no cover
    """表格（列宽均分 + 斑马纹 + 表头主题底；列格式/对齐已由 stream_view 专用构建器处理）."""
    table = QTableWidget(objectName="dslTable")
    table.setColumnCount(len(data.columns))
    table.setRowCount(len(data.rows))
    table.setHorizontalHeaderLabels(list(data.column_titles))
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    palette = theme.current_palette()
    for row, values in enumerate(data.rows):
        for column, value in enumerate(values):
            col_def = data.columns[column]
            fmt = col_def.format or ".6g"
            cell_text = _format_cell_with_format(value, fmt) if col_def.format else _format_cell(value)
            item = QTableWidgetItem(cell_text)
            if col_def.align == "right":
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            elif col_def.align == "left":
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            elif col_def.align == "center":
                item.setTextAlignment(Qt.AlignCenter)
            if row % 2 == 1:
                item.setBackground(Qt.GlobalColor.transparent)  # QSS zebra 处理
            table.setItem(row, column, item)
    table.setStyleSheet(
        f"QTableWidget#dslTable {{ gridline-color: {palette.border}; }}"
        f"QHeaderView::section {{ background: {palette.bg_muted}; }}"
    )
    return table


def _format_cell_with_format(value: Any, fmt: str) -> str:  # pragma: no cover
    """按 printf 格式规格格式化单元格."""
    if fmt and isinstance(value, float):
        try:
            return format(value, fmt)
        except (ValueError, TypeError):
            return str(value)
    return _format_cell(value)


def build_text_widget(data: TextData) -> QWidget:  # pragma: no cover
    """文本（自动换行正文）."""
    label = QLabel(data.text, objectName="resultText")
    label.setWordWrap(True)
    return label


def _cloud_hint(data: CloudData) -> str:  # pragma: no cover
    """云图占位提示（实际渲染由参数化计算应用页路由到解算视图）."""
    return f"云图结果 {data.node_id!r} 由解算视图渲染"


def _format_cell(value: Any) -> str:
    """表格单元格文本（浮点 6 位有效数字，其余 str）."""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
