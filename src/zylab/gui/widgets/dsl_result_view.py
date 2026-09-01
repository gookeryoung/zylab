"""DSL 结果视图：按标准化视图数据（Curve/Table/Text）渲染单页结果.

与 :class:`~zylab.gui.widgets.result_view.ResultView`（解对象 -> 云图/
振型等类型化渲染）互补：本视图消费 DSL ``results`` 声明解析出的
:data:`~zylab.studio.results.ViewData`（曲线/表格/文本），供模板应用页
（P6）按结果页签组装；云图声明（CloudData）由模板应用页路由到既有
ResultView，本视图仅显示占位说明。
"""

from __future__ import annotations

from typing import Any

import pyqtgraph as pg

from zylab.studio.results import CloudData, CurveData, TableData, TextData, ViewData

from .. import theme
from ..qt_compat import (
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

__all__ = ["DslResultView"]


class DslResultView(QWidget):
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
            self._body = self._build_curve(data)
        elif isinstance(data, TableData):
            self._body = self._build_table(data)
        elif isinstance(data, TextData):
            self._body = self._build_text(data)
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

    # ------------------------------------------------------------------ 渲染

    def _build_curve(self, data: CurveData) -> QWidget:
        """pyqtgraph 曲线页（多序列图例 + 轴标签）."""
        plot = pg.PlotWidget(background=theme.current_palette().bg_app)
        plot.showGrid(x=True, y=True, alpha=0.3)
        if data.series:
            plot.addLegend(offset=(12, 12))
        if data.x_label:
            plot.setLabel("bottom", data.x_label)
        if data.y_label:
            plot.setLabel("left", data.y_label)
        for index, series in enumerate(data.series):
            plot.plot(
                list(series.x),
                list(series.y),
                name=series.name,
                pen=pg.mkPen(pg.intColor(index, hues=max(len(data.series), 2)), width=2),
            )
        return plot

    def _build_table(self, data: TableData) -> QWidget:
        """表格页（列宽均分，数值 6 位有效数字）."""
        table = QTableWidget(objectName="dslTable")
        table.setColumnCount(len(data.columns))
        table.setRowCount(len(data.rows))
        table.setHorizontalHeaderLabels(list(data.columns))
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for row, values in enumerate(data.rows):
            for column, value in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(_format_cell(value)))
        return table

    def _build_text(self, data: TextData) -> QWidget:
        """文本页（自动换行正文）."""
        label = QLabel(data.text, objectName="resultText")
        label.setWordWrap(True)
        return label

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


def _cloud_hint(data: CloudData) -> str:
    """云图占位提示（实际渲染由模板应用页路由到解算视图）."""
    return f"云图结果 {data.node_id!r} 由解算视图渲染"


def _format_cell(value: Any) -> str:
    """表格单元格文本（浮点 6 位有效数字，其余 str）."""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
