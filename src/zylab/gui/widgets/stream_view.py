"""ResultStreamView — Jupyter 式参数化计算结果输出流（分块容器）.

对齐 :class:`~zylab.gui.pages.notebook_page.CellWidget` 的设计语言：
默认透明背景、hover 微背景提示、语义色左边框（由 QSS 属性选择器
渲染），无重卡片边框。每块为 :class:`ResultBlockCard`（标题栏 +
可折叠正文），块间纵向堆叠，整页置于 QScrollArea 滚动查看。

消费方（:class:`~zylab.gui.pages.template_page.TemplatePage`）按 DSL
``results`` 声明聚合：

- 未声明 ``group`` 的非 cloud 结果 → 默认「结果」流页；
- 显式 ``group`` → 单独页签，页内仍为流渲染；
- cloud 结果 → 独立页签（整页解算视图，不参与流）。

正文 curve / table / text 渲染委托 :mod:`dsl_result_view` 的
``build_*_widget`` 系列辅助函数（保持复用）。
"""

from __future__ import annotations

from typing import Any

from zylab.studio.results import CloudData, CurveData, TableColumn, TableData, TextData, ViewData
from zylab.studio.richtext import markdown_to_html

from .. import theme
from ..qt_compat import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QSizePolicy,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from .dsl_result_view import _format_cell, build_curve_widget

__all__ = ["ResultBlockCard", "ResultStreamView"]

#: 分组页内曲线块高度（px，同页多块时固定高度避免挤占）.
_STREAM_CURVE_HEIGHT = 300

#: 分组页内表格块高度上限（px，超出内部滚动）.
_STREAM_TABLE_MAX_HEIGHT = 320

#: 语义色到 QSS 属性选择器颜色的映射（与 style.qss resultCell[semantic="*"] 规则配套）.
_SEMANTIC_BAR_COLORS: dict[str, str] = {
    "info": "#3B82F6",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
}


# ------------------------------------------------------------------ 辅助函数


def _kind_badge(payload: ViewData | str) -> str:
    """类型徽标文本（块标题栏旁小标签）."""
    if isinstance(payload, str):
        return "错误"
    if isinstance(payload, TextData):
        return "text"
    if isinstance(payload, TableData):
        return "table"
    if isinstance(payload, CurveData):
        return "curve"
    if isinstance(payload, CloudData):
        return "cloud"
    return "?"


def _build_table_widget(data: TableData) -> QTableWidget:
    """表格正文（列格式/对齐透传；斑马纹由 QSS 控制）."""
    table = QTableWidget(objectName="dslTable")
    table.setColumnCount(len(data.columns))
    table.setRowCount(len(data.rows))
    table.setHorizontalHeaderLabels(list(data.column_titles))
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    for row, values in enumerate(data.rows):
        for column, value in enumerate(values):
            col_def = data.columns[column]
            fmt = col_def.format or ".6g"
            cell_text = _format_cell_with_format(value, fmt)
            item = QTableWidgetItem(cell_text)
            item.setTextAlignment(_col_alignment(col_def))
            table.setItem(row, column, item)
    return table


def _build_text_body(data: TextData) -> QWidget:
    """文本正文：markdown 用 QTextBrowser.setHtml，plain 用 QLabel.

    样式全由 style.qss 的 resultPlainText / resultMarkdownBody 规则控制，
    零内联 padding。
    """
    if data.format == "markdown":
        html_out = markdown_to_html(data.text)
        browser = QTextBrowser(objectName="resultMarkdownBody")
        browser.setHtml(html_out)
        browser.setOpenExternalLinks(False)
        palette = theme.current_palette()
        browser.setStyleSheet(f"border: none; background: transparent; color: {palette.text_primary};")
        browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return browser

    label = QLabel(data.text, objectName="resultText")
    label.setWordWrap(True)
    return label


def _format_cell_with_format(value: Any, fmt: str) -> str:
    """按 printf 格式规格格式化单元格."""
    if fmt and isinstance(value, float):
        try:
            return format(value, fmt)
        except (ValueError, TypeError):
            return str(value)
    return _format_cell(value)


def _col_alignment(col_def: TableColumn) -> int:
    """列对齐（右对齐数值列；或按声明）."""
    if col_def.align == "right":
        return Qt.AlignRight | Qt.AlignVCenter
    if col_def.align == "left":
        return Qt.AlignLeft | Qt.AlignVCenter
    if col_def.align == "center":
        return Qt.AlignCenter
    return Qt.AlignRight | Qt.AlignVCenter


# ------------------------------------------------------------------ 结果块卡片


class ResultBlockCard(QFrame):
    """单个结果块卡片（标题栏 + 可折叠正文）——Jupyter 式轻盈风格.

    对齐 :class:`~zylab.gui.pages.notebook_page.CellWidget`：
    默认透明背景、hover 微背景提示、语义色左边框（3px，由 QSS 属性
    选择器 ``resultCell[semantic="*"]`` 渲染，替代 CellWidget 的聚焦
    主色左边框），无重卡片边框。

    标题栏：标题文本 + 类型徽标 + 折叠箭头；点击切换正文折叠。
    样式完全由 style.qss 的 QFrame#resultCell 系列规则驱动，零内联样式。
    """

    def __init__(
        self,
        title: str,
        payload: ViewData | str,
        style: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, objectName="resultCell")
        self._collapsed = False

        # 语义色作为 QSS 动态属性，供属性选择器渲染左边框
        self.setProperty("semantic", style if style else "default")

        root = QVBoxLayout(self)
        root.setContentsMargins(theme.SPACING_MD, theme.SPACING_SM, theme.SPACING_MD, theme.SPACING_SM)
        root.setSpacing(theme.SPACING_XS)

        # ---- 标题栏 ----
        header = QFrame(objectName="resultCellHeader")
        header.setCursor(Qt.PointingHandCursor)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(theme.SPACING_SM)

        # 标题
        self._title_label = QLabel(title, objectName="resultTitle")
        header_layout.addWidget(self._title_label)

        # 类型徽标（胶囊样式由 QSS resultKindBadge 控制）
        kind_name = _kind_badge(payload)
        self._badge = QLabel(kind_name, objectName="resultKindBadge")
        header_layout.addWidget(self._badge)

        # 拉伸 + 折叠箭头（jupyter 式 ▾/▸）
        header_layout.addStretch()
        self._caret = QLabel("▾", objectName="resultCellCaret")
        header_layout.addWidget(self._caret)

        header.mousePressEvent = self._toggle  # type: ignore[assignment]
        root.addWidget(header)

        # ---- 正文 ----
        self._body = self._build_body(payload)
        root.addWidget(self._body)

    def _toggle(self, _event: Any = None) -> None:
        """切换折叠状态（jupyter 式箭头联动）."""
        self._collapsed = not self._collapsed
        self._body.setVisible(not self._collapsed)
        self._caret.setText("▸" if self._collapsed else "▾")

    def _build_body(self, payload: ViewData | str) -> QWidget:
        """按 payload 类型构建正文控件（零内联样式，QSS 驱动）."""
        # 错误消息（str）
        if isinstance(payload, str):
            label = QLabel(payload, objectName="errorText")
            label.setWordWrap(True)
            return label

        # CloudData 占位
        if isinstance(payload, CloudData):
            label = QLabel(f"云图结果 {payload.node_id!r} 由解算视图渲染", objectName="secondaryText")
            label.setWordWrap(True)
            return label

        # 曲线
        if isinstance(payload, CurveData):
            widget = build_curve_widget(payload)
            widget.setFixedHeight(_STREAM_CURVE_HEIGHT)
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            return widget

        # 表格
        if isinstance(payload, TableData):
            widget = _build_table_widget(payload)
            widget.setMaximumHeight(_STREAM_TABLE_MAX_HEIGHT)
            return widget

        # 文本
        if isinstance(payload, TextData):
            return _build_text_body(payload)

        # 兜底：未知类型
        return QLabel(str(payload), objectName="secondaryText")


# ------------------------------------------------------------------ 流容器


class ResultStreamView(QWidget):
    """参数化计算结果输出流容器（Jupyter 式分块）.

    用法：

    .. code-block:: python

        stream = ResultStreamView()
        stream.set_blocks([
            ("估计摘要", text_data, "success"),
            ("试验记录", table_data, ""),
            ("曲线拟合", curve_data, ""),
        ])

    - ``set_blocks`` 接受 ``(标题, 视图数据或错误消息, 语义色)`` 序列；
    - 空序列时显示占位提示。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 运行状态头（极简，兼容既有测试；jupyter 无状态头但保留此 API）
        self._run_header = QLabel("尚未运行", objectName="secondaryText")
        root.addWidget(self._run_header)

        # 滚动区 + 容器
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.NoFrame)
        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(
            theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD
        )
        # 块间距与 notebook cell 流一致（SPACING_MD），但 ResultBlockCard 内
        # 已有内边距，所以间距稍密避免重复空隙
        self._container_layout.setSpacing(theme.SPACING_SM)
        self._container_layout.addStretch()
        self._scroll.setWidget(self._container)
        root.addWidget(self._scroll)

    # ------------------------------------------------------------------ 外部 API

    def set_blocks(self, blocks: list[tuple[str, ViewData | str, str]]) -> None:
        """按 ``(块标题, 视图数据或错误消息, 语义色)`` 序列重建流.

        :param blocks: 块声明序列；语义色见 :data:`_SEMANTIC_BAR_COLORS` 键。
        """
        # 清空
        while self._container_layout.count():
            item = self._container_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if not blocks:
            self._run_header.setText("")
            placeholder = QLabel("暂无结果", objectName="secondaryText")
            placeholder.setAlignment(Qt.AlignCenter)
            self._container_layout.addWidget(placeholder)
            self._container_layout.addStretch()
            return

        # 运行完成状态统一由主窗口右下角 indicator 承载，此处仅静默重建块
        self._run_header.setText("")
        for title, payload, style in blocks:
            self._container_layout.addWidget(ResultBlockCard(title, payload, style))
        self._container_layout.addStretch()

    def set_error(self, message: str) -> None:
        """流整体运行失败：错误卡置顶."""
        self.set_blocks([("运行失败", message, "danger")])
