"""表格构建公共模块：DSL 参数化计算与结果流共用的 QTableWidget 构建器.

合并 :mod:`dsl_result_view` 与 :mod:`stream_view` 中重复的表格构建、
单元格格式化、列对齐逻辑，消除两处各约 30 行的重复实现。

消费者统一从此模块导入 :func:`build_table_widget` / :func:`format_cell` /
:func:`format_cell_with_format` / :func:`col_alignment`。
"""

from __future__ import annotations

from typing import Any

from zylab.flowchart.results import TableColumn, TableData

from ..qt_compat import QHeaderView, Qt, QTableWidget, QTableWidgetItem

__all__ = [
    "build_table_widget",
    "col_alignment",
    "format_cell",
    "format_cell_with_format",
]


def format_cell(value: Any) -> str:
    """表格单元格默认格式化（浮点 6 位有效数字，其余 str）."""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_cell_with_format(value: Any, fmt: str) -> str:
    """按 printf 格式规格格式化单元格（浮点且声明了 fmt 时用 format()，其余回退 format_cell）."""
    if fmt and isinstance(value, float):
        try:
            return format(value, fmt)
        except (ValueError, TypeError):
            return str(value)
    return format_cell(value)


def col_alignment(col_def: TableColumn) -> int:
    """列对齐（right/left/center，缺省右对齐数值列）."""
    if col_def.align == "right":
        return Qt.AlignRight | Qt.AlignVCenter
    if col_def.align == "left":
        return Qt.AlignLeft | Qt.AlignVCenter
    if col_def.align == "center":
        return Qt.AlignCenter
    return Qt.AlignRight | Qt.AlignVCenter


def build_table_widget(
    data: TableData,
    *,
    include_zebra: bool = False,
) -> QTableWidget:
    """按 TableData 构建只读 QTableWidget.

    Args:
        data: DSL 声明的表格视图数据。
        include_zebra: 偶数行背景透明（由 QSS 斑马纹规则渲染，需配合 QSS）。

    Returns:
        构建好的 QTableWidget（objectName="dslTable"，可编辑已禁用，列宽均分）。
    """
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
            cell_text = format_cell_with_format(value, fmt)
            item = QTableWidgetItem(cell_text)
            item.setTextAlignment(col_alignment(col_def))
            if include_zebra and row % 2 == 1:
                item.setBackground(Qt.GlobalColor.transparent)  # QSS zebra 处理
            table.setItem(row, column, item)

    # gridline-color + header 背景全由 QTableWidget#dslTable（fragments 20_containers.qss）驱动
    return table
