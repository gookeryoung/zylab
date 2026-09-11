"""模块工具箱：列出 BUILTIN + SOLVER modules，双击或拖放添加到画布.

按 ModuleCategory 分组，每组 QTreeWidget 的顶层节点，模块作为叶子项。
双击叶子 → 发出 ``node_requested`` 信号，由 FlowchartPage 桥接创建节点。
"""

from __future__ import annotations

from typing import Iterable

try:
    from PySide2.QtCore import Qt, Signal
    from PySide2.QtWidgets import QTreeWidget, QTreeWidgetItem
except ImportError:
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

from zylab.flowchart.module import ModuleCategory, ModuleSpec, all_modules

__all__ = ["ModuleToolbox"]

_CATEGORY_LABELS = {
    ModuleCategory.SOURCE: "数据源",
    ModuleCategory.ANALYSIS: "分析求解器",
    ModuleCategory.POST: "后处理",
}


def _group_modules(specs: Iterable[ModuleSpec]) -> dict[ModuleCategory, list[ModuleSpec]]:
    """按 ModuleCategory 分组，保持原始顺序."""
    buckets: dict[ModuleCategory, list[ModuleSpec]] = {}
    for spec in specs:
        buckets.setdefault(spec.category, []).append(spec)
    return buckets


class ModuleToolbox(QTreeWidget):
    """模块工具箱面板。

    信号：
        node_requested(type_id: str): 用户双击/回车请求添加该类型节点
    """

    node_requested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setHeaderLabels(["模块"])
        self.setMinimumWidth(220)
        self.setMaximumWidth(320)
        self.setIndentation(14)
        self.setExpandsOnDoubleClick(False)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.itemActivated.connect(self._on_item_double_clicked)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollMode(QTreeWidget.ScrollPerPixel)
        self.refresh()

    def refresh(self) -> None:
        """从 all_modules() 重新构建树."""
        self.clear()
        buckets = _group_modules(all_modules())
        for category in (
            ModuleCategory.SOURCE,
            ModuleCategory.ANALYSIS,
            ModuleCategory.POST,
        ):
            specs = buckets.get(category, [])
            if not specs:
                continue
            group = QTreeWidgetItem(self, [_CATEGORY_LABELS.get(category, category.value)])
            group.setFlags(group.flags() & ~Qt.ItemIsSelectable & ~Qt.ItemIsDragEnabled)
            for spec in specs:
                leaf = QTreeWidgetItem(group, [spec.name])
                leaf.setData(0, Qt.UserRole, spec.type_id)
                leaf.setToolTip(0, f"{spec.type_id}\n{self._describe(spec)}")
            group.setExpanded(True)

    @staticmethod
    def _describe(spec: ModuleSpec) -> str:
        """端口/参数简要描述供 Tooltip."""
        parts = []
        if spec.inputs:
            parts.append(f"输入: {', '.join(p.name for p in spec.inputs)}")
        if spec.outputs:
            parts.append(f"输出: {', '.join(p.name for p in spec.outputs)}")
        if spec.params:
            parts.append(f"参数: {', '.join(p.key for p in spec.params)}")
        return " | ".join(parts)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        type_id = item.data(0, Qt.UserRole)
        if type_id:
            self.node_requested.emit(type_id)
