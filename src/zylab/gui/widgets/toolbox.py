"""模块工具箱：按 catalog 分类树（大类 → 子类 → 模块）渲染，支持拖放与双击.

Phase 1 起引入 :mod:`zylab.flowchart.catalog` 五大类目录，工具箱消费
:func:`catalog_tree` 返回的分层结构：

    材料参数
      └ 弹性材料
          └ 线弹性材料 (material.linear_elastic)
    参数化建模
      └ 梁系几何
          └ 悬臂梁 (geom.cantilever_2d)
    ...

顶层大类按 catalog 定义序（material → geometry → mesh → solver → post → legacy）。
"""

from __future__ import annotations

try:
    from PySide2.QtCore import QMimeData, Qt, Signal
    from PySide2.QtGui import QDrag
    from PySide2.QtWidgets import QTreeWidget, QTreeWidgetItem
except ImportError:
    from PySide6.QtCore import QMimeData, Qt, Signal
    from PySide6.QtGui import QDrag
    from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

from zylab.flowchart.catalog import CatalogNode, catalog_tree
from zylab.flowchart.module import ModuleSpec, all_modules

__all__ = ["ModuleToolbox"]


class ModuleToolbox(QTreeWidget):
    """模块工具箱面板（五大类分层树）.

    信号：
        node_requested(type_id: str): 用户双击/回车请求添加该类型节点
        node_dropped(type_id: str, global_pos): 用户将模块拖放到画布
    """

    node_requested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setHeaderLabels(["模块"])
        self.setMinimumWidth(220)
        self.setMaximumWidth(340)
        self.setIndentation(14)
        self.setExpandsOnDoubleClick(False)
        self.setDragEnabled(True)
        self.setDragDropMode(QTreeWidget.DragOnly)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.itemActivated.connect(self._on_item_double_clicked)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollMode(QTreeWidget.ScrollPerPixel)
        self.refresh()

    def refresh(self) -> None:
        """重建工具箱树（按当前 all_modules + catalog 分类）."""
        self.clear()
        tree = catalog_tree(all_modules())
        for top in tree:
            self._populate_top(top, self)
        # 默认展开前三个大类（常用）
        for i in range(min(3, self.topLevelItemCount())):
            item = self.topLevelItem(i)
            if item is not None:
                item.setExpanded(True)

    def _populate_top(self, top: CatalogNode, parent: QTreeWidget | QTreeWidgetItem) -> None:
        """递归填充树节点（顶层挂到 QTreeWidget，子类挂到父 QTreeWidgetItem）."""
        parent_item = QTreeWidgetItem(parent, [top.label])
        # 子类别（递归）
        for sub in top.children:
            self._populate_sub(sub, parent_item)
        # 直接挂在本节点下的模块（顶层大类可能没有直接模块）
        for spec in top.modules:
            self._add_leaf(spec, parent_item)
        # 若父节点无任何子项，至少显示一个空占位，不处理（QTreeWidget 会隐藏空组）

    def _populate_sub(self, node: CatalogNode, parent: QTreeWidgetItem) -> None:
        """子类节点：挂到 parent 下，其下再递归挂子类 + 模块."""
        group_item = QTreeWidgetItem(parent, [node.label])
        group_item.setFlags(group_item.flags() & ~Qt.ItemIsSelectable & ~Qt.ItemIsDragEnabled)
        group_item.setExpanded(True)
        for child in node.children:
            self._populate_sub(child, group_item)
        for spec in node.modules:
            self._add_leaf(spec, group_item)

    def _add_leaf(self, spec: ModuleSpec, parent: QTreeWidgetItem) -> None:
        """添加模块叶子节点."""
        leaf = QTreeWidgetItem(parent, [spec.name])
        leaf.setData(0, Qt.UserRole, spec.type_id)
        leaf.setToolTip(0, f"{spec.type_id}\n{self._describe(spec)}")

    @staticmethod
    def _describe(spec: ModuleSpec) -> str:
        """端口/参数简要描述供 Tooltip."""
        parts = []
        if spec.inputs:
            parts.append(f"输入: {', '.join(p.label or p.name for p in spec.inputs)}")
        if spec.outputs:
            parts.append(f"输出: {', '.join(p.label or p.name for p in spec.outputs)}")
        if spec.params:
            parts.append(f"参数: {', '.join(p.label for p in spec.params)}")
        return " | ".join(parts)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        type_id = item.data(0, Qt.UserRole)
        if type_id:
            self.node_requested.emit(type_id)

    def startDrag(self, _supported_actions) -> None:  # Qt 命名约定
        """重写 startDrag：仅叶子节点（带 type_id）能拖；拖出携带 ``application/x-zylab-module-type``."""
        item = self.currentItem()
        type_id = item.data(0, Qt.UserRole) if item else None
        if not type_id:
            return  # 类别节点不允许拖
        mime = QMimeData()
        mime.setData("application/x-zylab-module-type", type_id.encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec_(Qt.CopyAction)
