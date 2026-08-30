"""模板选择对话框：Workbench 风格分组树 + 搜索过滤 + 详情面板.

- 左侧学科分组树（学科 → 模板两级，模板数量随注册表）；
- 顶部搜索框按名称/标签/描述即时过滤（跨组匹配）；
- 右侧详情面板：选中模板的名称、学科、标签、描述与节点数；
- 双击模板或「确定」确认选择，返回模板 id（取消返回 None）。
"""

from __future__ import annotations

from zylab.studio import Template

from .. import theme
from ..qt_compat import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QSplitter,
    Qt,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

__all__ = ["DISCIPLINE_LABELS", "TemplateDialog", "discipline_label"]

#: 学科显示名表（未知学科归「其它」）
DISCIPLINE_LABELS = {
    "structural": "结构分析",
    "thermal": "热分析",
    "electromagnetic": "电磁分析",
    "acoustic": "声学分析",
    "fluid": "流体分析",
}


def discipline_label(discipline: str) -> str:
    """学科标识转中文显示名（未知归「其它」）."""
    return DISCIPLINE_LABELS.get(discipline, "其它")


class TemplateDialog(QDialog):
    """模板选择对话框（分组树 + 搜索 + 详情面板，Workbench Analysis Systems 风格）."""

    def __init__(self, templates: list[Template], parent: QWidget | None = None) -> None:
        """初始化对话框.

        Args:
            templates: 可选模板表（按注册表顺序；分组在内部完成）。
            parent: 父窗口。
        """
        super().__init__(parent)
        self.setWindowTitle("选择分析模板")
        self.resize(720, 480)
        self._templates = templates
        self._selected_id: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD)
        layout.setSpacing(theme.SPACING_SM)

        self._search = QLineEdit()
        self._search.setPlaceholderText("搜索：名称 / 标签 / 描述")
        self._search.textChanged.connect(self._on_search_changed)
        layout.addWidget(self._search)

        splitter = QSplitter(Qt.Horizontal)
        self._tree = QTreeWidget(objectName="templateTree")
        self._tree.setHeaderHidden(True)
        self._tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self._tree.itemSelectionChanged.connect(self._on_tree_selection)
        self._tree.itemDoubleClicked.connect(self._on_tree_double_clicked)
        splitter.addWidget(self._tree)

        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(theme.SPACING_MD, 0, 0, 0)
        self._detail_title = QLabel(objectName="pageTitle")
        self._detail_meta = QLabel(objectName="secondaryText")
        self._detail_meta.setWordWrap(True)
        self._detail_desc = QLabel()
        self._detail_desc.setWordWrap(True)
        detail_layout.addWidget(self._detail_title)
        detail_layout.addWidget(self._detail_meta)
        detail_layout.addWidget(self._detail_desc)
        detail_layout.addStretch(1)
        splitter.addWidget(detail)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("确定")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._rebuild_tree("")
        self._tree.expandAll()

    # ------------------------------------------------------------------ 公共接口

    @property
    def selected_id(self) -> str | None:
        """确认选择的模板 id（未选择或取消为 None）."""
        return self._selected_id

    # ------------------------------------------------------------------ 内部

    def _matches(self, template: Template, keyword: str) -> bool:
        """模板是否匹配搜索关键字（名称/标签/描述，大小写不敏感）."""
        if not keyword:
            return True
        haystack = " ".join((template.name, *template.tags, template.description)).lower()
        return keyword.lower() in haystack

    def _rebuild_tree(self, keyword: str) -> None:
        """按关键字过滤重建分组树（模板项携带模板 id）."""
        self._tree.clear()
        grouped: dict[str, list[Template]] = {}
        for template in self._templates:
            if self._matches(template, keyword):
                grouped.setdefault(template.discipline, []).append(template)
        for discipline in sorted(grouped):
            group = QTreeWidgetItem(self._tree, [f"{discipline_label(discipline)}（{len(grouped[discipline])}）"])
            group.setFlags(Qt.ItemIsEnabled)  # 组头不可选中
            for template in grouped[discipline]:
                item = QTreeWidgetItem(group, [template.name])
                item.setData(0, Qt.UserRole, template.id)
        self._tree.expandAll()

    def _current_template(self) -> Template | None:
        """当前选中模板（组头未选中任何模板时为 None）."""
        item = self._tree.currentItem()
        if item is None or item.parent() is None:
            return None
        template_id = item.data(0, Qt.UserRole)
        return next((t for t in self._templates if t.id == template_id), None)

    def _on_search_changed(self, keyword: str) -> None:
        """搜索过滤：重建树并保持详情面板（首个匹配自动选中）."""
        self._rebuild_tree(keyword)
        if keyword and self._tree.topLevelItemCount() > 0:
            group = self._tree.topLevelItem(0)
            if group.childCount() > 0:
                group.child(0).setSelected(True)
                self._tree.setCurrentItem(group.child(0))
                self._show_detail()

    def _on_tree_selection(self) -> None:
        """树选择变化：刷新详情面板."""
        self._show_detail()

    def _show_detail(self) -> None:
        """详情面板：选中模板的名称/学科/标签/描述/节点数."""
        template = self._current_template()
        if template is None:
            self._detail_title.setText("")
            self._detail_meta.setText("")
            self._detail_desc.setText("")
            return
        tags = " · ".join(template.tags) if template.tags else "—"
        self._detail_title.setText(template.name)
        self._detail_meta.setText(
            f"{discipline_label(template.discipline)} · 标签: {tags} · {len(template.nodes)} 个节点"
        )
        self._detail_desc.setText(template.description or "（无描述）")

    def _on_tree_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        """双击模板项：直接确认选择（双击组头无效）."""
        if item.parent() is not None:
            self._on_accept()

    def _on_accept(self) -> None:
        """确定：记录选中模板 id 并接受对话框（未选中时忽略）."""
        template = self._current_template()
        if template is None:
            return
        self._selected_id = template.id
        self.accept()
