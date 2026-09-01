"""gui.widgets.template_dialog 模板选择对话框测试：分组树、搜索过滤、详情与确认."""

from __future__ import annotations

import pytest

from zylab.gui.qt_compat import Qt
from zylab.gui.widgets.template_dialog import TemplateDialog, discipline_label
from zylab.studio import BUILTIN_TEMPLATES
from zylab.studio.dsl import DslTemplate

# 对话框承载经典节点图模板；DSL 模板由模板应用页下拉加载，须排除
_CLASSIC_TEMPLATES = [t for t in BUILTIN_TEMPLATES if not isinstance(t, DslTemplate)]


@pytest.mark.gui
def test_discipline_label_mapping() -> None:
    """学科显示名：已知学科中文名，未知归「其它」."""
    assert discipline_label("structural") == "结构分析"
    assert discipline_label("thermal") == "热分析"
    assert discipline_label("unknown_x") == "其它"


@pytest.mark.gui
def test_dialog_groups_by_discipline(qtbot) -> None:
    """分组树：顶级项为学科组头（带计数），模板为子项（携带模板 id）."""
    dialog = TemplateDialog(list(_CLASSIC_TEMPLATES))
    qtbot.addWidget(dialog)
    roots = [dialog._tree.topLevelItem(i) for i in range(dialog._tree.topLevelItemCount())]
    assert len(roots) == 2  # structural + thermal
    texts = [r.text(0) for r in roots]
    assert any("结构分析" in t for t in texts)
    assert any("热分析" in t for t in texts)
    # 子项：模板 id 可回查
    ids = set()
    for root in roots:
        for i in range(root.childCount()):
            ids.add(root.child(i).data(0, Qt.UserRole))
    assert "structural.cantilever_static" in ids
    assert dialog.selected_id is None  # 未确认


@pytest.mark.gui
def test_dialog_search_filters(qtbot) -> None:
    """搜索过滤：按名称跨组匹配，命中组保留，首个匹配自动选中."""
    dialog = TemplateDialog(list(_CLASSIC_TEMPLATES))
    qtbot.addWidget(dialog)
    dialog._search.setText("悬臂梁")  # 名称匹配（全部为结构学科）
    assert dialog._tree.topLevelItemCount() == 1
    assert dialog._tree.topLevelItem(0).childCount() >= 4
    assert dialog._current_template() is not None  # 首个匹配自动选中
    dialog._search.setText("电加热板")  # 热学科命中
    assert dialog._tree.topLevelItemCount() == 1
    assert "热分析" in dialog._tree.topLevelItem(0).text(0)
    dialog._search.setText("不存在的关键字xyz")
    assert dialog._tree.topLevelItemCount() == 0


@pytest.mark.gui
def test_dialog_selection_and_accept(qtbot) -> None:
    """选择与确认：详情面板随选择刷新，_on_accept 记录模板 id；未选中忽略."""
    dialog = TemplateDialog(list(_CLASSIC_TEMPLATES))
    qtbot.addWidget(dialog)
    dialog._on_accept()  # 未选中任何模板
    assert dialog.selected_id is None
    # 选中第二组首个模板 -> 详情刷新 -> 确认
    root = dialog._tree.topLevelItem(0)
    item = root.child(1)
    dialog._tree.setCurrentItem(item)
    item.setSelected(True)
    dialog._show_detail()  # 选择信号在部分平台不触发，显式刷新详情
    template_id = item.data(0, Qt.UserRole)
    template = next(t for t in _CLASSIC_TEMPLATES if t.id == template_id)
    assert dialog._detail_title.text() == template.name
    assert discipline_label(template.discipline) in dialog._detail_meta.text()
    dialog._on_accept()
    assert dialog.selected_id == template_id


@pytest.mark.gui
def test_dialog_double_click_accepts(qtbot) -> None:
    """双击模板项：直接确认选择（双击组头无效）."""
    dialog = TemplateDialog(list(_CLASSIC_TEMPLATES))
    qtbot.addWidget(dialog)
    root = dialog._tree.topLevelItem(0)
    dialog._on_tree_double_clicked(root, 0)  # 组头双击：不确认
    assert dialog.selected_id is None
    item = root.child(0)
    item.setSelected(True)
    dialog._tree.setCurrentItem(item)
    dialog._on_tree_double_clicked(item, 0)
    assert dialog.selected_id == item.data(0, Qt.UserRole)
