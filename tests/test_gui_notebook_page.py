"""gui.pages.notebook_page 笔记本页测试."""

from __future__ import annotations

from pathlib import Path

import pytest

from zylab.console import ReplKernel
from zylab.core import EventBus
from zylab.gui.pages.notebook_page import CellEditor, CellWidget, NotebookPage, VarTableModel
from zylab.gui.qt_compat import Qt
from zylab.sci import (
    Notebook,
    PlotOutput,
    PlotSeries,
    ResultOutput,
    StreamOutput,
    load_notebook,
    new_cell,
    save_notebook,
)


@pytest.fixture
def bus() -> EventBus:
    """每测试独立事件总线."""
    return EventBus()


@pytest.fixture
def kernel(bus: EventBus) -> ReplKernel:
    """每测试独立内核（与页面共享事件总线，绘图事件可达）."""
    return ReplKernel(bus)


@pytest.fixture
def page(qtbot, kernel: ReplKernel, bus: EventBus) -> NotebookPage:
    """已显示的笔记本页（含欢迎首格）."""
    widget = NotebookPage(kernel, bus)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def status_messages(page: NotebookPage) -> list[str]:
    """捕获笔记本页状态消息（主窗口状态栏承接显示）."""
    seen: list[str] = []
    page.status_message.connect(seen.append)
    return seen


@pytest.mark.gui
def test_cell_tool_buttons_use_icons(page: NotebookPage) -> None:
    """单元级悬停工具条为 SVG 图标按钮（空文字、非空图标），主题切换后重着色."""
    widget = page._widgets[0]
    for button in widget._tool_buttons:
        assert button.text() == ""
        assert not button.icon().isNull()
    page.refresh_theme()
    for button in widget._tool_buttons:
        assert not button.icon().isNull()


@pytest.mark.gui
def test_initial_page_has_welcome_cell(page: NotebookPage) -> None:
    """新建页面应含一个演示首格，未执行无输出."""
    assert len(page._widgets) == 1
    cell = page._widgets[0].cell
    assert "linspace" in cell.source
    assert cell.outputs == []
    assert cell.execution_count is None
    assert page.is_dirty() is False
    assert page.path is None


@pytest.mark.gui
def test_run_current_populates_outputs_and_vars(page: NotebookPage, kernel: ReplKernel) -> None:
    """运行当前格应回写输出/计数并刷新变量浏览器."""
    page._widgets[0].editor.setPlainText("x = 41\nx + 1")
    page.run_current()
    cell = page._widgets[0].cell
    assert cell.execution_count == 1
    result = next(o for o in cell.outputs if isinstance(o, ResultOutput))
    assert result.repr_text == "42"
    assert page.is_dirty() is True
    assert "x" in kernel.namespace
    assert page._var_model.rowCount() >= 1  # 至少含 x（用户变量）


@pytest.mark.gui
def test_run_all_continues_after_error(page: NotebookPage) -> None:
    """全部运行遇错不中断：错误格得 ErrorOutput，后续格仍执行."""
    from zylab.sci import ErrorOutput

    page.insert_at(1)
    page._widgets[0].editor.setPlainText("1 / 0")
    page._widgets[1].editor.setPlainText("ok = 2")
    page.run_all()
    assert any(isinstance(o, ErrorOutput) for o in page._widgets[0].cell.outputs)
    assert page._widgets[1].cell.execution_count == 2
    assert page._kernel.namespace["ok"] == 2


@pytest.mark.gui
def test_run_from_current(page: NotebookPage) -> None:
    """从此运行应从当前焦点格执行到末尾."""
    for i in range(3):
        page.insert_at(page._current + 1)
        page._widgets[i].editor.setPlainText(f"a{i} = {i}")
    page._current = 1
    page.run_from_current()
    assert page._widgets[0].cell.execution_count is None  # 第 0 格未执行
    assert page._widgets[1].cell.execution_count == 1
    assert page._widgets[2].cell.execution_count == 2


@pytest.mark.gui
def test_edit_invalidates_outputs(page: NotebookPage) -> None:
    """编辑已执行格应清空旧输出并重置计数（结果失效）."""
    page._widgets[0].editor.setPlainText("1 + 1")
    page.run_current()
    assert page._widgets[0].cell.execution_count == 1
    page._widgets[0].editor.setPlainText("1 + 2")
    cell = page._widgets[0].cell
    assert cell.outputs == []
    assert cell.execution_count is None


@pytest.mark.gui
def test_insert_delete_move(page: NotebookPage) -> None:
    """插入/删除/移动应正确改变单元结构并保持模型同步."""
    page.insert_at(1)
    assert len(page._notebook.cells) == 2
    page._widgets[0].editor.setPlainText("first")
    page._current = 0  # 插入后焦点在新格，改选首格再下移
    page.move_current(1)
    assert page._notebook.cells[1].source == "first"
    page._current = 1
    page.delete_current()
    assert len(page._notebook.cells) == 1
    # 删空自动补格
    page.delete_current()
    assert len(page._notebook.cells) == 1


@pytest.mark.gui
def test_shift_enter_advances_and_appends(qtbot, page: NotebookPage) -> None:
    """Shift+Enter 运行并推进；末格运行后自动新建下格."""
    editor = page._widgets[0].editor
    editor.setPlainText("v = 1")
    qtbot.keyClick(editor, Qt.Key_Return, modifier=Qt.ShiftModifier)
    assert len(page._widgets) == 2
    assert page._current == 1
    assert page._kernel.namespace["v"] == 1
    # 末格 Shift+Enter 仍新建（jupyter 语义）
    qtbot.keyClick(page._widgets[1].editor, Qt.Key_Return, modifier=Qt.ShiftModifier)
    assert len(page._widgets) == 3
    # 中间格 Shift+Enter 只推进不新建
    qtbot.keyClick(page._widgets[1].editor, Qt.Key_Return, modifier=Qt.ShiftModifier)
    assert len(page._widgets) == 3
    assert page._current == 2


@pytest.mark.gui
def test_ctrl_enter_runs_in_place(qtbot, page: NotebookPage) -> None:
    """Ctrl+Enter 只运行本格不推进."""
    editor = page._widgets[0].editor
    editor.setPlainText("w = 9")
    qtbot.keyClick(editor, Qt.Key_Return, modifier=Qt.ControlModifier)
    assert len(page._widgets) == 1
    assert page._kernel.namespace["w"] == 9


@pytest.mark.gui
def test_tab_inserts_spaces(qtbot, page: NotebookPage) -> None:
    """Tab 键应插入 4 空格缩进."""
    editor = page._widgets[0].editor
    editor.setPlainText("")
    qtbot.keyClick(editor, Qt.Key_Tab)
    assert editor.toPlainText() == "    "


@pytest.mark.gui
def test_save_and_reload_roundtrip(page: NotebookPage, tmp_path: Path) -> None:
    """执行后保存 .znbk，重新打开应还原源码与输出（含绘图数值）."""
    editor = page._widgets[0].editor
    editor.setPlainText("import numpy as np\nxv = np.arange(5)\nplot(xv, xv * 2, title='t')\nxv")
    page.run_current()
    target = tmp_path / "demo.znbk"
    assert page._save_to(target) is True
    assert page.is_dirty() is False
    assert page.path == target

    page.open_path(target)
    cell = page._widgets[0].cell
    assert "np.arange" in cell.source
    assert any(isinstance(o, PlotOutput) for o in cell.outputs)
    assert any(isinstance(o, ResultOutput) for o in cell.outputs)
    # 内嵌绘图控件已渲染
    assert page._widgets[0]._output_layout.count() >= 2
    # 离线加载（不经内核）同样可还原绘图数值
    reloaded = load_notebook(target)
    plot = next(o for o in reloaded.cells[0].outputs if isinstance(o, PlotOutput))
    assert plot.series[0].y == [0, 2, 4, 6, 8]


@pytest.mark.gui
def test_open_invalid_file_keeps_current(page: NotebookPage, status_messages: list[str], tmp_path: Path) -> None:
    """打开损坏文件应发状态消息且不破坏当前文档."""
    bad = tmp_path / "bad.znbk"
    bad.write_text("{ not json", encoding="utf-8")
    source_before = page._notebook.cells[0].source
    page.open_path(bad)
    assert page._notebook.cells[0].source == source_before
    assert any("打开失败" in m for m in status_messages)


@pytest.mark.gui
def test_clear_event_clears_outputs(page: NotebookPage, bus: EventBus) -> None:
    """clc() 清屏事件应清空全部单元输出."""
    page._widgets[0].editor.setPlainText("q = 1\nprint('hi')")
    page.run_current()
    assert page._widgets[0].cell.outputs  # print 产生流输出
    page._kernel.execute_cell("clc()")
    assert page._widgets[0].cell.outputs == []
    assert "q" in page._kernel.namespace  # 变量保留


@pytest.mark.gui
def test_cell_widget_renders_all_output_kinds(qtbot) -> None:
    """CellWidget 应按类型渲染流/结果/错误/绘图四类输出."""
    cell = new_cell("pass")
    cell.execution_count = 3
    cell.outputs = [
        StreamOutput(name="stdout", text="hello"),
        ResultOutput(repr_text="42", type_name="int", shape=""),
        PlotOutput(title="t", xlabel="x", ylabel="y", series=[PlotSeries(x=[1, 2], y=[3, 4], label="s")]),
    ]
    widget = CellWidget(cell)
    qtbot.addWidget(widget)
    kinds = {type(widget._output_layout.itemAt(i).widget()).__name__ for i in range(widget._output_layout.count())}
    assert kinds == {"QLabel", "PlotWidget"}
    assert widget._count_label.text() == "In [3]:"


@pytest.mark.gui
def test_cell_widget_renders_error(qtbot) -> None:
    """CellWidget 应渲染错误输出（ename 粗体 + 回溯文本）."""
    from zylab.sci import ErrorOutput

    cell = new_cell("1 / 0")
    cell.execution_count = 4
    cell.outputs = [ErrorOutput(ename="ZeroDivisionError", traceback_text="ZeroDivisionError: division by zero")]
    widget = CellWidget(cell)
    qtbot.addWidget(widget)
    assert widget._count_label.text() == "In [4]:"
    label = widget._output_layout.itemAt(0).widget()
    assert "ZeroDivisionError" in label.text()


@pytest.mark.gui
def test_cell_widget_reloads_from_model(qtbot) -> None:
    """reload_from_model 应回填编辑器文本."""
    cell = new_cell("a = 1")
    widget = CellWidget(cell)
    qtbot.addWidget(widget)
    cell.source = "b = 2"
    widget.reload_from_model()
    assert widget.editor.toPlainText() == "b = 2"


@pytest.mark.gui
def test_cell_editor_signal(qtbot) -> None:
    """编辑器运行信号应携带 advance 标志."""
    editor = CellEditor()
    qtbot.addWidget(editor)
    seen: list[bool] = []
    editor.run_requested.connect(seen.append)
    with qtbot.waitSignal(editor.run_requested):
        qtbot.keyClick(editor, Qt.Key_Return, modifier=Qt.ControlModifier)
    assert seen == [False]


@pytest.mark.gui
def test_var_table_model_set_vars(qtbot) -> None:
    """变量模型整体替换应刷新行数."""
    from zylab.sci import VarInfo

    model = VarTableModel()
    assert model.rowCount() == 0
    model.set_vars([VarInfo(name="x", type_name="int", shape="", dtype="", nbytes=28, preview="5")])
    assert model.rowCount() == 1
    model.set_vars([])
    assert model.rowCount() == 0


@pytest.mark.gui
def test_new_document_resets(page: NotebookPage, tmp_path: Path) -> None:
    """新建应重置路径与 dirty 状态并回到欢迎格."""
    page._widgets[0].editor.setPlainText("dirty = 1")
    page._save_to(tmp_path / "a.znbk")
    assert page.path is not None
    page.new_document()
    assert page.path is None
    assert page.is_dirty() is False
    assert len(page._widgets) == 1
    assert "linspace" in page._widgets[0].cell.source


@pytest.mark.gui
def test_maybe_save_not_dirty_returns_true(page: NotebookPage) -> None:
    """无未保存修改时 maybe_save 直接放行."""
    assert page.maybe_save() is True


# ------------------------------------------------ 变量侧栏折叠（jupyterlab 语义）


@pytest.mark.gui
def test_var_panel_collapsed_by_default(page: NotebookPage) -> None:
    """变量面板默认收起，单元流占满全宽."""
    assert page._var_view.isHidden() is True
    assert page._btn_vars.isChecked() is False


@pytest.mark.gui
def test_var_panel_toggle(page: NotebookPage) -> None:
    """变量按钮切换侧栏显示并联动选中态."""
    page._btn_vars.toggle()
    assert page._var_view.isHidden() is False
    assert page._btn_vars.isChecked() is True
    page._toggle_vars(False)
    assert page._var_view.isHidden() is True


@pytest.mark.gui
def test_run_does_not_auto_expand_var_panel(page: NotebookPage) -> None:
    """运行单元不自动弹出变量面板（主动查看语义）."""
    page._widgets[0].editor.setPlainText("auto = 1")
    page.run_current()
    assert page._var_view.isHidden() is True
    # 面板展开时刷新仍生效
    page._toggle_vars(True)
    page.run_current()
    assert page._var_model.rowCount() >= 1


# ------------------------------------------------ 单元级工具条（悬停动作）


@pytest.mark.gui
def test_cell_toolbar_hidden_until_hover(qtbot) -> None:
    """工具条按钮默认隐藏，进入/离开卡片切换可见性."""
    widget = CellWidget(new_cell("a = 1"))
    qtbot.addWidget(widget)
    assert all(not b.isVisible() for b in widget._tool_buttons)


@pytest.mark.gui
def test_cell_action_run(page: NotebookPage) -> None:
    """工具条 run 动作执行对应单元."""
    page._widgets[0].editor.setPlainText("r = 7")
    page._on_cell_action(page._widgets[0], "run")
    assert page._widgets[0].cell.execution_count == 1
    assert page._kernel.namespace["r"] == 7


@pytest.mark.gui
def test_cell_action_insert_and_delete(page: NotebookPage) -> None:
    """工具条 insert 在目标格下方插入；delete 删除指定格."""
    page._widgets[0].editor.setPlainText("first")
    page._on_cell_action(page._widgets[0], "insert")
    assert len(page._notebook.cells) == 2
    page._widgets[1].editor.setPlainText("second")
    page._on_cell_action(page._widgets[0], "insert")
    assert page._notebook.cells[1].source == ""  # 插在 first 下方
    # 删除中间空格：剩余 first/second 且顺序不变
    page._on_cell_action(page._widgets[1], "delete")
    assert [c.source for c in page._notebook.cells] == ["first", "second"]


@pytest.mark.gui
def test_cell_action_move(page: NotebookPage) -> None:
    """工具条 up/down 移动指定格."""
    page._on_cell_action(page._widgets[0], "insert")
    page._widgets[0].editor.setPlainText("top")
    page._on_cell_action(page._widgets[0], "down")
    assert page._notebook.cells[1].source == "top"
    page._on_cell_action(page._widgets[1], "up")
    assert page._notebook.cells[0].source == "top"


# ------------------------------------------------ 重启内核


@pytest.mark.gui
def test_restart_kernel_resets_outputs(
    page: NotebookPage, status_messages: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """确认重启后：变量清空、单元输出失效、计数归零、状态提示."""
    from zylab.gui.qt_compat import QMessageBox

    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *_a, **_k: QMessageBox.Yes))
    page._widgets[0].editor.setPlainText("v = 3\nv")
    page.run_current()
    assert page._widgets[0].cell.execution_count == 1
    page.restart_kernel()
    assert page._widgets[0].cell.execution_count is None
    assert page._widgets[0].cell.outputs == []
    assert "v" not in page._kernel.namespace
    assert page._kernel.execution_count == 0
    assert any("内核已重启" in m for m in status_messages)


@pytest.mark.gui
def test_restart_kernel_cancel_keeps_state(page: NotebookPage, monkeypatch: pytest.MonkeyPatch) -> None:
    """取消重启应保持变量与输出不变."""
    from zylab.gui.qt_compat import QMessageBox

    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *_a, **_k: QMessageBox.No))
    page._widgets[0].editor.setPlainText("keep = 1")
    page.run_current()
    page.restart_kernel()
    assert page._kernel.namespace["keep"] == 1
    assert page._widgets[0].cell.execution_count == 1


# ------------------------------------------------ 顶部工具栏结构


@pytest.mark.gui
def test_toolbar_only_document_level_buttons(page: NotebookPage) -> None:
    """工具栏仅保留文档级按钮：纯图标（空文字）+ tooltip 承载语义（单元级操作在悬停工具条）."""
    buttons = (page._btn_new, page._btn_open, page._btn_save, page._btn_run_all, page._btn_restart)
    tooltips = {b.toolTip() for b in buttons}
    assert all(b.objectName() == "toolBtn" and b.text() == "" and b.toolTip() for b in buttons)
    assert "新建笔记本" in tooltips
    assert "打开笔记本 (.znbk)" in tooltips
    assert "保存笔记本 (Ctrl+S)，未保存过自动弹出另存对话框" in tooltips
    # 变量按钮独立分组：checkable 切换侧栏，图标着色随选中态
    assert page._btn_vars.objectName() == "notebookVarToggle"
    assert page._btn_vars.isCheckable()
    assert page._btn_vars.text() == ""
    assert not hasattr(page, "_btn_insert")
    assert not hasattr(page, "_btn_delete")


@pytest.mark.gui
def test_var_table_model_data_roles(qtbot) -> None:
    """变量模型 data/headerData 各角色取值（三列：名称/类型/字节数）."""
    from zylab.gui.qt_compat import QColor, QModelIndex, Qt
    from zylab.sci import VarInfo

    model = VarTableModel()
    model.set_vars([VarInfo(name="x", type_name="int", shape="", dtype="", nbytes=28, preview="5")])
    assert model.columnCount() == 3
    assert model.data(model.index(0, 0)) == "x"
    assert model.data(model.index(0, 1)) == "int"
    assert model.data(model.index(0, 2)) == "28"
    assert isinstance(model.data(model.index(0, 0), Qt.ForegroundRole), QColor)
    assert model.data(QModelIndex()) is None
    assert model.headerData(0, Qt.Horizontal) == "名称"
    assert model.headerData(0, Qt.Vertical) is None
    assert model.headerData(0, Qt.Horizontal, Qt.UserRole) is None


@pytest.mark.gui
def test_var_table_model_tags_role(qtbot) -> None:
    """类型列经 TAGS_ROLE 提供元素类型/形状标签（空项省略，仅类型列有效）."""
    from zylab.sci import VarInfo

    model = VarTableModel()
    model.set_vars(
        [
            VarInfo(name="a", type_name="ndarray", shape="3x4", dtype="float64", nbytes=96, preview=""),
            VarInfo(name="n", type_name="int", shape="", dtype="", nbytes=28, preview=""),
        ]
    )
    assert model.data(model.index(0, 1), model.TAGS_ROLE) == ("float64", "3x4")
    assert model.data(model.index(1, 1), model.TAGS_ROLE) == ()
    assert model.data(model.index(0, 0), model.TAGS_ROLE) is None
    assert model.info_at(0).name == "a"
    assert model.info_at(5) is None


# ------------------------------------------------ 变量详情对话框（MATLAB 风格）


@pytest.mark.gui
def test_var_detail_dialog_matrix_shapes(qtbot) -> None:
    """详情对话框：一维列向量/二维矩阵/多维切片导航均正确呈现."""
    import numpy as np

    from zylab.gui.pages.var_browser import VarDetailDialog
    from zylab.sci import VarInfo

    # 一维：索引 + 值 两列（列向量）
    info1 = VarInfo(name="v", type_name="ndarray", shape="3", dtype="float64", nbytes=24, preview="")
    d1 = VarDetailDialog(info1, np.array([1.0, 2.5, 3.0]))
    qtbot.addWidget(d1)
    assert d1._table.rowCount() == 3
    assert d1._table.columnCount() == 1
    assert d1._table.item(1, 0).text() == "2.5"
    # 二维：行列索引表头
    info2 = VarInfo(name="m", type_name="ndarray", shape="2x3", dtype="float64", nbytes=48, preview="")
    d2 = VarDetailDialog(info2, np.arange(6.0).reshape(2, 3))
    qtbot.addWidget(d2)
    assert d2._table.rowCount() == 2
    assert d2._table.columnCount() == 3
    assert d2._table.item(1, 2).text() == "5"
    # 三维：前置维索引条 + 后两维切片
    cube = np.arange(8.0).reshape(2, 2, 2)
    info3 = VarInfo(name="c", type_name="ndarray", shape="2x2x2", dtype="float64", nbytes=64, preview="")
    d3 = VarDetailDialog(info3, cube)
    qtbot.addWidget(d3)
    assert len(d3._ndim_spinners) == 1
    assert d3._table.item(0, 0).text() == "0"
    d3._ndim_spinners[0].setValue(1)
    assert d3._table.item(0, 0).text() == "4"
    # 摘要含统计（最小/最大/均值）
    assert "最大 7" in d3._build_summary().text()


@pytest.mark.gui
def test_var_detail_dialog_non_array(qtbot) -> None:
    """非数组变量（标量/字符串等）详情用等宽文本 repr 呈现."""
    from zylab.gui.pages.var_browser import VarDetailDialog
    from zylab.sci import VarInfo

    info = VarInfo(name="s", type_name="str", shape="", dtype="", nbytes=52, preview="'hello'")
    dialog = VarDetailDialog(info, "hello")
    qtbot.addWidget(dialog)
    assert "hello" in dialog._build_summary().text() or dialog._build_summary().text().startswith("名称 s")
    # 超大数组截断保护（200 行上限）
    import numpy as np

    info_big = VarInfo(name="b", type_name="ndarray", shape="500", dtype="float64", nbytes=4000, preview="")
    d_big = VarDetailDialog(info_big, np.arange(500.0))
    qtbot.addWidget(d_big)
    assert d_big._table.rowCount() == 200
    assert "仅显示前" in d_big.windowTitle()
    # 超长 repr 截断保护
    info_long = VarInfo(name="t", type_name="str", shape="", dtype="", nbytes=3000, preview="x" * 3000)
    d_long = VarDetailDialog(info_long, "x" * 3000)
    qtbot.addWidget(d_long)
    assert "超长截断" in d_long._build_text_view().text()
    # 非数值数组与全 NaN：统计降级为空，不抛错
    info_str = VarInfo(name="w", type_name="ndarray", shape="2", dtype="<U3", nbytes=24, preview="")
    d_str = VarDetailDialog(info_str, np.array(["ab", "cd"]))
    qtbot.addWidget(d_str)
    assert "最小" not in d_str._build_summary().text()
    info_nan = VarInfo(name="nn", type_name="ndarray", shape="2", dtype="float64", nbytes=16, preview="")
    d_nan = VarDetailDialog(info_nan, np.array([np.nan, np.nan]))
    qtbot.addWidget(d_nan)
    assert "最小" not in d_nan._build_summary().text()


@pytest.mark.gui
def test_var_panel_width_capped_to_quarter(page: NotebookPage) -> None:
    """变量侧栏宽度上限不超过页面 1/4（保底 220px）."""
    page.resize(1000, 700)
    page._update_var_max_width()  # 未显示控件收不到 resize 事件，直接触发
    assert page._var_view.maximumWidth() == 250
    page.resize(600, 700)
    page._update_var_max_width()
    assert page._var_view.maximumWidth() == 220  # 保底


@pytest.mark.gui
def test_var_tag_delegate_paints(qtbot) -> None:
    """类型列委托渲染：类型名 + 标签 chip 离屏绘制不报错."""
    from zylab.gui.pages.var_browser import VarTagDelegate
    from zylab.gui.qt_compat import QPainter, QPixmap, QStyleOptionViewItem, QTableView
    from zylab.sci import VarInfo

    model = VarTableModel()
    model.set_vars([VarInfo(name="a", type_name="ndarray", shape="3x4", dtype="float64", nbytes=96, preview="")])
    view = QTableView()
    view.setModel(model)
    qtbot.addWidget(view)
    view.show()
    assert not view.grab().isNull()  # 整表渲染（委托 paint 实际执行）
    # 直接离屏渲染类型列单元格（覆盖 chip 绘制分支）
    delegate = VarTagDelegate()
    pixmap = QPixmap(220, 24)
    painter = QPainter(pixmap)
    option = QStyleOptionViewItem()
    option.rect.setRect(0, 0, 220, 24)
    delegate.paint(painter, option, model.index(0, 1))
    painter.end()
    assert not pixmap.isNull()


@pytest.mark.gui
def test_open_var_detail(page: NotebookPage, monkeypatch: pytest.MonkeyPatch) -> None:
    """双击变量行弹出详情对话框（对话框经 exec_dialog 兼容执行），无效索引安全返回."""
    from zylab.gui.pages import notebook_page as mod

    opened: list[object] = []
    monkeypatch.setattr(mod, "exec_dialog", opened.append)
    page._widgets[0].editor.setPlainText("zz = [1, 2]")
    page.run_current()
    page.refresh_vars()
    row = next(r for r in range(page._var_model.rowCount()) if page._var_model.info_at(r).name == "zz")
    page._open_var_detail(page._var_model.index(row, 0))
    assert len(opened) == 1
    assert opened[0]._info.name == "zz"  # type: ignore[attr-defined]
    page._open_var_detail(page._var_model.index(-1, 0))  # 无效索引：不弹窗
    assert len(opened) == 1


@pytest.mark.gui
def test_save_as_dialog_cancel(page: NotebookPage, monkeypatch: pytest.MonkeyPatch) -> None:
    """另存对话框取消应返回 False 不设路径."""
    from zylab.gui.pages import notebook_page as mod

    monkeypatch.setattr(mod.QFileDialog, "getSaveFileName", staticmethod(lambda *_a, **_k: ("", "")))
    assert page.save_as() is False
    assert page.path is None


@pytest.mark.gui
def test_save_without_path_goes_dialog(page: NotebookPage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """无路径保存转另存对话框，成功后记录路径."""
    from zylab.gui.pages import notebook_page as mod

    target = tmp_path / "nb.znbk"
    monkeypatch.setattr(mod.QFileDialog, "getSaveFileName", staticmethod(lambda *_a, **_k: (str(target), "")))
    assert page.save() is True
    assert page.path == target


@pytest.mark.gui
def test_save_to_failure_sets_status(
    page: NotebookPage, status_messages: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """保存失败应发状态消息并返回 False."""
    from zylab.sci import NotebookError

    def _boom(*_a, **_k):
        raise NotebookError("boom")

    monkeypatch.setattr("zylab.gui.pages.notebook_page.save_notebook", _boom)
    assert page._save_to(tmp_path / "x.znbk") is False
    assert any("保存失败" in m for m in status_messages)


@pytest.mark.gui
def test_open_empty_notebook_appends_cell(page: NotebookPage, tmp_path: Path) -> None:
    """打开空 cells 笔记本应自动补一空格."""
    empty = tmp_path / "empty.znbk"
    save_notebook(empty, Notebook(cells=[]))
    page.open_path(empty)
    assert len(page._notebook.cells) == 1


@pytest.mark.gui
def test_on_new_cancel_keeps_document(page: NotebookPage, monkeypatch: pytest.MonkeyPatch) -> None:
    """新建询问取消应保持当前文档不变."""
    monkeypatch.setattr(NotebookPage, "maybe_save", lambda _self: False)
    before = page._notebook
    page._on_new()
    assert page._notebook is before


@pytest.mark.gui
def test_on_open_cancel_skips_dialog(page: NotebookPage, monkeypatch: pytest.MonkeyPatch) -> None:
    """打开询问取消不应弹出文件对话框."""
    calls: list = []
    monkeypatch.setattr(NotebookPage, "maybe_save", lambda _self: False)
    monkeypatch.setattr(
        "zylab.gui.pages.notebook_page.QFileDialog.getOpenFileName",
        staticmethod(lambda *_a, **_k: (calls.append(1), ("", ""))[1]),
    )
    page._on_open()
    assert calls == []


@pytest.mark.gui
def test_on_save_slots(page: NotebookPage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """工具栏保存槽与另存快捷键应转发文档操作."""
    target = tmp_path / "slot.znbk"
    monkeypatch.setattr(
        "zylab.gui.pages.notebook_page.QFileDialog.getSaveFileName",
        staticmethod(lambda *_a, **_k: (str(target), "")),
    )
    page._on_save(True)
    assert page.path == target
    other = tmp_path / "slot2.znbk"
    monkeypatch.setattr(
        "zylab.gui.pages.notebook_page.QFileDialog.getSaveFileName",
        staticmethod(lambda *_a, **_k: (str(other), "")),
    )
    page.save_as()
    assert page.path == other


@pytest.mark.gui
def test_move_current_out_of_range(page: NotebookPage) -> None:
    """首格上移/末格下移应无效."""
    page.insert_at(1)
    page._current = 0
    page.move_current(-1)
    assert page._notebook.cells[0].source != ""
    page._current = 1
    before = list(page._notebook.cells)
    page.move_current(1)
    assert page._notebook.cells == before


@pytest.mark.gui
def test_insert_after_current(page: NotebookPage) -> None:
    """在当前格下方插入空格并聚焦."""
    page.insert_after_current()
    assert len(page._notebook.cells) == 2
    assert page._current == 1


@pytest.mark.gui
def test_run_at_out_of_range_noop(page: NotebookPage) -> None:
    """越界索引执行应为空操作."""
    page._run_at(99)
    assert page._widgets[0].cell.execution_count is None


@pytest.mark.gui
def test_focus_index_out_of_range(page: NotebookPage) -> None:
    """越界索引聚焦不应崩溃."""
    page._focus_index(-1)
    assert page._current == -1


@pytest.mark.gui
def test_editor_plain_enter_newline(qtbot, page: NotebookPage) -> None:
    """无修饰 Enter 走默认换行，不发运行信号."""
    editor = page._widgets[0].editor
    seen: list[bool] = []
    editor.run_requested.connect(seen.append)
    qtbot.keyClick(editor, Qt.Key_Return)
    assert seen == []
    assert editor.blockCount() >= 2


@pytest.mark.gui
def test_focus_editor_moves_cursor_to_end(qtbot) -> None:
    """focus_editor 应聚焦并把光标移到末尾."""
    cell = new_cell("abc")
    widget = CellWidget(cell)
    qtbot.addWidget(widget)
    cursor = widget.editor.textCursor()
    cursor.setPosition(0)
    widget.editor.setTextCursor(cursor)
    widget.focus_editor()
    assert widget.editor.textCursor().position() == 3
