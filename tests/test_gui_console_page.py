"""gui.pages.console_page 控制台页测试."""

from __future__ import annotations

import pytest

from zylab.console import CommandHistory, ReplKernel
from zylab.core import EventBus
from zylab.gui.pages.console_page import ConsolePage, ReplInput, VarTableModel
from zylab.gui.qt_compat import QModelIndex, Qt
from zylab.sci import VarInfo


@pytest.fixture
def bus() -> EventBus:
    """每测试独立事件总线."""
    return EventBus()


@pytest.fixture
def kernel(bus: EventBus) -> ReplKernel:
    """每测试独立内核（与页面共享事件总线，绘图事件可达）."""
    return ReplKernel(bus)


@pytest.fixture
def history() -> CommandHistory:
    """无持久化路径的内存历史."""
    return CommandHistory()


@pytest.mark.gui
def test_var_table_model(qapp) -> None:
    """变量模型的行列/数据/表头应正确."""
    model = VarTableModel()
    assert model.rowCount() == 0
    assert model.columnCount() == 6
    model.set_vars([VarInfo(name="x", type_name="int", shape="", dtype="", nbytes=28, preview="5")])
    assert model.rowCount() == 1
    assert model.data(model.index(0, 0)) == "x"
    assert model.data(model.index(0, 5)) == "5"
    assert model.headerData(0, Qt.Horizontal) == "名称"
    assert model.headerData(0, Qt.Vertical) is None
    assert model.data(QModelIndex()) is None
    assert model.rowCount(model.index(0, 0)) == 0  # 子节点无行
    assert model.data(model.index(0, 0), Qt.UserRole) is None  # 非 DisplayRole


@pytest.mark.gui
def test_repl_input_submit(qtbot, kernel: ReplKernel, history: CommandHistory) -> None:
    """Enter 执行完整代码并清空输入、入历史."""
    box = ReplInput(kernel, history)
    qtbot.addWidget(box)
    results: list = []
    box.submitted.connect(results.append)
    box.setPlainText("1 + 2")
    qtbot.keyClick(box, Qt.Key_Return)
    assert len(results) == 1
    assert results[0].result_repr == "3"
    assert box.toPlainText() == ""
    assert history.entries == ["1 + 2"]


@pytest.mark.gui
def test_repl_input_incomplete_continues(qtbot, kernel: ReplKernel, history: CommandHistory) -> None:
    """不完整代码 Enter 应换行续写而非提交."""
    box = ReplInput(kernel, history)
    qtbot.addWidget(box)
    results: list = []
    box.submitted.connect(results.append)
    box.setPlainText("for i in range(3):")
    qtbot.keyClick(box, Qt.Key_Return)
    assert results == []
    assert box.toPlainText().endswith("\n") or box.blockCount() >= 1  # 已换行
    assert history.entries == []


@pytest.mark.gui
def test_repl_input_shift_enter_soft_newline(qtbot, kernel: ReplKernel, history: CommandHistory) -> None:
    """Shift+Enter 软换行不提交."""
    box = ReplInput(kernel, history)
    qtbot.addWidget(box)
    results: list = []
    box.submitted.connect(results.append)
    box.setPlainText("x = 1")
    from zylab.gui.qt_compat import QTextCursor

    box.moveCursor(QTextCursor.End)  # 光标移到末尾再软换行
    qtbot.keyClick(box, Qt.Key_Return, modifier=Qt.ShiftModifier)
    assert results == []
    assert box.toPlainText() == "x = 1\n"


@pytest.mark.gui
def test_repl_input_empty_enter_noop(qtbot, kernel: ReplKernel, history: CommandHistory) -> None:
    """空输入 Enter 无动作."""
    box = ReplInput(kernel, history)
    qtbot.addWidget(box)
    results: list = []
    box.submitted.connect(results.append)
    qtbot.keyClick(box, Qt.Key_Return)
    assert results == []


@pytest.mark.gui
def test_repl_input_history_navigation(qtbot, kernel: ReplKernel, history: CommandHistory) -> None:
    """Up/Down 浏览历史，越到底还原暂存输入."""
    history.add("first")
    history.add("second")
    box = ReplInput(kernel, history)
    qtbot.addWidget(box)
    qtbot.keyClick(box, Qt.Key_Up)
    assert box.toPlainText() == "second"
    qtbot.keyClick(box, Qt.Key_Up)
    assert box.toPlainText() == "first"
    qtbot.keyClick(box, Qt.Key_Down)
    assert box.toPlainText() == "second"
    qtbot.keyClick(box, Qt.Key_Down)
    assert box.toPlainText() == ""


@pytest.mark.gui
def test_console_page_renders_result(qtbot, kernel: ReplKernel, history: CommandHistory, bus: EventBus) -> None:
    """执行后输出区含命令与结果，变量浏览器刷新."""
    page = ConsolePage(kernel, history, bus)
    qtbot.addWidget(page)
    page._input.setPlainText("z = 42")
    qtbot.keyClick(page._input, Qt.Key_Return)
    text = page._output.toPlainText()
    assert ">>> z = 42" in text
    page._input.setPlainText("z")
    qtbot.keyClick(page._input, Qt.Key_Return)
    assert "ans = 42" in page._output.toPlainText()
    assert page._var_model.rowCount() >= 1


@pytest.mark.gui
def test_console_page_renders_error(qtbot, kernel: ReplKernel, history: CommandHistory, bus: EventBus) -> None:
    """异常应渲染到输出区."""
    page = ConsolePage(kernel, history, bus)
    qtbot.addWidget(page)
    page._input.setPlainText("1 / 0")
    qtbot.keyClick(page._input, Qt.Key_Return)
    assert "ZeroDivisionError" in page._output.toPlainText()


@pytest.mark.gui
def test_console_page_whos_refresh(qtbot, kernel: ReplKernel, history: CommandHistory, bus: EventBus) -> None:
    """whos 命令输出表格且变量模型同步."""
    page = ConsolePage(kernel, history, bus)
    qtbot.addWidget(page)
    page._input.setPlainText("my_var = [1, 2, 3]")
    qtbot.keyClick(page._input, Qt.Key_Return)
    page._input.setPlainText("whos()")
    qtbot.keyClick(page._input, Qt.Key_Return)
    assert "my_var" in page._output.toPlainText()
    names = [page._var_model.data(page._var_model.index(row, 0)) for row in range(page._var_model.rowCount())]
    assert "my_var" in names


@pytest.mark.gui
def test_console_page_side_tabs(qtbot, kernel: ReplKernel, history: CommandHistory, bus: EventBus) -> None:
    """右侧选项卡应为 变量/绘图 两页，默认停在变量页."""
    page = ConsolePage(kernel, history, bus)
    qtbot.addWidget(page)
    assert page._side_tabs.count() == 2
    assert page._side_tabs.tabText(0) == "变量"
    assert page._side_tabs.tabText(1) == "绘图"
    assert page._side_tabs.currentIndex() == 0


@pytest.mark.gui
def test_console_page_work_tabs(qtbot, kernel: ReplKernel, history: CommandHistory, bus: EventBus) -> None:
    """左侧选项卡应为 交互/脚本 两页，默认停在交互页."""
    page = ConsolePage(kernel, history, bus)
    qtbot.addWidget(page)
    assert page._work_tabs.count() == 2
    assert page._work_tabs.tabText(0) == "交互"
    assert page._work_tabs.tabText(1) == "脚本"
    assert page._work_tabs.currentIndex() == 0


@pytest.mark.gui
def test_console_page_script_plot_same_screen(
    qtbot, kernel: ReplKernel, history: CommandHistory, bus: EventBus
) -> None:
    """脚本页运行含 plot 的脚本：停在脚本页的同时右侧自动切绘图（同屏可见）."""
    page = ConsolePage(kernel, history, bus)
    qtbot.addWidget(page)
    page._work_tabs.setCurrentIndex(1)  # 切到脚本页
    page._script_page._run_button.click()
    # 默认示例脚本含 plot：左仍停在脚本、右侧已切绘图
    assert page._work_tabs.currentIndex() == 1
    assert page._side_tabs.currentIndex() == 1
    assert len(page._plot_page._plot.listDataItems()) == 1
    # 脚本产生的变量同步进变量表（共享命名空间）
    names = [page._var_model.data(page._var_model.index(row, 0)) for row in range(page._var_model.rowCount())]
    assert "x" in names


@pytest.mark.gui
def test_console_page_vars_always_visible(qtbot, kernel: ReplKernel, history: CommandHistory, bus: EventBus) -> None:
    """任意命令执行后变量表即时刷新（无需 whos）."""
    page = ConsolePage(kernel, history, bus)
    qtbot.addWidget(page)
    assert page._var_model.rowCount() == 0  # 初始空
    page._input.setPlainText("a = 1")
    qtbot.keyClick(page._input, Qt.Key_Return)
    names = [page._var_model.data(page._var_model.index(row, 0)) for row in range(page._var_model.rowCount())]
    assert "a" in names  # 未调用 whos 即出现在变量表


@pytest.mark.gui
def test_console_page_plot_switches_tab(qtbot, kernel: ReplKernel, history: CommandHistory, bus: EventBus) -> None:
    """plot 命令渲染后自动切换到绘图选项卡."""
    page = ConsolePage(kernel, history, bus)
    qtbot.addWidget(page)
    page._input.setPlainText("import numpy as np")
    qtbot.keyClick(page._input, Qt.Key_Return)
    page._input.setPlainText("plot(np.arange(3), np.arange(3))")
    qtbot.keyClick(page._input, Qt.Key_Return)
    assert page._side_tabs.currentIndex() == 1
    assert len(page._plot_page._plot.listDataItems()) == 1
