"""gui.pages.script_page 脚本页测试."""

from __future__ import annotations

import pytest

from zylab.console import ReplKernel
from zylab.gui.pages.script_page import ScriptPage


@pytest.mark.gui
def test_script_page_builds(qtbot) -> None:
    """脚本页应含示例脚本与空输出区."""
    page = ScriptPage(ReplKernel())
    qtbot.addWidget(page)
    assert "linspace" in page._editor.toPlainText()
    assert page._output.toPlainText() == ""


@pytest.mark.gui
def test_script_page_run_outputs(qtbot) -> None:
    """运行脚本应执行整块并渲染输出与结果."""
    kernel = ReplKernel()
    page = ScriptPage(kernel)
    qtbot.addWidget(page)
    page._editor.setPlainText("x = 1 + 2\nprint(x * 10)")
    page._on_run()
    output = page._output.toPlainText()
    assert "30" in output
    # 工作区已更新（与控制台共享命名空间）
    assert kernel.execute("x").result_repr == "3"


@pytest.mark.gui
def test_script_page_run_error(qtbot) -> None:
    """脚本报错应渲染异常文本且不影响后续执行."""
    page = ScriptPage(ReplKernel())
    qtbot.addWidget(page)
    page._editor.setPlainText("1 / 0")
    page._on_run()
    assert "ZeroDivisionError" in page._output.toPlainText()
    page._editor.setPlainText("ok = 42")
    page._on_run()
    assert page._output.toPlainText() == "(无输出)"
