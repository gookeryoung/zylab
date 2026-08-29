"""gui.main_window 主窗口测试."""

from __future__ import annotations

from pathlib import Path

import pytest

from zylab.gui.main_window import MainWindow


@pytest.fixture
def isolated_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """劫持默认数据目录到临时路径，避免污染真实用户目录."""
    monkeypatch.setattr("zylab.gui.main_window.default_data_dir", lambda: tmp_path)
    return tmp_path


@pytest.mark.gui
def test_main_window_builds(qtbot, isolated_data_dir: Path) -> None:
    """主窗口应完成四区装配."""
    win = MainWindow()
    qtbot.addWidget(win)
    assert "zylab" in win.windowTitle()
    assert win._stack.count() == 3
    assert win._sidebar.currentRow() == 0


@pytest.mark.gui
def test_main_window_switches_to_plot_on_plot_command(qtbot, isolated_data_dir: Path) -> None:
    """plot 命令渲染后应自动切换到绘图页."""
    win = MainWindow()
    qtbot.addWidget(win)
    result = win.kernel.execute("plot([1, 2, 3])")
    assert result.error is None
    assert win._sidebar.currentRow() == 1
    assert win._stack.currentIndex() == 1


@pytest.mark.gui
def test_main_window_close_saves_history(qtbot, isolated_data_dir: Path) -> None:
    """关闭窗口应持久化命令历史."""
    win = MainWindow()
    qtbot.addWidget(win)
    win._history.add("plot(x)")
    win.close()
    assert (isolated_data_dir / "history.json").exists()


@pytest.mark.gui
def test_main_window_sidebar_switch(qtbot, isolated_data_dir: Path) -> None:
    """侧边栏切换应联动内容区."""
    win = MainWindow()
    qtbot.addWidget(win)
    win._sidebar.setCurrentRow(2)
    assert win._stack.currentIndex() == 2
