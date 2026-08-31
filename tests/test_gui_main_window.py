"""gui.main_window 主窗口测试."""

from __future__ import annotations

from pathlib import Path

import pytest

from zylab.gui.main_window import MainWindow
from zylab.gui.qt_compat import Qt


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
def test_main_window_plot_renders_in_notebook(qtbot, isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """笔记本页运行绘图单元应内嵌渲染，不离开当前页."""
    from zylab.gui.qt_compat import QMessageBox

    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *_args, **_kw: QMessageBox.Discard))
    win = MainWindow()
    qtbot.addWidget(win)
    editor = win._notebook_page._widgets[0].editor
    editor.setPlainText("import numpy as np\nxv = np.arange(4)\nplot(xv, xv + 1)\nxv")
    win._notebook_page.run_current()
    assert win._sidebar.currentRow() == 0
    assert win._stack.currentIndex() == 0
    assert win._notebook_page._widgets[0].cell.execution_count == 1


@pytest.mark.gui
def test_main_window_close_prompts_notebook_save(qtbot, isolated_data_dir: Path) -> None:
    """无未保存修改时关闭应直接放行（不弹保存询问）."""
    win = MainWindow()
    qtbot.addWidget(win)
    win.close()
    assert not win.isVisible()


@pytest.mark.gui
def test_main_window_sidebar_switch(qtbot, isolated_data_dir: Path) -> None:
    """侧边栏切换应联动内容区."""
    win = MainWindow()
    qtbot.addWidget(win)
    win._sidebar.setCurrentRow(2)
    assert win._stack.currentIndex() == 2


@pytest.mark.gui
def test_main_window_sidebar_icons(qtbot, isolated_data_dir: Path) -> None:
    """侧边栏导航项应有非空图标（SVG 着色加载成功）."""
    win = MainWindow()
    qtbot.addWidget(win)
    for row in range(win._sidebar.count()):
        assert not win._sidebar.item(row).icon().isNull(), f"第 {row} 项图标为空"


@pytest.mark.gui
def test_load_icon_renders_and_falls_back(qtbot) -> None:
    """load_icon：彩色 SVG 原色渲染非空；缺失文件退化空图标."""
    from zylab.gui.icons import load_icon

    for name in ("save_project", "save_as_template", "open_project"):
        assert not load_icon(name).isNull(), f"图标 {name} 加载失败"
    assert load_icon("nonexistent").isNull()


@pytest.mark.gui
def test_nav_icon_background_transparent(qtbot) -> None:
    """图标角落像素应全透明（无背景色块；采样图标四角均无笔画）."""
    from zylab.gui.icons import nav_icon

    icon = nav_icon("run_all")
    assert not icon.isNull()
    image = icon.pixmap(32, 32).toImage()
    for x, y in ((1, 1), (30, 1), (1, 30), (30, 30)):
        corner = image.pixelColor(x, y)
        assert corner.alpha() == 0, f"({x}, {y}) 存在非透明背景: alpha={corner.alpha()}"


@pytest.mark.gui
def test_nav_icon_overrides_embedded_fill(qtbot) -> None:
    """iconfont 原始件残留的 path 级 fill 应被剥除，主题着色始终生效."""
    from zylab.gui import icons
    from zylab.gui.icons import nav_icon

    for name in ("run_all", "rerun", "open_file", "cross"):  # 曾带显式 fill 的图标
        svg = (icons._ICONS_DIR / f"{name}.svg").read_text("utf-8")
        assert "fill=" not in svg, f"{name}.svg 仍含显式 fill"
    assert icons._FILL_ATTR_RE.sub("", '<path d="M0 0" fill="#666"></path>') == '<path d="M0 0"></path>'
    assert not nav_icon("run_all", "#123456").isNull()


@pytest.mark.gui
def test_main_window_icons_follow_theme(qtbot, isolated_data_dir: Path) -> None:
    """切换主题后侧边栏图标应重新着色（命令面板预览联动，不持久化）."""
    from zylab.gui import theme

    win = MainWindow()
    qtbot.addWidget(win)
    before = win._sidebar.item(0).icon().pixmap(18, 18).toImage()
    target = next(name for name in theme.THEMES if name != theme.current_palette().name)
    win._set_theme(target, persist=False)
    after = win._sidebar.item(0).icon().pixmap(18, 18).toImage()
    assert before != after


@pytest.mark.gui
def test_main_window_command_search_opens_palette(qtbot, isolated_data_dir: Path) -> None:
    """点击头部命令搜索框应弹出命令面板并列出全部命令."""
    win = MainWindow()
    qtbot.addWidget(win)
    qtbot.mouseClick(win._command_search, Qt.MouseButton.LeftButton)
    assert win._palette._list.count() == len(win._palette._commands) > 0
    win._palette.close()


@pytest.mark.gui
def test_main_window_registers_global_commands(qtbot, isolated_data_dir: Path) -> None:
    """全局命令（导航/笔记本/主题）应注册进面板命令表."""
    win = MainWindow()
    qtbot.addWidget(win)
    assert {
        "go.notebook",
        "go.analysis",
        "go.about",
        "notebook.new",
        "notebook.open",
        "notebook.save",
        "notebook.run_all",
        "notebook.restart",
        "theme.select",
    } <= set(win._palette._by_id)
