"""gui.main_window 主窗口测试."""

from __future__ import annotations

from pathlib import Path

import pytest

from zylab.gui.main_window import MainWindow
from zylab.gui.qt_compat import Qt


@pytest.fixture
def isolated_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """劫持默认数据目录到临时路径，避免污染真实用户目录.

    Python from-import 在目标模块创建新绑定，monkeypatch 源头不会
    影响已绑定名字，因此须同时 patch 所有引用目标。此处覆盖被测
    main_window 模块与测试自身两处。
    """
    monkeypatch.setattr("zylab.core.default_data_dir", lambda: tmp_path)
    monkeypatch.setattr("zylab.gui.main_window.default_data_dir", lambda: tmp_path)
    return tmp_path


@pytest.mark.gui
def test_main_window_builds(qtbot, isolated_data_dir: Path) -> None:
    """主窗口应完成四区装配."""
    win = MainWindow()
    qtbot.addWidget(win)
    assert "zylab" in win.windowTitle()
    assert win._stack.count() == 3  # 笔记本/工作台/模板（关于已降级为头部帮助按钮）
    assert win._sidebar.currentRow() == 0


@pytest.mark.gui
def test_main_window_plot_renders_in_notebook(qtbot, isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """笔记本页运行绘图单元应内嵌渲染，不离开当前页."""
    # 运行单元后笔记本变脏，teardown 关窗会触发 maybe_save 的模态确认框
    # （_confirm 自定义按钮，不走 QMessageBox.question）；patch 底层 _confirm
    # 返回“放弃”，避免测试环境弹模态框导致 worker 崩溃。
    monkeypatch.setattr(
        "zylab.gui.pages.notebook_page._confirm",
        lambda *_args, **_kwargs: "放弃",
    )
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
def test_qt_compat_declares_qsvg(qtbot) -> None:
    """qt_compat 显式声明 QtSvg 导入：锁定 SVG 运行时依赖不被当作无用导入清理.

    图标资源为 SVG，运行时依赖 imageformats/qsvg 插件；fspack 按 QtSvg
    是否出现在 import 闭包决定插件去留，该导入被移除时打包产物中 qsvg.dll
    会被裁剪、图标全部静默丢失（QPixmap 加载失败仅返回空对象）。
    """
    from zylab.gui import qt_compat

    assert "QSvgRenderer" in qt_compat.__all__
    assert qt_compat.QSvgRenderer is not None


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
    try:
        win._set_theme(target, persist=False)
        after = win._sidebar.item(0).icon().pixmap(18, 18).toImage()
        assert before != after
    finally:
        theme.set_current_theme(theme.DEFAULT_THEME)


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
        "run.global",
        "notebook.new",
        "notebook.open",
        "notebook.save",
        "notebook.run_all",
        "notebook.restart",
        "theme.select",
    } <= set(win._palette._by_id)


@pytest.mark.gui
def test_main_window_about_dialog_opens(qtbot, isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """帮助按钮存在；_open_about_dialog 调用可执行（monkeypatch exec_ 避免弹模态）."""
    win = MainWindow()
    qtbot.addWidget(win)
    assert hasattr(win, "_help_btn") and win._help_btn is not None
    # patch exec_/exec 避免真正弹模态（_open_about_dialog 内部局部导入 QDialog）
    from zylab.gui.qt_compat import QDialog

    monkeypatch.setattr(QDialog, "exec_", lambda _self: None, raising=False)
    monkeypatch.setattr(QDialog, "exec", lambda _self: None, raising=False)
    win._open_about_dialog()
    # go.about 命令存在且已改为打开对话框
    cmd = win._palette._by_id.get("go.about")
    assert cmd is not None
    assert "关于" in cmd.title


@pytest.mark.gui
def test_main_window_page_shortcuts_installed(qtbot, isolated_data_dir: Path) -> None:
    """Ctrl+1/2/3 与 Ctrl+PgDn/PgUp 快捷键应安装."""
    win = MainWindow()
    qtbot.addWidget(win)
    # 验证 5 个快捷键已安装（对象列表中能找到 activated 信号）
    shortcuts = [obj for obj in win.children() if hasattr(obj, "activated") and "QShortcut" in type(obj).__name__]
    assert len(shortcuts) >= 5, f"Expected >=5 shortcuts, got {len(shortcuts)}"


@pytest.mark.gui
def test_main_window_cycle_pages(qtbot, isolated_data_dir: Path) -> None:
    """Ctrl+PgDn/PgUp 应循环切换侧边栏选中行."""
    win = MainWindow()
    qtbot.addWidget(win)
    win._sidebar.setCurrentRow(0)
    win._cycle_next_page()
    assert win._sidebar.currentRow() == 1
    win._cycle_next_page()
    assert win._sidebar.currentRow() == 2
    win._cycle_next_page()
    assert win._sidebar.currentRow() == 0
    win._cycle_prev_page()
    assert win._sidebar.currentRow() == 2
    win._cycle_prev_page()
    assert win._sidebar.currentRow() == 1


@pytest.mark.gui
def test_main_window_sidebar_toggle(qtbot, isolated_data_dir):
    """_toggle_sidebar 应切换折叠状态并更新宽度与手柄."""
    win = MainWindow()
    qtbot.addWidget(win)
    assert not win._sidebar_folded
    win._toggle_sidebar()
    assert win._sidebar_folded
    win._toggle_sidebar()
    assert not win._sidebar_folded


@pytest.mark.gui
def test_main_window_gui_state_save_load(qtbot, isolated_data_dir):
    """_save_gui_state 写入后 _load_gui_state 应能恢复."""
    import json

    from zylab.core import default_data_dir as _ddd

    win = MainWindow()
    qtbot.addWidget(win)
    win._sidebar_folded = True
    win._save_gui_state()
    path = _ddd() / "gui_state.json"
    assert path.is_file()
    state = json.loads(path.read_text(encoding="utf-8"))
    assert state["sidebar_folded"] is True
    win._sidebar_folded = False
    win._load_gui_state()
    assert win._sidebar_folded is True
    path.unlink()


@pytest.mark.gui
def test_main_window_f5_installed(qtbot, isolated_data_dir: Path) -> None:
    """F5 全局运行快捷键应已安装."""
    win = MainWindow()
    qtbot.addWidget(win)
    shortcuts = [obj for obj in win.children() if hasattr(obj, "activated") and "QShortcut" in type(obj).__name__]
    # 快捷键总数：Ctrl+1/2/3 + Ctrl+PgDn/PgUp + Ctrl+B + F5 = 7
    assert len(shortcuts) >= 7, f"Expected >=7 shortcuts, got {len(shortcuts)}"


@pytest.mark.gui
def test_main_window_f5_global_run_dispatches(qtbot, isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F5 应按当前页分发给对应 run 方法."""
    win = MainWindow()
    qtbot.addWidget(win)

    # patch 三个 run 方法为 spy，捕获调用
    calls = {"nb": 0, "tp": 0, "msg": []}

    def _spy_nb():
        calls["nb"] += 1

    def _spy_tp():
        calls["tp"] += 1

    def _spy_msg(text):
        calls["msg"].append(text)

    monkeypatch.setattr(win._notebook_page, "run_all", _spy_nb)
    monkeypatch.setattr(win._template_page, "run", _spy_tp)
    monkeypatch.setattr(win.statusBar(), "showMessage", _spy_msg, raising=False)

    # 笔记本页 → run_all
    win._sidebar.setCurrentRow(0)
    win._global_run()
    assert calls["nb"] == 1
    assert calls["tp"] == 0
    assert any("笔记本" in m for m in calls["msg"])

    # 工作台页 → 提示不支持
    calls["nb"] = 0
    calls["tp"] = 0
    calls["msg"].clear()
    win._sidebar.setCurrentRow(1)
    win._global_run()
    assert calls["nb"] == 0
    assert calls["tp"] == 0
    assert any("工作台" in m for m in calls["msg"])

    # 模板页 → run()
    calls["nb"] = 0
    calls["tp"] = 0
    calls["msg"].clear()
    win._sidebar.setCurrentRow(2)
    win._global_run()
    assert calls["nb"] == 0
    assert calls["tp"] == 1
    assert any("模板" in m for m in calls["msg"])
