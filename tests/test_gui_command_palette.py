"""gui.widgets.command_palette 命令面板测试：注册/过滤/执行/主题预览/Esc 还原."""

from __future__ import annotations

from zylab.gui import theme
from zylab.gui.qt_compat import Qt, QWidget
from zylab.gui.widgets.command_palette import Command, CommandPalette


def _make_palette(qtbot) -> tuple[CommandPalette, QWidget]:
    """构建附着于占位父窗口的命令面板（返回父窗口以维持其存活）.

    pytest-qt 的 addWidget 仅存弱引用，父窗口若无强引用会被 GC 级联
    销毁面板的 C++ 对象，故调用方须在测试全程持有返回的父窗口。
    """
    parent = QWidget()
    qtbot.addWidget(parent)
    pal = CommandPalette(parent)
    qtbot.addWidget(pal)
    return pal, parent


def _theme_row(pal: CommandPalette, name: str) -> int:
    """定位主题名对应的列表行."""
    for row in range(pal._list.count()):
        info = pal._list.item(row).data(Qt.UserRole)
        if info[0] == "theme" and info[1] == name:
            return row
    raise AssertionError(f"主题 {name} 不在面板列表中")


# ---------------------------------------------------------------------------
# 命令注册与过滤
# ---------------------------------------------------------------------------


class TestCommandRegistry:
    def test_register_and_overwrite_keeps_position(self, qtbot) -> None:
        """同 id 重复注册应覆盖定义且保持首次注册顺序."""
        pal, _parent = _make_palette(qtbot)
        pal.register(Command("a", "命令A", lambda: None))
        pal.register(Command("b", "命令B", lambda: None))
        pal.register(Command("a", "命令A（新）", lambda: None, keywords="alpha"))
        assert [c.id for c in pal._commands] == ["a", "b"]
        assert pal._commands[0].title == "命令A（新）"
        assert pal._commands[0].keywords == "alpha"

    def test_filter_by_title_and_keywords(self, qtbot) -> None:
        """过滤应同时命中中文标题与英文关键词，多词取交集."""
        pal, _parent = _make_palette(qtbot)
        pal.register(Command("go.notebook", "转到：笔记本", lambda: None, keywords="goto notebook"))
        pal.register(Command("notebook.new", "新建笔记本", lambda: None, keywords="new notebook"))
        pal.register(Command("go.about", "转到：关于", lambda: None, keywords="goto about"))
        pal.open_commands()
        assert pal._list.count() == 3
        pal._search.setText("笔记本")
        assert pal._list.count() == 2
        pal._search.setText("notebook new")  # 英文关键词多词交集
        assert pal._list.count() == 1
        pal._search.setText("转到 about")
        assert pal._list.count() == 1
        pal._search.setText("不存在的命令")
        assert pal._list.count() == 0

    def test_execute_command_via_enter(self, qtbot) -> None:
        """搜索框按 Enter 应执行选中命令并关闭面板."""
        pal, _parent = _make_palette(qtbot)
        calls: list[str] = []
        pal.register(Command("demo.run", "执行我", lambda: calls.append("run"), shortcut="Ctrl+R"))
        pal.open_commands()
        qtbot.keyPress(pal._search, Qt.Key_Return)
        assert calls == ["run"]
        assert not pal.isVisible()

    def test_theme_command_prefix_enter_theme_mode(self, qtbot) -> None:
        """键入 ``>`` 前缀应进入主题模式并列出全部主题."""
        pal, _parent = _make_palette(qtbot)
        pal.open_commands()
        pal._search.setText(">")
        assert pal._list.count() == len(theme.THEMES)


# ---------------------------------------------------------------------------
# 主题模式：预览 / 确认 / Esc 还原
# ---------------------------------------------------------------------------


class TestThemeMode:
    def test_open_theme_picker_selects_current(self, qtbot) -> None:
        """主题模式默认静默选中当前主题（不触发预览信号）."""
        pal, _parent = _make_palette(qtbot)
        try:
            theme.set_current_theme("dark")
            previewed: list[str] = []
            pal.theme_previewed.connect(previewed.append)
            pal.open_theme_picker()
            assert pal._list.currentItem().data(Qt.UserRole) == ("theme", "dark")
            assert previewed == []  # 静默选中不应触发预览
        finally:
            theme.set_current_theme(theme.DEFAULT_THEME)

    def test_navigation_previews(self, qtbot) -> None:
        """主题模式上下导航应实时发出预览信号."""
        pal, _parent = _make_palette(qtbot)
        try:
            theme.set_current_theme("light")
            pal.open_theme_picker()
            previewed: list[str] = []
            pal.theme_previewed.connect(previewed.append)
            other = next(name for name in theme.THEMES if name != "light")
            pal._list.setCurrentRow(_theme_row(pal, other))
            assert previewed[-1] == other
        finally:
            theme.set_current_theme(theme.DEFAULT_THEME)

    def test_enter_confirms_theme(self, qtbot) -> None:
        """主题模式按 Enter 应发出确认信号并关闭面板."""
        pal, _parent = _make_palette(qtbot)
        confirmed: list[str] = []
        pal.theme_confirmed.connect(confirmed.append)
        pal.open_theme_picker()
        qtbot.keyPress(pal._search, Qt.Key_Return)
        assert confirmed == [theme.current_palette().name]
        assert not pal.isVisible()

    def test_theme_filter_narrows_list(self, qtbot) -> None:
        """主题模式下继续输入应按显示名/主题名过滤."""
        pal, _parent = _make_palette(qtbot)
        pal.open_theme_picker()
        pal._search.setText(">dark")
        infos = [pal._list.item(r).data(Qt.UserRole)[1] for r in range(pal._list.count())]
        assert infos == ["dark"]

    def test_esc_restores_original_theme(self, qtbot) -> None:
        """主题预览后按 Esc 应关闭面板并还原原主题（主窗口语义模拟）."""
        pal, _parent = _make_palette(qtbot)
        # 模拟主窗口：预览/还原信号直接应用主题
        pal.theme_previewed.connect(theme.set_current_theme)
        try:
            theme.set_current_theme("light")
            pal.open_theme_picker()
            pal._list.setCurrentRow(_theme_row(pal, "dark"))
            assert theme.current_palette().name == "dark"  # 导航预览已应用
            qtbot.keyPress(pal._search, Qt.Key_Escape)
            assert theme.current_palette().name == "light"  # Esc 还原
        finally:
            theme.set_current_theme(theme.DEFAULT_THEME)

    def test_esc_without_change_no_restore_signal(self, qtbot) -> None:
        """未切换预览直接 Esc 不应发出还原信号."""
        pal, _parent = _make_palette(qtbot)
        previewed: list[str] = []
        pal.theme_previewed.connect(previewed.append)
        pal.open_theme_picker()
        qtbot.keyPress(pal._search, Qt.Key_Escape)
        assert previewed == []
