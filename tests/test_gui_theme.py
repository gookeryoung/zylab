"""gui.theme 主题系统测试：对比度达标（WCAG AA）、令牌完整性、运行时切换."""

from __future__ import annotations

from pathlib import Path

import pytest

from zylab.gui import theme
from zylab.gui.app import apply_theme, load_stylesheet, load_theme_name, save_theme_name

ALL_THEMES = tuple(theme.THEMES.values())


# ---------------------------------------------------------------------------
# 对比度数学
# ---------------------------------------------------------------------------


class TestContrastRatio:
    def test_black_white_is_21(self) -> None:
        """纯黑白对比度应为 21:1."""
        assert theme.contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)

    def test_same_color_is_1(self) -> None:
        """同色对比度应为 1:1."""
        assert theme.contrast_ratio("#ABCDEF", "#ABCDEF") == pytest.approx(1.0)

    def test_symmetric(self) -> None:
        """对比度与参数顺序无关."""
        assert theme.contrast_ratio("#056574", "#FFFFFF") == pytest.approx(theme.contrast_ratio("#FFFFFF", "#056574"))


# ---------------------------------------------------------------------------
# 每主题关键配对全部达标 WCAG AA（正文 >= 4.5:1）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pal", ALL_THEMES, ids=lambda p: p.name)
class TestThemeAccessibility:
    def test_primary_text_on_app(self, pal: theme.Palette) -> None:
        """正文文字对窗口底色 >= 4.5:1."""
        assert theme.contrast_ratio(pal.text_primary, pal.bg_app) >= 4.5

    def test_secondary_text_on_app_and_muted(self, pal: theme.Palette) -> None:
        """次要文字对窗口底与分组底均 >= 4.5:1（状态栏可读性）."""
        assert theme.contrast_ratio(pal.text_secondary, pal.bg_app) >= 4.5
        assert theme.contrast_ratio(pal.text_secondary, pal.bg_muted) >= 4.5

    def test_text_on_input(self, pal: theme.Palette) -> None:
        """输入控件文字对输入底 >= 4.5:1."""
        assert theme.contrast_ratio(pal.text_primary, pal.bg_input) >= 4.5

    def test_button_text(self, pal: theme.Palette) -> None:
        """主按钮文字对主色 >= 4.5:1."""
        assert theme.contrast_ratio(pal.primary_text, pal.primary) >= 4.5

    def test_nav_text(self, pal: theme.Palette) -> None:
        """导航文字对导航底（含悬停/选中态）>= 4.5:1."""
        assert theme.contrast_ratio(pal.nav_text, pal.nav_bg) >= 4.5
        assert theme.contrast_ratio(pal.nav_text, pal.nav_bg_hover) >= 4.5
        assert theme.contrast_ratio(pal.nav_text, pal.nav_bg_selected) >= 4.5

    def test_header_on_nav(self, pal: theme.Palette) -> None:
        """头部标题/元信息对导航底 >= 4.5:1."""
        assert theme.contrast_ratio(pal.nav_text, pal.nav_bg) >= 4.5

    def test_selection(self, pal: theme.Palette) -> None:
        """选区文字对选区底 >= 4.5:1."""
        assert theme.contrast_ratio(pal.selection_text, pal.selection_bg) >= 4.5

    def test_status_text_on_muted(self, pal: theme.Palette) -> None:
        """状态栏文字（次要文字）对状态栏底 >= 4.5:1."""
        assert theme.contrast_ratio(pal.text_secondary, pal.bg_muted) >= 4.5

    def test_status_colors_readable(self, pal: theme.Palette) -> None:
        """状态色（成功/警告/危险/错误）作文字时对窗口底 >= 4.5:1."""
        for color in (pal.success_text, pal.warning_text, pal.danger_text, pal.error_text):
            assert theme.contrast_ratio(color, pal.bg_app) >= 4.5

    def test_nav_accent_visible(self, pal: theme.Palette) -> None:
        """导航强调竖条对导航底 >= 3:1（图形元素达标线）."""
        assert theme.contrast_ratio(pal.nav_accent, pal.nav_bg) >= 3.0

    def test_progress_chunk_visible(self, pal: theme.Palette) -> None:
        """进度条滑块对进度条底 >= 3:1（图形元素达标线）."""
        assert theme.contrast_ratio(pal.primary, pal.bg_muted) >= 3.0

    def test_disabled_text(self, pal: theme.Palette) -> None:
        """禁用文字对窗口底 >= 3:1（豁免正文线，但保持可辨）."""
        assert theme.contrast_ratio(pal.text_disabled, pal.bg_app) >= 3.0


# ---------------------------------------------------------------------------
# 主题注册与切换
# ---------------------------------------------------------------------------


class TestThemeRegistry:
    def test_three_themes_registered(self) -> None:
        """应注册浅色/深色/高对比三套主题，且色板互不相同."""
        assert set(theme.THEMES) == {"light", "dark", "high_contrast"}
        palettes = {tuple(sorted(vars(p).items())) for p in ALL_THEMES}
        assert len(palettes) == 3

    def test_palette_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="未知主题"):
            theme.palette("solarized")

    def test_set_current_theme_roundtrip(self) -> None:
        """set_current_theme/current_palette 应一致（结束后还原默认）."""
        try:
            theme.set_current_theme("dark")
            assert theme.current_palette().name == "dark"
        finally:
            theme.set_current_theme(theme.DEFAULT_THEME)

    def test_set_current_theme_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="未知主题"):
            theme.set_current_theme("neon")


# ---------------------------------------------------------------------------
# QSS 令牌完整性
# ---------------------------------------------------------------------------


class TestQssTokens:
    @pytest.mark.parametrize("pal", ALL_THEMES, ids=lambda p: p.name)
    def test_substitute_no_residual(self, pal: theme.Palette) -> None:
        """任一主题替换后 QSS 不应残留占位符（缺失令牌会抛 KeyError）."""
        qss = load_stylesheet(pal)
        assert "${" not in qss
        assert qss.count("#") >= 30  # 颜色已展开

    @pytest.mark.parametrize("pal", ALL_THEMES, ids=lambda p: p.name)
    def test_theme_colors_present(self, pal: theme.Palette) -> None:
        """QSS 应包含当前主题的关键色值."""
        qss = load_stylesheet(pal)
        assert pal.bg_app.lower() in qss.lower()
        assert pal.primary.lower() in qss.lower()
        assert pal.nav_bg.lower() in qss.lower()

    def test_dark_differs_from_light(self) -> None:
        """深色主题样式表应不同于浅色."""
        assert load_stylesheet(theme.LIGHT) != load_stylesheet(theme.DARK)


# ---------------------------------------------------------------------------
# 运行时切换与持久化
# ---------------------------------------------------------------------------


class TestApplyTheme:
    def test_switch_updates_stylesheet(self, qapp) -> None:
        """切换主题后应用样式表应随主题变化."""
        try:
            apply_theme(qapp, "dark")
            dark_qss = qapp.styleSheet()
            assert theme.DARK.bg_app.lower() in dark_qss.lower()
            apply_theme(qapp, "light")
            assert qapp.styleSheet() != dark_qss
        finally:
            apply_theme(qapp, theme.DEFAULT_THEME)

    def test_invalid_name_keeps_state(self, qapp) -> None:
        """未知名抛错且当前主题不变."""
        before = theme.current_palette().name
        with pytest.raises(ValueError, match="未知主题"):
            apply_theme(qapp, "neon")
        assert theme.current_palette().name == before


class TestThemePersistence:
    def test_roundtrip(self, tmp_path: Path) -> None:
        save_theme_name(tmp_path, "dark")
        assert (tmp_path / "theme.txt").read_text(encoding="utf-8") == "dark"
        assert load_theme_name(tmp_path) == "dark"

    def test_missing_file_falls_back(self, tmp_path: Path) -> None:
        assert load_theme_name(tmp_path) == theme.DEFAULT_THEME

    def test_corrupted_file_falls_back(self, tmp_path: Path) -> None:
        (tmp_path / "theme.txt").write_text("nonexistent-theme\n", encoding="utf-8")
        assert load_theme_name(tmp_path) == theme.DEFAULT_THEME

    def test_save_invalid_name_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="未知主题"):
            save_theme_name(tmp_path, "neon")


# ---------------------------------------------------------------------------
# 主窗口主题下拉框
# ---------------------------------------------------------------------------


class TestMainWindowThemeSwitch:
    def test_select_theme(self, qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """下拉选择主题应立即应用并持久化到数据目录."""
        monkeypatch.setattr("zylab.gui.main_window.default_data_dir", lambda: tmp_path)
        from zylab.gui.main_window import MainWindow

        win = MainWindow()
        qtbot.addWidget(win)
        assert theme.current_palette().name == "light"
        # 初始下拉应显示当前主题
        assert win._theme_combo.currentIndex() == 0
        # 选择深色
        win._theme_combo.setCurrentIndex(1)
        assert theme.current_palette().name == "dark"
        assert (tmp_path / "theme.txt").read_text(encoding="utf-8") == "dark"
        # 选择高对比
        win._theme_combo.setCurrentIndex(2)
        assert theme.current_palette().name == "high_contrast"
        # 选回浅色
        win._theme_combo.setCurrentIndex(0)
        assert theme.current_palette().name == "light"
        theme.set_current_theme(theme.DEFAULT_THEME)
