"""gui.theme 主题系统测试：对比度达标（WCAG AA）、令牌完整性、JSON 扩展、运行时切换."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from zylab.gui import theme
from zylab.gui.app import apply_theme, load_stylesheet, load_theme_name, save_theme_name
from zylab.gui.qt_compat import Qt

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

    @pytest.mark.parametrize("pal", ALL_THEMES, ids=lambda p: p.name)
    def test_arrow_svg_tokens_resolved(self, pal: theme.Palette) -> None:
        """箭头 SVG 资源路径应注入 QSS 且文件存在、颜色随主题."""
        qss = load_stylesheet(pal)
        urls = re.findall(r"image: url\(([^)]+\.svg)\)", qss)
        assert len(urls) == 3  # 通用下拉 + spinbox 上下（主题下拉箭头已随命令面板移除）
        for url in urls:
            assert Path(url).exists()
        # 生成的 SVG 含当前主题次级文字色
        content = Path(urls[0]).read_text(encoding="utf-8")
        assert pal.text_secondary.lstrip("#").lower() in content.lower()

    def test_arrow_svg_isolated_per_process_and_theme(self) -> None:
        """箭头文件名须含进程号+主题名（xdist 并行写同一文件曾致半截 SVG）."""
        from zylab.gui.app import _write_arrow_svgs

        light = _write_arrow_svgs(theme.LIGHT)
        dark = _write_arrow_svgs(theme.DARK)
        assert light["QSS_ARROW_DOWN"] != dark["QSS_ARROW_DOWN"]
        # 切换主题后旧主题文件被清理（同进程），新主题文件存在
        assert not Path(light["QSS_ARROW_DOWN"]).exists()
        assert Path(dark["QSS_ARROW_DOWN"]).exists()


# ---------------------------------------------------------------------------
# 箭头形状回归（Qt QSS 不支持 border 画三角，曾整体渲染成实心方块）
# ---------------------------------------------------------------------------


@pytest.mark.gui
class TestArrowShapes:
    @staticmethod
    def _dark_zone(widget, right_w: int = 24) -> list[int]:
        """抓取控件右侧按钮区，返回每行深色像素数（形状签名）."""
        img = widget.grab().toImage()
        w, h = img.width(), img.height()
        rows: list[int] = []
        for y in range(h):
            count = 0
            for x in range(w - right_w, w):
                c = img.pixelColor(x, y)
                if c.alpha() > 50 and (c.red() + c.green() + c.blue()) < 400:
                    count += 1
            if count:
                rows.append(count)
        return rows

    def test_spinbox_arrows_are_triangles(self, qtbot) -> None:
        """SpinBox 上下箭头应为三角形（逐行变宽/变窄），不是恒宽方块."""
        from zylab.gui.app import create_app
        from zylab.gui.qt_compat import QDoubleSpinBox

        create_app()
        spin = QDoubleSpinBox()
        qtbot.addWidget(spin)
        spin.resize(140, 28)
        spin.show()
        rows = self._dark_zone(spin)
        # 上下两个三角：中间宽两端窄，且存在宽度差（方块则恒定）
        assert rows, "箭头未渲染"
        assert max(rows) > min(rows) * 2, f"箭头形状异常（疑似方块）: {rows}"

    def test_combobox_arrow_is_triangle(self, qtbot) -> None:
        """ComboBox 下拉箭头应为下指三角."""
        from zylab.gui.app import create_app
        from zylab.gui.qt_compat import QComboBox

        create_app()
        combo = QComboBox()
        combo.addItem("示例")
        qtbot.addWidget(combo)
        combo.resize(140, 28)
        combo.show()
        rows = self._dark_zone(combo)
        assert rows, "箭头未渲染"
        assert max(rows) > min(rows) * 2, f"箭头形状异常（疑似方块）: {rows}"


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
# 主题 JSON 扩展（assets/themes 内置 + 用户目录扩展）
# ---------------------------------------------------------------------------


class TestThemeJsonLoading:
    def test_partial_override_inherits_base(self, tmp_path: Path) -> None:
        """同名主题部分字段 JSON 应继承基底色板其余字段."""
        (tmp_path / "dark_tweaked.json").write_text(
            json.dumps({"name": "dark", "display_name": "深色微调", "primary": "#123456"}),
            encoding="utf-8",
        )
        themes = theme.load_themes_from_dir(tmp_path, base={"dark": theme.DARK})
        pal = themes["dark"]
        assert pal.display_name == "深色微调"
        assert pal.primary == "#123456"
        assert pal.bg_app == theme.DARK.bg_app  # 未覆盖字段继承基底

    def test_invalid_files_skipped(self, tmp_path: Path) -> None:
        """非法 JSON / 未知字段 / 缺必填字段均跳过，不影响其余主题."""
        (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
        (tmp_path / "unknown_field.json").write_text(
            json.dumps({"name": "x", "display_name": "X", "bogus": 1}),
            encoding="utf-8",
        )
        (tmp_path / "noname.json").write_text(json.dumps({"display_name": "无名"}), encoding="utf-8")
        themes = theme.load_themes_from_dir(tmp_path, base={"dark": theme.DARK})
        assert set(themes) == {"dark"}

    def test_order_controls_sorting(self, tmp_path: Path) -> None:
        """``order`` 字段应控制应用顺序（小者先应用，大者后覆盖同名主题）."""
        (tmp_path / "b_second.json").write_text(
            json.dumps({"name": "b", "display_name": "B", "order": 20}),
            encoding="utf-8",
        )
        (tmp_path / "a_first.json").write_text(
            json.dumps({"name": "b", "display_name": "B2", "order": 10}),
            encoding="utf-8",
        )
        themes = theme.load_themes_from_dir(tmp_path, base={"b": theme.DARK})
        assert themes["b"].display_name == "B"  # order 大者后应用并覆盖

    def test_register_theme_dir_extends(self, tmp_path: Path) -> None:
        """register_theme_dir 应追加新主题并支持按名取色板（结束后清理）."""
        src = json.loads((theme._THEMES_DIR / "dark.json").read_text(encoding="utf-8"))
        src.update({"name": "solarized", "display_name": "Solarized", "primary": "#268BD2"})
        (tmp_path / "solarized.json").write_text(json.dumps(src), encoding="utf-8")
        try:
            assert theme.register_theme_dir(tmp_path) == ["solarized"]
            assert theme.palette("solarized").primary == "#268BD2"
        finally:
            theme.THEMES.pop("solarized", None)

    def test_register_user_themes_app_hook(self, tmp_path: Path) -> None:
        """app.register_user_themes 应注册数据目录 ``themes/``；缺失目录返回空."""
        from zylab.gui.app import register_user_themes

        assert register_user_themes(tmp_path) == []  # 目录不存在
        (tmp_path / "themes").mkdir()
        src = json.loads((theme._THEMES_DIR / "light.json").read_text(encoding="utf-8"))
        src.update({"name": "user_custom", "display_name": "用户自定义"})
        (tmp_path / "themes" / "user_custom.json").write_text(json.dumps(src), encoding="utf-8")
        try:
            assert register_user_themes(tmp_path) == ["user_custom"]
            assert theme.palette("user_custom").display_name == "用户自定义"
        finally:
            theme.THEMES.pop("user_custom", None)


# ---------------------------------------------------------------------------
# 主窗口命令面板主题切换
# ---------------------------------------------------------------------------


class TestMainWindowThemeSwitch:
    def test_palette_theme_confirm_persists(self, qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """命令面板确认主题应立即应用并持久化到数据目录."""
        monkeypatch.setattr("zylab.gui.main_window.default_data_dir", lambda: tmp_path)
        from zylab.gui.main_window import MainWindow

        try:
            win = MainWindow()
            qtbot.addWidget(win)
            assert theme.current_palette().name == "light"
            win._palette.open_theme_picker()
            # 默认选中当前主题
            assert win._palette._list.currentItem().data(Qt.UserRole) == ("theme", "light")
            # 选中深色并确认
            row = next(
                r
                for r in range(win._palette._list.count())
                if win._palette._list.item(r).data(Qt.UserRole)[1] == "dark"
            )
            win._palette._list.setCurrentRow(row)
            win._palette._activate_current(win._palette._list.currentItem())
            assert theme.current_palette().name == "dark"
            assert (tmp_path / "theme.txt").read_text(encoding="utf-8") == "dark"
        finally:
            theme.set_current_theme(theme.DEFAULT_THEME)

    def test_palette_theme_preview_not_persisted(self, qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """主题预览仅应用不持久化；Esc 还原原主题."""
        monkeypatch.setattr("zylab.gui.main_window.default_data_dir", lambda: tmp_path)
        from zylab.gui.main_window import MainWindow

        try:
            win = MainWindow()
            qtbot.addWidget(win)
            win._palette.open_theme_picker()
            row = next(
                r
                for r in range(win._palette._list.count())
                if win._palette._list.item(r).data(Qt.UserRole)[1] == "dark"
            )
            win._palette._list.setCurrentRow(row)  # 导航预览
            assert theme.current_palette().name == "dark"
            assert not (tmp_path / "theme.txt").exists()
            qtbot.keyPress(win._palette._search, Qt.Key_Escape)  # 取消还原
            assert theme.current_palette().name == "light"
        finally:
            theme.set_current_theme(theme.DEFAULT_THEME)
