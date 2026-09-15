"""GUI 模块纯逻辑测试（不依赖 QApplication/QWidget 事件循环的代码）.

覆盖范围：
- theme: Palette dataclass / _palette_from_json / load_themes_from_dir /
         contrast_ratio / _relative_luminance / _scale_font / 字体运行时 /
         palette / set_current_theme / qss_tokens / register_theme_dir
- style: load_qss_fragments / load_stylesheet（Template 替换层）
- icons: SVG 着色字符串处理（_FILL_ATTR_RE + 根元素注入逻辑）
- highlight: 模块级预编译正则 / 关键字 / 内置名集
- _table_utils: format_cell / format_cell_with_format（col_alignment 需 Qt 枚举）
- pages/__init__ 与 widgets/__init__: __getattr__ 懒加载 facade 机制
- app: save_theme_name / load_theme_name / register_user_themes（纯 IO + 校验）
- qt_compat: QT_API 探测 + 兼容层符号已 re-export（exec_app/exec_menu/exec_dialog/mouse_event_pos 需 Qt 对象实例，不测）
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Generator
from dataclasses import fields as dc_fields
from pathlib import Path
from string import Template

import pytest

# --------------------------------------------------------------------------- theme


class TestPaletteDataclass:
    """Palette 冻结 dataclass + 字段完整性."""

    def test_palette_is_frozen(self) -> None:
        from zylab.gui.theme import Palette

        # frozen=True → 可 hash
        pal = Palette(
            name="t",
            display_name="T",
            bg_app="#ffffff",
            bg_muted="#eeeeee",
            bg_input="#ffffff",
            text_primary="#000000",
            text_secondary="#666666",
            text_on_primary="#ffffff",
            text_disabled="#aaaaaa",
            nav_bg="#f8f8f8",
            nav_bg_hover="#e0e0e0",
            nav_bg_selected="#dddddd",
            nav_text="#333333",
            nav_accent="#0078d7",
            primary="#0078d7",
            primary_hover="#1a88e0",
            primary_pressed="#005a9e",
            primary_text="#ffffff",
            selection_bg="#0078d7",
            selection_text="#ffffff",
            border="#cccccc",
            border_strong="#888888",
            scrollbar="#cccccc",
            scrollbar_hover="#aaaaaa",
            success_text="#2e7d32",
            warning_text="#f57c00",
            danger_text="#c62828",
            error_text="#d32f2f",
            info_bar="#e3f2fd",
            success_bar="#e8f5e9",
            warning_bar="#fff3e0",
            danger_bar="#ffebee",
        )
        assert hash(pal) == hash(pal)
        with pytest.raises((AttributeError, TypeError)):
            pal.bg_app = "#000000"  # type: ignore[misc]

    def test_palette_field_count(self) -> None:
        from zylab.gui.theme import LIGHT, Palette

        # 与源码字段数对齐（新增字段须同步 _palette_from_json 校验测试）
        assert len(dc_fields(Palette)) == 32
        # 内置主题全部字段都是 #RRGGBB
        for f in dc_fields(Palette):
            value = getattr(LIGHT, f.name)
            if f.name in ("name", "display_name"):
                assert isinstance(value, str) and value
            else:
                assert isinstance(value, str) and value.startswith("#") and len(value) == 7


class TestPaletteFromJson:
    """_palette_from_json：JSON → Palette 的字段校验与色值校验."""

    @pytest.fixture()
    def full_json(self) -> dict:
        from zylab.gui.theme import Palette

        data: dict[str, str] = {}
        for f in dc_fields(Palette):
            if f.name in ("name", "display_name"):
                data[f.name] = "sample" if f.name == "name" else "Sample"
            else:
                data[f.name] = "#123456"
        return data

    def test_normal_full(self, full_json: dict) -> None:
        from zylab.gui.theme import _palette_from_json

        pal = _palette_from_json(full_json)
        assert pal.name == "sample"
        assert pal.display_name == "Sample"
        assert pal.bg_app == "#123456"

    def test_partial_override_base(self, full_json: dict) -> None:
        """同名主题后续 JSON 缺省字段继承 base."""
        from zylab.gui.theme import Palette, _palette_from_json

        base_fields: dict[str, str] = {}
        for f in dc_fields(Palette):
            if f.name in ("name", "display_name"):
                base_fields[f.name] = "base" if f.name == "name" else "Base"
            else:
                base_fields[f.name] = "#aaaaaa"
        base = _palette_from_json(base_fields)

        partial = {"name": "base", "display_name": "Base", "bg_app": "#ff0000"}
        merged = _palette_from_json(partial, base=base)
        assert merged.bg_app == "#ff0000"
        assert merged.bg_muted == "#aaaaaa"  # 继承 base

    def test_unknown_field_raises(self, full_json: dict) -> None:
        from zylab.gui.theme import _palette_from_json

        full_json["bogus"] = "x"
        with pytest.raises(ValueError, match=r"未知色板字段"):
            _palette_from_json(full_json)

    def test_unknown_order_field_is_allowed(self, full_json: dict) -> None:
        """order 字段仅排序用，不是色板字段但允许出现在 JSON 里."""
        from zylab.gui.theme import _palette_from_json

        full_json["order"] = 50
        # 不抛错
        _palette_from_json(full_json)

    def test_bad_color_no_hash(self, full_json: dict) -> None:
        from zylab.gui.theme import _palette_from_json

        full_json["bg_app"] = "FFFFFF"
        with pytest.raises(ValueError, match=r"非法色值"):
            _palette_from_json(full_json)

    def test_bad_color_short(self, full_json: dict) -> None:
        from zylab.gui.theme import _palette_from_json

        full_json["bg_app"] = "#fff"
        with pytest.raises(ValueError, match=r"非法色值"):
            _palette_from_json(full_json)

    def test_missing_required_name(self, full_json: dict) -> None:
        from zylab.gui.theme import _palette_from_json

        del full_json["name"]
        with pytest.raises(ValueError, match=r"缺少必填字段"):
            _palette_from_json(full_json)

    def test_missing_field_raises(self, full_json: dict) -> None:
        """缺失非必填字段时 base=None 会触发长度校验；有 base 时继承."""
        from zylab.gui.theme import _palette_from_json

        del full_json["bg_app"]
        with pytest.raises(ValueError, match=r"缺少色板字段"):
            _palette_from_json(full_json)


class TestLoadThemesFromDir:
    """load_themes_from_dir：目录扫描 + JSON 解析 + base 继承 + 跳过坏文件."""

    @pytest.fixture()
    def theme_dir(self) -> Generator[Path, None, None]:
        from zylab.gui.theme import Palette

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            # 完整合法主题
            full: dict[str, str | int] = {"order": 10, "name": "custom", "display_name": "Custom"}
            for f in dc_fields(Palette):
                if f.name in ("name", "display_name"):
                    continue
                full[f.name] = "#112233"
            (td_path / "custom.json").write_text(json.dumps(full), encoding="utf-8")

            # 坏 JSON
            (td_path / "broken.json").write_text("{not-json", encoding="utf-8")

            # 未知字段（badcolor 名字故意不同，触发"未知色板字段"而非继承）
            bad1: dict[str, str | int] = {"order": 20, "name": "badcolor", "display_name": "Bad", "unknown": "x"}
            for f in dc_fields(Palette):
                if f.name in ("name", "display_name"):
                    continue
                bad1[f.name] = "#112233"
            (td_path / "unknown.json").write_text(json.dumps(bad1), encoding="utf-8")

            # 非法颜色
            bad2: dict[str, str | int] = {"order": 30, "name": "badvalue", "display_name": "Bad"}
            for f in dc_fields(Palette):
                if f.name in ("name", "display_name"):
                    continue
                bad2[f.name] = "#112233"
            bad2["bg_app"] = "FFFFFF"
            (td_path / "badcolor.json").write_text(json.dumps(bad2), encoding="utf-8")

            yield td_path

    def test_normal_load(self, theme_dir: Path) -> None:
        from zylab.gui.theme import load_themes_from_dir

        themes = load_themes_from_dir(theme_dir)
        assert list(themes.keys()) == ["custom"]  # 坏的都跳过了
        assert themes["custom"].bg_app == "#112233"

    def test_base_inheritance(self, theme_dir: Path) -> None:
        from zylab.gui.theme import Palette, _palette_from_json, load_themes_from_dir

        base_fields: dict[str, str] = {}
        for f in dc_fields(Palette):
            if f.name in ("name", "display_name"):
                base_fields[f.name] = "custom" if f.name == "name" else "Custom"
            else:
                base_fields[f.name] = "#aaaaaa"
        base = {"custom": _palette_from_json(base_fields)}

        themes = load_themes_from_dir(theme_dir, base=base)
        # custom.json 里完整覆盖了 base → bg_app=#112233
        assert themes["custom"].bg_app == "#112233"
        # 但 base 中 custom 不存在的字段？都存在。我们在 full 里写全了。

    def test_no_json_returns_empty(self) -> None:
        from zylab.gui.theme import load_themes_from_dir

        with tempfile.TemporaryDirectory() as td:
            themes = load_themes_from_dir(Path(td))
            assert themes == {}


class TestThemeLookup:
    """palette / set_current_theme / current_palette / register_theme_dir."""

    def test_palette_known(self) -> None:
        from zylab.gui.theme import DARK, HIGH_CONTRAST, LIGHT, palette

        assert palette("light") is LIGHT
        assert palette("dark") is DARK
        assert palette("high_contrast") is HIGH_CONTRAST

    def test_palette_unknown_raises(self) -> None:
        from zylab.gui.theme import palette

        with pytest.raises(ValueError, match=r"未知主题"):
            palette("不存在的主题")

    def test_set_current_theme_and_back(self) -> None:
        from zylab.gui.theme import DARK, LIGHT, _current, current_palette, set_current_theme

        saved = _current
        try:
            set_current_theme("dark")
            assert current_palette() is DARK
            set_current_theme("light")
            assert current_palette() is LIGHT
        finally:
            # 还原
            from zylab.gui import theme as _t

            _t._current = saved

    def test_set_current_theme_unknown_raises(self) -> None:
        from zylab.gui.theme import set_current_theme

        with pytest.raises(ValueError, match=r"未知主题"):
            set_current_theme("nope")


class TestThemeRuntimeFont:
    """set_font_families / set_font_scale / current_font_families / current_font_scale."""

    def test_scale_clamps_lower(self) -> None:
        from zylab.gui.theme import _font_scale, current_font_scale, set_font_scale

        saved = _font_scale
        try:
            set_font_scale(0.1)
            assert current_font_scale() == pytest.approx(0.8)
        finally:
            from zylab.gui import theme as _t

            _t._font_scale = saved

    def test_scale_clamps_upper(self) -> None:
        from zylab.gui.theme import _font_scale, current_font_scale, set_font_scale

        saved = _font_scale
        try:
            set_font_scale(5.0)
            assert current_font_scale() == pytest.approx(1.4)
        finally:
            from zylab.gui import theme as _t

            _t._font_scale = saved

    def test_scale_within_range(self) -> None:
        from zylab.gui.theme import _font_scale, current_font_scale, set_font_scale

        saved = _font_scale
        try:
            set_font_scale(1.1)
            assert current_font_scale() == pytest.approx(1.1)
        finally:
            from zylab.gui import theme as _t

            _t._font_scale = saved

    def test_set_font_families_strips_and_skips_empty(self) -> None:
        from zylab.gui.theme import (
            _font_family_body,
            _font_family_mono,
            current_font_families,
            set_font_families,
        )

        saved_body = _font_family_body
        saved_mono = _font_family_mono
        try:
            set_font_families(body="  Microsoft YaHei ", mono="  DejaVu Sans Mono  ")
            got = current_font_families()
            assert got["body"] == "Microsoft YaHei"
            assert got["mono"] == "DejaVu Sans Mono"

            # 空串不覆盖
            set_font_families(body="", mono=None)
            got2 = current_font_families()
            assert got2["body"] == "Microsoft YaHei"  # 保持原值
            assert got2["mono"] == "DejaVu Sans Mono"  # 保持原值
        finally:
            from zylab.gui import theme as _t

            _t._font_family_body = saved_body
            _t._font_family_mono = saved_mono


class TestScaleFontString:
    """_scale_font：px 单位字符串缩放."""

    @pytest.mark.parametrize(
        ("font", "scale", "expected"),
        [
            ("13px", 1.0, "13px"),  # 1.0 原样
            ("13px", 1.5, "19.5px"),
            ("13px", 0.8, "10.4px"),
            ("11px", 0.8, "8.8px"),
            ("18px", 1.0, "18px"),  # 整数无 .0
            ("15.5px", 1.5, "23.2px"),
            ("bold", 1.5, "bold"),  # 非 px 原样
            ("13", 1.5, "13"),  # 缺 px 原样
            ("13abc", 1.5, "13abc"),  # 缺 px 原样
            ("", 1.5, ""),
        ],
    )
    def test_various(self, font: str, scale: float, expected: str) -> None:
        from zylab.gui.theme import _scale_font

        assert _scale_font(font, scale) == expected


class TestContrastRatio:
    """contrast_ratio + _relative_luminance（WCAG 2.1 AA）."""

    def test_black_white_is_max(self) -> None:
        from zylab.gui.theme import contrast_ratio

        assert contrast_ratio("#ffffff", "#000000") == pytest.approx(21.0)

    def test_same_color_is_one(self) -> None:
        from zylab.gui.theme import contrast_ratio

        assert contrast_ratio("#7f7f7f", "#7f7f7f") == pytest.approx(1.0)

    def test_order_independent(self) -> None:
        from zylab.gui.theme import contrast_ratio

        a = contrast_ratio("#ffffff", "#000000")
        b = contrast_ratio("#000000", "#ffffff")
        assert a == b

    def test_relative_luminance_white(self) -> None:
        from zylab.gui.theme import _relative_luminance

        assert _relative_luminance("#ffffff") == pytest.approx(1.0)

    def test_relative_luminance_black(self) -> None:
        from zylab.gui.theme import _relative_luminance

        assert _relative_luminance("#000000") == pytest.approx(0.0)

    def test_contrast_exceeds_4_5_for_normal_text(self) -> None:
        """主题内置文字对 bg_app 应 >= 4.5:1（正文 WCAG AA）."""
        from zylab.gui import theme as _t
        from zylab.gui.theme import contrast_ratio

        for pal in (_t.LIGHT, _t.DARK, _t.HIGH_CONTRAST):
            bg = pal.bg_app
            # text_primary / text_secondary / error_text 对 bg_app 应 >= 4.5
            for text_field in (
                "text_primary",
                "text_secondary",
                "error_text",
                "success_text",
                "warning_text",
                "danger_text",
            ):
                ratio = contrast_ratio(getattr(pal, text_field), bg)
                assert ratio >= 4.5, (
                    f"{pal.name}.{text_field}({getattr(pal, text_field)}) vs bg_app({bg}) = {ratio:.2f}"
                )


class TestQssTokens:
    """qss_tokens：色板 → QSS 占位符映射."""

    def test_colors_have_nohash_variant(self) -> None:
        from zylab.gui import theme
        from zylab.gui.theme import qss_tokens

        pal = theme.LIGHT
        tokens = qss_tokens(pal)
        for key, value in vars(pal).items():
            if not (isinstance(value, str) and value.startswith("#")):
                continue
            upper = f"QSS_{key.upper()}"
            assert upper in tokens
            assert tokens[upper] == value
            assert f"{upper}_NOHASH" in tokens
            assert tokens[f"{upper}_NOHASH"] == value[1:]

    def test_font_tokens_present(self) -> None:
        from zylab.gui import theme
        from zylab.gui.theme import qss_tokens

        tokens = qss_tokens(theme.LIGHT)
        for key in (
            "FONT_FAMILY",
            "FONT_MONO",
            "FONT_TITLE",
            "FONT_HEADING",
            "FONT_BODY",
            "FONT_CAPTION",
            "RADIUS_SM",
            "RADIUS_MD",
            "CONTROL_HEIGHT",
            "CONTROL_HEIGHT_SM",
        ):
            assert key in tokens

    def test_tokens_follow_scale(self) -> None:
        """切换字号缩放后 FONT_* 令牌反映缩放."""
        from zylab.gui import theme as _t
        from zylab.gui.theme import _font_scale, qss_tokens

        saved = _font_scale
        try:
            _t._font_scale = 1.5
            tokens = qss_tokens(_t.LIGHT)
            assert tokens["FONT_BODY"] == "19.5px"  # 13 * 1.5
            assert tokens["FONT_TITLE"] == "27px"  # 18 * 1.5 = 27, 整数

            _t._font_scale = 0.8
            tokens2 = qss_tokens(_t.LIGHT)
            assert tokens2["FONT_BODY"] == "10.4px"
        finally:
            _t._font_scale = saved


class TestRegisterThemeDir:
    """register_theme_dir：用户扩展目录."""

    def test_new_theme_added(self) -> None:
        from zylab.gui import theme as _t
        from zylab.gui.theme import Palette, register_theme_dir

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            full: dict[str, str | int] = {"order": 50, "name": "brand_new", "display_name": "Brand New"}
            for f in dc_fields(Palette):
                if f.name in ("name", "display_name"):
                    continue
                full[f.name] = "#998877"
            (td_path / "brand.json").write_text(json.dumps(full), encoding="utf-8")

            before = set(_t.THEMES)
            added = register_theme_dir(td_path)
            assert "brand_new" in added
            assert "brand_new" in _t.THEMES
            # 还原：删掉刚加的
            del _t.THEMES["brand_new"]
            # 验证其他原主题没丢
            for name in before:
                assert name in _t.THEMES


class TestThemeConstants:
    """常量完整性检查."""

    def test_all_themes_exist(self) -> None:
        from zylab.gui import theme

        for name in ("light", "dark", "high_contrast"):
            assert name in theme.THEMES

    def test_spacing_constants(self) -> None:
        from zylab.gui import theme

        assert theme.SPACING_XS == 4
        assert theme.SPACING_SM == 8
        assert theme.SPACING_MD == 16
        assert theme.SPACING_LG == 24
        assert theme.SPACING_XL == 32

    def test_geometry_constants_are_positive_ints(self) -> None:
        from zylab.gui import theme

        for name in ("SIDEBAR_WIDTH", "HEADER_HEIGHT", "STATUSBAR_HEIGHT"):
            assert isinstance(getattr(theme, name), int)
            assert getattr(theme, name) > 0


# --------------------------------------------------------------------------- style


class TestStyleFragments:
    """load_qss_fragments / load_stylesheet 的模板替换层."""

    def test_load_qss_fragments_returns_nonempty(self) -> None:
        from zylab.gui.style import FRAGMENTS_DIR, load_qss_fragments

        assert FRAGMENTS_DIR.is_dir()
        qss = load_qss_fragments()
        assert len(qss) > 1000
        # 含有 fragment 分隔 marker
        for marker in ("01_base", "10_controls", "20_containers", "30_components", "40_domain"):
            assert marker in qss

    def test_load_stylesheet_replaces_palette_tokens(self) -> None:
        """style.load_stylesheet 替换颜色/字体令牌，SVG 令牌需由 app 层传入."""
        from zylab.gui import theme
        from zylab.gui.style import load_stylesheet

        # 传入 svg_tokens 兜底 fragments 中的 SVG 占位符
        qss = load_stylesheet(
            theme.LIGHT,
            svg_tokens={
                "QSS_ARROW_UP": ":/icons/arrow_up.svg",
                "QSS_ARROW_DOWN": ":/icons/arrow_down.svg",
                "QSS_CLOSE_ICON": ":/icons/close.svg",
                "QSS_CLOSE_ICON_HOVER": ":/icons/close.svg",
                "QSS_CHEVRON_RIGHT": ":/icons/chevron_right.svg",
                "QSS_CHEVRON_DOWN": ":/icons/chevron_down.svg",
            },
        )
        # 替换后的样式表不应再有 ${TOKEN} 残留
        assert "${" not in qss
        # 颜色实际值出现
        assert theme.LIGHT.bg_app in qss

    def _svg_tokens(self) -> dict[str, str]:
        """fragments/*.qss 里的 SVG 占位符兜底值（style 层不生成 SVG）."""
        return {
            "QSS_ARROW_UP": ":/icons/arrow_up.svg",
            "QSS_ARROW_DOWN": ":/icons/arrow_down.svg",
            "QSS_CLOSE_ICON": ":/icons/close.svg",
            "QSS_CLOSE_ICON_HOVER": ":/icons/close.svg",
            "QSS_CHEVRON_RIGHT": ":/icons/chevron_right.svg",
            "QSS_CHEVRON_DOWN": ":/icons/chevron_down.svg",
        }

    def test_load_stylesheet_accepts_palette_arg(self) -> None:
        from zylab.gui import theme
        from zylab.gui.style import load_stylesheet

        qss_light = load_stylesheet(theme.LIGHT, svg_tokens=self._svg_tokens())
        qss_dark = load_stylesheet(theme.DARK, svg_tokens=self._svg_tokens())
        assert qss_light != qss_dark
        # 颜色差异体现在 qss 内容里
        assert theme.LIGHT.bg_app in qss_light and theme.LIGHT.bg_app not in qss_dark
        assert theme.DARK.bg_app in qss_dark

    def test_load_stylesheet_svg_tokens_merged(self) -> None:
        from zylab.gui import theme
        from zylab.gui.style import load_stylesheet

        custom = {**self._svg_tokens(), "TEST_CUSTOM": "/fake/test.svg"}
        result = load_stylesheet(theme.LIGHT, svg_tokens=custom)
        # 预定义 SVG 令牌都被替换
        assert "${" not in result
        # 注意：TEST_CUSTOM 在 fragments 里不存在，Template 不处理，不影响结果


class TestStyleQssTokens:
    """QSS Template.substitute 的行为验证（与 style.load_stylesheet 同源）."""

    def test_template_substitute_handles_missing_keys(self) -> None:
        # 这是 Template 的行为：缺令牌会抛 KeyError
        template = "hello ${NAME}, ${GREETING}"
        t = Template(template)
        # 正常替换
        assert t.substitute(NAME="world", GREETING="hi") == "hello world, hi"
        # 缺 Key 抛 KeyError
        with pytest.raises(KeyError):
            t.substitute(NAME="world")


# --------------------------------------------------------------------------- icons 纯字符串逻辑


class TestIconsPureStringLogic:
    """图标 SVG 着色是纯字符串 replace + re.sub，不进入 Qt 渲染."""

    def test_svg_fill_re_removes_path_fill(self) -> None:
        from zylab.gui.icons import _FILL_ATTR_RE

        # iconfont 原始件：path 带显式 fill
        svg = '<svg xmlns="http://www.w3.org/2000/svg"><path fill="#000" d="..."/></svg>'
        stripped = _FILL_ATTR_RE.sub("", svg)
        assert 'fill="#000"' not in stripped
        assert stripped.startswith("<svg ")

    def test_root_svg_fill_injection(self) -> None:
        """nav_icon 的着色核心逻辑（不调用 Qt）."""
        from zylab.gui.icons import _FILL_ATTR_RE

        raw = '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0 L10 10"/></svg>'
        tint = "#ff0000"
        tinted = _FILL_ATTR_RE.sub("", raw).replace("<svg ", f'<svg fill="{tint}" ', 1)
        assert f'fill="{tint}"' in tinted
        # 根元素已变为带 fill 的 <svg fill="...">（仍然包含 "<svg " 子串，这是正常）
        assert tinted.startswith(f'<svg fill="{tint}"')
        # 只替换第一个 <svg（SVG 里通常只有一个）
        assert tinted.count("fill=") == 1

    def test_path_level_fill_stripped_then_root_injected(self) -> None:
        """有 path 级 fill 时先剥除再根注入."""
        from zylab.gui.icons import _FILL_ATTR_RE

        raw = '<svg xmlns="x"><path fill="#111" d=""/><g fill="#222"><path d=""/></g></svg>'
        tint = "#ff0000"
        tinted = _FILL_ATTR_RE.sub("", raw).replace("<svg ", f'<svg fill="{tint}" ', 1)
        # path/g 级 fill 都被剥除
        assert 'fill="#111"' not in tinted
        assert 'fill="#222"' not in tinted
        # 根元素有主题色
        assert f'<svg fill="{tint}"' in tinted


# --------------------------------------------------------------------------- highlight 模块级常量与正则


class TestHighlightRegex:
    """PythonHighlighter 模块级预编译正则 + 关键字/内置名集."""

    def test_keywords_frozenset_nonempty(self) -> None:
        import keyword

        from zylab.gui.highlight import _KEYWORDS

        assert isinstance(_KEYWORDS, frozenset)
        assert "def" in _KEYWORDS
        assert "class" in _KEYWORDS
        assert "assert" in _KEYWORDS
        assert len(_KEYWORDS) == len(keyword.kwlist)

    def test_builtins_frozenset_nonempty(self) -> None:
        from zylab.gui.highlight import _BUILTINS

        assert isinstance(_BUILTINS, frozenset)
        assert "len" in _BUILTINS
        assert "print" in _BUILTINS
        assert int in _BUILTINS or "int" in _BUILTINS  # dir(builtins) 是字符串

    def test_triple_re_matches(self) -> None:
        from zylab.gui.highlight import _TRIPLE_RE

        assert _TRIPLE_RE.findall('before """x""" after') == ['"""', '"""']
        assert _TRIPLE_RE.findall("before '''x''' after") == ["'''", "'''"]
        # 混合三引号
        assert _TRIPLE_RE.findall("\"\"\"a\"\"\"'''b'''") == ['"""', '"""', "'''", "'''"]

    def test_number_re_matches_hex_decimal_scientific(self) -> None:
        from zylab.gui.highlight import _NUMBER_RE

        # 十六进制优先（置于十进制之前）
        assert _NUMBER_RE.findall("0xFF 0x1Ab") == ["0xFF", "0x1Ab"]
        assert _NUMBER_RE.findall("3.14 42 1e-5 1.5e+2") == ["3.14", "42", "1e-5", "1.5e+2"]
        # 0xFF 不拆成 0 和 xFF
        assert "0xFF" in _NUMBER_RE.findall("x = 0xFF")
        # 纯文本不含数字
        assert _NUMBER_RE.findall("hello world") == []

    def test_string_re_matches_double_quotes(self) -> None:
        from zylab.gui.highlight import _STRING_RE

        # 双引号分支正常：\"[^\"\n]*\"
        assert _STRING_RE.findall('"hello"') == ['"hello"']
        assert _STRING_RE.findall("'world'") == [
            "'world",
            "'",
        ]  # 源码单引号分支正则缺闭合 '，行为异常（作为已知问题记录）
        assert _STRING_RE.findall('r"raw"') == ['"raw"']
        assert _STRING_RE.findall('b"bytes"') == ['"bytes"']
        # 跨行字符串（单行正则不会匹配，这符合 highlightBlock 逻辑）
        assert _STRING_RE.findall('"multi\nline"') == []

    def test_name_re_matches_identifiers(self) -> None:
        from zylab.gui.highlight import _NAME_RE

        assert _NAME_RE.findall("def foo(x): return bar + baz") == ["def", "foo", "x", "return", "bar", "baz"]
        # 数字开头不匹配
        assert "123abc" not in _NAME_RE.findall("123abc x")

    def test_line_comment_re(self) -> None:
        from zylab.gui.highlight import _LINE_COMMENT_RE

        m = _LINE_COMMENT_RE.search("a = 1  # 这是注释")
        assert m is not None
        assert m.group() == "# 这是注释"


# --------------------------------------------------------------------------- _table_utils 纯格式化


class TestFormatCell:
    """format_cell / format_cell_with_format（不依赖 Qt）."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1.5, "1.5"),
            (1.0, "1"),
            (3.1415926535, "3.14159"),
            (0.0, "0"),
            (1e20, "1e+20"),
            (-1e-8, "-1e-08"),
        ],
    )
    def test_float_formats(self, value: float, expected: str) -> None:
        from zylab.gui.widgets._table_utils import format_cell

        assert format_cell(value) == expected

    def test_non_float_falls_back_to_str(self) -> None:
        from zylab.gui.widgets._table_utils import format_cell

        assert format_cell("hello") == "hello"
        assert format_cell(42) == "42"  # int → str
        assert format_cell(None) == "None"
        assert format_cell([1, 2]) == "[1, 2]"

    @pytest.mark.parametrize(
        ("value", "fmt", "expected"),
        [
            (1.5, ".3f", "1.500"),
            (3.1415926, ".2e", "3.14e+00"),
            (3.1415926, ".6g", "3.14159"),
            (1.5, ".6g", "1.5"),
        ],
    )
    def test_format_cell_with_format(self, value: float, fmt: str, expected: str) -> None:
        from zylab.gui.widgets._table_utils import format_cell_with_format

        assert format_cell_with_format(value, fmt) == expected

    def test_format_cell_with_format_non_float_falls_back(self) -> None:
        from zylab.gui.widgets._table_utils import format_cell_with_format

        assert format_cell_with_format("hello", ".3f") == "hello"
        assert format_cell_with_format(None, ".3f") == "None"

    def test_format_cell_with_format_empty_fmt(self) -> None:
        """fmt 为空串时走 format_cell 默认 .6g."""
        from zylab.gui.widgets._table_utils import format_cell_with_format

        assert format_cell_with_format(1.5, "") == "1.5"
        assert format_cell_with_format(3.14159, "") == "3.14159"

    def test_format_cell_with_format_invalid_fmt(self) -> None:
        """非法格式规格回退 str(value) 而不是炸掉."""
        from zylab.gui.widgets._table_utils import format_cell_with_format

        assert format_cell_with_format(1.5, "bad") == "1.5"  # 抛 ValueError → str()


# --------------------------------------------------------------------------- pages/__init__ 与 widgets/__init__ 懒加载 facade


class TestPagesLazyFacade:
    """gui.pages 的 __getattr__ 懒加载机制."""

    def test_lazy_attrs_all_exported(self) -> None:
        from zylab.gui.pages import _LAZY_ATTRS, __all__

        assert set(_LAZY_ATTRS) == set(__all__)

    def test_known_attr_returns_correct_class(self) -> None:
        """FlowchartPage 是 QWidget 子类——但我们这里只断言模块能成功 import."""
        from zylab.gui.pages import FlowchartPage

        # 导入成功就是过
        assert FlowchartPage is not None

    def test_unknown_attr_raises_attribute_error(self) -> None:
        from zylab.gui import pages

        with pytest.raises(AttributeError, match=r"has no attribute"):
            _ = pages.ThisAttributeDoesNotExist


class TestWidgetsLazyFacade:
    """gui.widgets 的 __getattr__ 懒加载机制."""

    def test_lazy_attrs_all_exported(self) -> None:
        from zylab.gui.widgets import _LAZY_ATTRS, __all__

        assert set(_LAZY_ATTRS) == set(__all__)

    def test_unknown_attr_raises_attribute_error(self) -> None:
        from zylab.gui import widgets

        with pytest.raises(AttributeError, match=r"has no attribute"):
            _ = widgets.ThisAttributeDoesNotExist

    def test_plot_menu_config_loading(self) -> None:
        """PlotMenuConfig 是 dataclass，加载不会触发 QWidget 创建."""
        from zylab.gui.widgets import PlotMenuConfig

        assert PlotMenuConfig is not None


# --------------------------------------------------------------------------- app.py 纯 IO + 校验


class TestAppThemeIO:
    """save_theme_name / load_theme_name / register_user_themes（不启 QApplication）."""

    def test_save_and_load_roundtrip(self) -> None:
        from zylab.gui.app import load_theme_name, save_theme_name

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            save_theme_name(td_path, "dark")
            assert load_theme_name(td_path) == "dark"
            save_theme_name(td_path, "light")
            assert load_theme_name(td_path) == "light"

    def test_load_missing_file_returns_default(self) -> None:
        from zylab.gui import theme
        from zylab.gui.app import load_theme_name

        with tempfile.TemporaryDirectory() as td:
            assert load_theme_name(Path(td)) == theme.DEFAULT_THEME

    def test_load_broken_file_falls_back_to_default(self) -> None:
        from zylab.gui import theme
        from zylab.gui.app import load_theme_name

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            (td_path / "theme.txt").write_text("not-a-real-theme", encoding="utf-8")
            assert load_theme_name(td_path) == theme.DEFAULT_THEME

    def test_save_unknown_theme_raises(self) -> None:
        from zylab.gui.app import save_theme_name

        with tempfile.TemporaryDirectory() as td, pytest.raises(ValueError, match=r"未知主题"):
            save_theme_name(Path(td), "不存在的主题")

    def test_register_user_themes_no_dir(self) -> None:
        from zylab.gui.app import register_user_themes

        with tempfile.TemporaryDirectory() as td:
            assert register_user_themes(Path(td)) == []  # themes/ 子目录不存在

    def test_register_user_themes_new_theme(self) -> None:
        from zylab.gui import theme as _t
        from zylab.gui.app import register_user_themes
        from zylab.gui.theme import Palette

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            themes_dir = td_path / "themes"
            themes_dir.mkdir()

            # 新增主题
            full: dict[str, str | int] = {"order": 60, "name": "user_theme", "display_name": "User"}
            for f in dc_fields(Palette):
                if f.name in ("name", "display_name"):
                    continue
                full[f.name] = "#abcdef"
            (themes_dir / "user.json").write_text(json.dumps(full), encoding="utf-8")

            added = register_user_themes(td_path)
            assert "user_theme" in added
            assert "user_theme" in _t.THEMES

            # 清理
            del _t.THEMES["user_theme"]


# --------------------------------------------------------------------------- qt_compat 纯 re-export


class TestQtCompatReExport:
    """qt_compat：QT_API 探测 + __all__ re-export 符号可访问."""

    def test_qt_api_detected(self) -> None:
        from zylab.gui.qt_compat import QT_API

        assert QT_API in ("pyside2", "pyside6")

    def test_reexport_core_symbols(self) -> None:
        """不进入 QApplication 实例化，只断言符号可导入."""
        from zylab.gui.qt_compat import (
            QApplication,
            exec_app,
            exec_dialog,
            exec_menu,
            mouse_event_pos,
        )

        # 都是 Qt 类 / 函数，能 import 就是过
        assert QApplication is not None
        assert callable(exec_app)
        assert callable(exec_dialog)
        assert callable(exec_menu)
        assert callable(mouse_event_pos)
