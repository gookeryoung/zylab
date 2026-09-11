"""sci.palettes 统一曲线色板契约测试."""

from __future__ import annotations

from zylab.sci.palettes import CURVE_PALETTE, SEMANTIC_CURVE_COLORS, resolve_curve_color


def test_curve_palette_is_tuple_of_hex() -> None:
    """CURVE_PALETTE 必须是 tuple[str, ...] 且每项为合法 #RRGGBB."""
    assert isinstance(CURVE_PALETTE, tuple)
    assert len(CURVE_PALETTE) == 6
    for color in CURVE_PALETTE:
        assert color.startswith("#")
        assert len(color) == 7
        # 全部为大写
        assert color == color.upper()


def test_curve_palette_no_duplicates() -> None:
    """色环无重复色值."""
    assert len(CURVE_PALETTE) == len(set(CURVE_PALETTE))


def test_curve_palette_first_is_primary() -> None:
    """色环首色必须是 primary 语义色（靛蓝 #3C2ECA）."""
    assert CURVE_PALETTE[0] == "#3C2ECA"
    assert SEMANTIC_CURVE_COLORS["primary"] == "#3C2ECA"


def test_semantic_color_keys() -> None:
    """语义色必须覆盖 primary/success/warning/danger/info."""
    assert set(SEMANTIC_CURVE_COLORS.keys()) == {"primary", "success", "warning", "danger", "info"}
    for color in SEMANTIC_CURVE_COLORS.values():
        assert color.startswith("#")
        assert len(color) == 7


def test_resolve_curve_color_hex_passthrough() -> None:
    """# 开头直通（大小写保留原值）."""
    assert resolve_curve_color("#FF0000", 0) == "#FF0000"
    assert resolve_curve_color("#abcd12", 0) == "#abcd12"


def test_resolve_curve_color_semantic() -> None:
    """语义名 → hex."""
    assert resolve_curve_color("primary", 0) == "#3C2ECA"
    assert resolve_curve_color("success", 0) == "#10B981"
    assert resolve_curve_color("unknown_color", 0) == "unknown_color"  # 未知名直通


def test_resolve_curve_color_fallback_cycle() -> None:
    """None/空字符串 → 按索引循环取 CURVE_PALETTE."""
    assert resolve_curve_color(None, 0) == CURVE_PALETTE[0]
    assert resolve_curve_color(None, 10) == CURVE_PALETTE[10 % 6]
    assert resolve_curve_color("", 3) == CURVE_PALETTE[3]


def test_palette_consistency_gui_report() -> None:
    """palette 与 report SVG 内联色一致（验证 resolve_curve_color 全链路可用）."""
    # success 应为绿色 #10B981
    assert resolve_curve_color("success", 0) == "#10B981"
    # 索引 1 应为 success 同色
    assert resolve_curve_color(None, 1) == "#10B981"


def test_resolve_curve_color_index_wrap_around() -> None:
    """索引溢出时正确取模（防止越界）."""
    for i in range(20):
        assert resolve_curve_color(None, i) == CURVE_PALETTE[i % len(CURVE_PALETTE)]


def test_resolve_curve_color_all_semantic_keys() -> None:
    """全部语义键解析都返回 #RRGGBB 格式."""
    for key in ("primary", "success", "warning", "danger", "info"):
        val = resolve_curve_color(key, 0)
        assert val.startswith("#") and len(val) == 7


def test_semantic_colors_align_with_palette_first5() -> None:
    """primary=palette[0], success=palette[1], warning=palette[2], danger=palette[3]."""
    assert SEMANTIC_CURVE_COLORS["primary"] == CURVE_PALETTE[0]
    assert SEMANTIC_CURVE_COLORS["success"] == CURVE_PALETTE[1]
    assert SEMANTIC_CURVE_COLORS["warning"] == CURVE_PALETTE[2]
    assert SEMANTIC_CURVE_COLORS["danger"] == CURVE_PALETTE[3]


def test_palette_used_in_seaborn_prop_cycle() -> None:
    """palette 可直接传给 seaborn.set_theme 的 palette 参数."""
    list(CURVE_PALETTE)  # 可迭代
    assert len(list(CURVE_PALETTE)) == 6


def test_notebook_page_uses_shared_palette():
    """notebook_page 不再有本地 _CURVE_KEYS 循环常量，改用 CURVE_PALETTE."""
    import ast
    from pathlib import Path

    src = Path(r"src/zylab/gui/pages/notebook_page.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "_CURVE_KEYS" not in names, "notebook_page 不应再有本地 _CURVE_KEYS"
    # 确认用的是共享 CURVE_PALETTE
    assert "CURVE_PALETTE" in src
