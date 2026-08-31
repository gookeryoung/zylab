"""主题系统：语义化设计令牌 + 多主题（浅色/深色/高对比）.

设计原则（WCAG 2.1 AA）：
- 正文文字对背景对比度 >= 4.5:1，大号标题/图形 >= 3:1；
- 令牌按「语义」命名（背景层级/文字层级/主色/状态色），QSS 与代码
  只引用语义名，换主题不改样式表；
- 颜色仅出现在本模块（禁止散落硬编码）；字体/间距/尺寸等非色令牌
  为全局常量，与主题无关。

配色架构（每主题一套 :class:`Palette`）：
- 背景三级：``bg_app``（窗口底）→ ``bg_muted``（状态栏/分组底）→
  ``bg_input``（输入控件底，通常与 bg_surface 同级）；
- 文字三级：``text_primary``（正文）→ ``text_secondary``（说明/辅助）→
  ``text_on_primary``（主色底上的文字）；
- 导航区独立配色：``nav_*``（侧边栏/头部条），选中态带 ``nav_accent`` 竖条；
- 状态色区分「文字用途」（在 bg_app 上 >= 4.5:1）。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields
from pathlib import Path

__all__ = [
    "CONTROL_HEIGHT",
    "CONTROL_HEIGHT_SM",
    "DARK",
    "DEFAULT_THEME",
    "FONT_BODY",
    "FONT_CAPTION",
    "FONT_FAMILY",
    "FONT_HEADING",
    "FONT_MONO",
    "FONT_TITLE",
    "HEADER_HEIGHT",
    "HIGH_CONTRAST",
    "LIGHT",
    "RADIUS_MD",
    "RADIUS_SM",
    "SIDEBAR_WIDTH",
    "SPACING_LG",
    "SPACING_MD",
    "SPACING_SM",
    "SPACING_XL",
    "SPACING_XS",
    "STATUSBAR_HEIGHT",
    "THEMES",
    "Palette",
    "contrast_ratio",
    "current_palette",
    "load_themes_from_dir",
    "palette",
    "qss_tokens",
    "register_theme_dir",
    "set_current_theme",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 语义色板
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Palette:
    """一套语义色令牌（全部为 ``#RRGGBB`` 字符串）.

    Attributes:
        name: 主题标识（持久化与切换用）。
        display_name: 界面显示名。
        bg_app: 窗口/内容区底色。
        bg_muted: 状态栏、分组、卡片间隙底色。
        bg_input: 输入控件（编辑框/下拉/微调框）底色。
        text_primary: 主文字（对 bg_app >= 4.5:1）。
        text_secondary: 次要文字/说明（对 bg_app 与 bg_muted >= 4.5:1）。
        text_on_primary: 主色底上的文字。
        text_disabled: 禁用态文字。
        nav_bg: 侧边栏/头部条底色。
        nav_bg_hover: 导航项悬停底色。
        nav_bg_selected: 导航项选中底色。
        nav_text: 导航文字（对 nav_bg >= 4.5:1）。
        nav_accent: 导航选中竖条/强调装饰色。
        primary: 主色（主按钮、选中、聚焦、进度条）。
        primary_hover: 主按钮悬停。
        primary_pressed: 主按钮按下。
        primary_text: 主按钮文字（对 primary >= 4.5:1）。
        selection_bg: 文本选区/表格选中底色。
        selection_text: 选区文字。
        border: 常规边框/分割线/滚动条。
        border_strong: 强边框（输入聚焦、表格外框）。
        scrollbar: 滚动条滑块。
        scrollbar_hover: 滚动条滑块悬停。
        success_text: 成功状态文字。
        warning_text: 警告状态文字。
        danger_text: 危险状态文字（按钮/错误共用色源）。
        error_text: REPL 错误/stderr 文字。
    """

    name: str
    display_name: str
    bg_app: str
    bg_muted: str
    bg_input: str
    text_primary: str
    text_secondary: str
    text_on_primary: str
    text_disabled: str
    nav_bg: str
    nav_bg_hover: str
    nav_bg_selected: str
    nav_text: str
    nav_accent: str
    primary: str
    primary_hover: str
    primary_pressed: str
    primary_text: str
    selection_bg: str
    selection_text: str
    border: str
    border_strong: str
    scrollbar: str
    scrollbar_hover: str
    success_text: str
    warning_text: str
    danger_text: str
    error_text: str


# ---------------------------------------------------------------------------
# 主题加载（JSON 唯一来源，支持用户目录扩展）
# ---------------------------------------------------------------------------

#: 内置主题资源目录（随包分发的 assets/themes/*.json）
_THEMES_DIR = Path(__file__).resolve().parent.parent / "assets" / "themes"

_PALETTE_FIELDS = frozenset(f.name for f in fields(Palette))


def _palette_from_json(data: dict, base: Palette | None = None) -> Palette:
    """JSON 字典构建色板：``name``/``display_name`` 必填，色值缺省继承 base.

    Args:
        data: 主题 JSON 解析结果（``order`` 字段仅排序用，非色板字段）。
        base: 同名既有主题（部分字段覆盖时继承其色值；完整定义传 None）。

    Returns:
        构建的色板。

    Raises:
        ValueError: 存在未知字段、色值非法或缺必填字段时抛出。
    """
    unknown = sorted(set(data) - _PALETTE_FIELDS - {"order"})
    if unknown:
        raise ValueError(f"未知色板字段: {unknown}")
    merged: dict[str, str] = dict(vars(base)) if base is not None else {}
    for key in sorted(_PALETTE_FIELDS):
        value = data.get(key)
        if value is None:
            continue
        if key in ("name", "display_name"):
            merged[key] = str(value)
            continue
        color = str(value)
        if len(color) != 7 or color[0] != "#":
            raise ValueError(f"非法色值 {key}={color!r}")
        merged[key] = color
    if not merged.get("name") or not merged.get("display_name"):
        raise ValueError("缺少必填字段 name/display_name")
    if len(merged) != len(_PALETTE_FIELDS):
        raise ValueError(f"缺少色板字段: {sorted(_PALETTE_FIELDS - set(merged))}")
    return Palette(**merged)


def load_themes_from_dir(directory: Path, base: dict[str, Palette] | None = None) -> dict[str, Palette]:
    """扫描目录下 ``*.json`` 主题定义构建主题表.

    Args:
        directory: 主题 JSON 目录。
        base: 基底主题表（同名主题部分字段覆盖时继承其色值）。

    Returns:
        ``{主题名: Palette}``（按 ``order`` 升序、同序按文件名；非法文件
        跳过并告警，不影响其余主题）。
    """
    themes: dict[str, Palette] = dict(base) if base is not None else {}
    entries: list[tuple[int, str, Path, dict]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            order = int(data.get("order", 100))
        except (OSError, ValueError) as exc:  # JSONDecodeError 是 ValueError 子类
            logger.warning("主题文件加载失败，已跳过: %s (%s)", path, exc)
            continue
        entries.append((order, path.name, path, data))
    for _order, _fname, path, data in sorted(entries, key=lambda e: (e[0], e[1])):
        try:
            name = str(data.get("name", ""))
            themes[name] = _palette_from_json(data, themes.get(name))
        except ValueError as exc:
            logger.warning("主题定义非法，已跳过: %s (%s)", path, exc)
    return {name: pal for name, pal in themes.items() if name}


THEMES: dict[str, Palette] = load_themes_from_dir(_THEMES_DIR)

for _required in ("light", "dark", "high_contrast"):
    if _required not in THEMES:
        raise RuntimeError(f"内置主题资源缺失或不完整: {_THEMES_DIR}")

#: 便捷引用（内置主题快照；运行时经 register_theme_dir 覆盖同名主题后请用 palette() 取新值）
LIGHT = THEMES["light"]
DARK = THEMES["dark"]
HIGH_CONTRAST = THEMES["high_contrast"]

DEFAULT_THEME = "light"


def register_theme_dir(directory: Path) -> list[str]:
    """注册用户主题扩展目录（同名覆盖内置、新主题追加），返回新增主题名.

    供未来配置扩展：应用启动时扫描数据目录 ``themes/``，用户可放置
    部分字段的自定义 JSON 微调内置主题，或新增完整主题。

    Args:
        directory: 用户主题 ``*.json`` 目录。

    Returns:
        本次新引入的主题名列表（覆盖内置的不计入）。
    """
    before = set(THEMES)
    merged = load_themes_from_dir(directory, base=THEMES)
    THEMES.clear()
    THEMES.update(merged)
    return sorted(set(THEMES) - before)


# 模块级当前主题（GUI 单线程读写；set_current_theme 切换后由 app 重刷 QSS）
_current: Palette = THEMES[DEFAULT_THEME]


def palette(name: str) -> Palette:
    """按主题标识取色板.

    Args:
        name: 主题标识（light/dark/high_contrast）。

    Returns:
        对应 :class:`Palette`。

    Raises:
        ValueError: 主题名不存在时抛出。
    """
    try:
        return THEMES[name]
    except KeyError:
        raise ValueError(f"未知主题: {name!r}（可选: {sorted(THEMES)}）") from None


def set_current_theme(name: str) -> None:
    """设置当前主题（随后应调用 app 层重刷样式表）.

    Raises:
        ValueError: 主题名不存在时抛出。
    """
    global _current  # noqa: PLW0603  GUI 单线程模块级当前主题，切换入口唯一
    _current = palette(name)


def current_palette() -> Palette:
    """当前生效的色板（代码取色统一入口，如 pyqtgraph 画笔）."""
    return _current


def qss_tokens(pal: Palette) -> dict[str, str]:
    """色板 + 非色令牌合成 QSS 占位符映射（``string.Template.substitute`` 入参）."""
    tokens: dict[str, str] = {f"QSS_{key.upper()}": value for key, value in vars(pal).items()}
    tokens.update(
        {
            "FONT_FAMILY": FONT_FAMILY,
            "FONT_MONO": FONT_MONO,
            "FONT_TITLE": FONT_TITLE,
            "FONT_HEADING": FONT_HEADING,
            "FONT_BODY": FONT_BODY,
            "FONT_CAPTION": FONT_CAPTION,
            "RADIUS_SM": RADIUS_SM,
            "RADIUS_MD": RADIUS_MD,
            "CONTROL_HEIGHT": CONTROL_HEIGHT,
            "CONTROL_HEIGHT_SM": CONTROL_HEIGHT_SM,
        }
    )
    return tokens


# ---------------------------------------------------------------------------
# 对比度工具（WCAG 2.1）
# ---------------------------------------------------------------------------


def contrast_ratio(color_a: str, color_b: str) -> float:
    """计算两色 WCAG 相对对比度（>= 1.0，黑白为 21:1）.

    Args:
        color_a: ``#RRGGBB`` 颜色。
        color_b: ``#RRGGBB`` 颜色。

    Returns:
        对比度比值（正文达标线 4.5，大字/图形 3.0）。
    """
    la, lb = _relative_luminance(color_a), _relative_luminance(color_b)
    lighter, darker = max(la, lb), min(la, lb)
    return (lighter + 0.05) / (darker + 0.05)


def _relative_luminance(color: str) -> float:
    """sRGB 相对亮度（WCAG 公式，线性化后按 0.2126/0.7152/0.0722 加权）."""
    channels = [int(color[i : i + 2], 16) / 255.0 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


# ---------------------------------------------------------------------------
# 非色令牌（与主题无关）
# ---------------------------------------------------------------------------

# 排版
FONT_FAMILY = '"PingFang SC", "Microsoft YaHei", "Segoe UI", "Helvetica Neue", Arial, sans-serif'
FONT_MONO = '"DejaVu Sans Mono", "Consolas", "Cascadia Mono", monospace'
FONT_TITLE = "18px"
FONT_HEADING = "15px"
FONT_BODY = "13px"
FONT_CAPTION = "11px"

# 间距（8px 基准网格）
SPACING_XS = 4
SPACING_SM = 8
SPACING_MD = 16
SPACING_LG = 24
SPACING_XL = 32

# 圆角与尺寸
RADIUS_SM = "4px"
RADIUS_MD = "6px"
CONTROL_HEIGHT = "32px"
CONTROL_HEIGHT_SM = "26px"
SIDEBAR_WIDTH = 200
HEADER_HEIGHT = 40
STATUSBAR_HEIGHT = 28
