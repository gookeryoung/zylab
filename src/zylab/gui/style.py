"""样式系统：QPalette 层 + QSS Fragment 层.

三层架构：

1. **QPalette 层**（:func:`build_qpalette`）
   将 theme.Palette 的语义色映射到 Qt QPalette ColorRole，让 Fusion
   风格的原生系统绘制（下拉箭头、滚动条滑块、checkbox 勾选框、
   QToolTip 背景等 QSS 无法完全覆盖的部分）也跟随主题。

2. **QSS Fragment 层**（:func:`load_qss_fragments` / :func:`load_stylesheet`）
   将全局样式表拆分为语义化的 ``fragments/*.qss`` 片段（基础 →
   控件 → 容器 → 组件 → 业务），由 :func:`load_stylesheet` 按序
   聚合并替换 `${TOKEN}` 占位符，消除业务代码中的内联 setStyleSheet。

3. **QProxyStyle 层**（:mod:`zylab.gui.proxy_style`）
   继承 QProxyStyle 重写 polish/unpolish，在控件被 Qt 样式系统
    polish 时同步 QPalette，并拦截 pixelMetric / drawPrimitive
   处理 QSS 无对应属性的绘制细节。
"""

from __future__ import annotations

import logging
from pathlib import Path
from string import Template

from . import theme
from .qt_compat import QColor, QPalette

__all__ = [
    "FRAGMENTS_DIR",
    "build_qpalette",
    "load_qss_fragments",
    "load_stylesheet",
]

logger = logging.getLogger(__name__)

#: QSS fragment 目录（相对于本文件的 fragments/ 子目录）
FRAGMENTS_DIR = Path(__file__).resolve().parent / "fragments"


# ---------------------------------------------------------------------------
# Layer 1: QPalette 构建
# ---------------------------------------------------------------------------


def build_qpalette(pal: theme.Palette) -> QPalette:
    """将 :class:`theme.Palette` 语义色映射到 Qt QPalette ColorRole.

    Qt QPalette ColorRole 与 Palette 字段映射：

    +-----------------------------+------------------------+
    | QPalette.ColorRole          | Palette 字段           |
    +=============================+========================+
    | Window                      | bg_app                 |
    | Base                        | bg_input               |
    | ToolTipBase                 | bg_muted               |
    | ToolTipText                 | text_primary           |
    | WindowText / Text           | text_primary           |
    | ButtonText                  | primary_text           |
    | PlaceholderText             | text_disabled          |
    | Disabled (Window/Base/Text) | bg_muted / bg_muted /  |
    |                             | text_disabled          |
    | Highlight                   | selection_bg           |
    | HighlightedText             | selection_text         |
    | Link / LinkVisited          | primary                |
    +-----------------------------+------------------------+

    Args:
        pal: 当前主题的语义色板。

    Returns:
        构建好的 QPalette 对象。
    """
    qpal = QPalette()

    # ---- 背景类 ----
    qpal.setColor(QPalette.Window, QColor(pal.bg_app))
    qpal.setColor(QPalette.Base, QColor(pal.bg_input))
    qpal.setColor(QPalette.AlternateBase, QColor(pal.bg_muted))
    qpal.setColor(QPalette.ToolTipBase, QColor(pal.bg_muted))

    # ---- 文字类 ----
    qpal.setColor(QPalette.WindowText, QColor(pal.text_primary))
    qpal.setColor(QPalette.Text, QColor(pal.text_primary))
    qpal.setColor(QPalette.ButtonText, QColor(pal.primary_text))
    qpal.setColor(QPalette.ToolTipText, QColor(pal.text_primary))
    qpal.setColor(QPalette.PlaceholderText, QColor(pal.text_disabled))

    # ---- 按钮 base（Fusion pushbutton 的非 QSS 系统绘制部分用 Button color） ----
    qpal.setColor(QPalette.Button, QColor(pal.primary))

    # ---- 禁用态 ----
    disabled_color = QColor(pal.text_disabled)
    disabled_bg = QColor(pal.bg_muted)
    qpal.setColor(QPalette.Disabled, QPalette.WindowText, disabled_color)
    qpal.setColor(QPalette.Disabled, QPalette.Text, disabled_color)
    qpal.setColor(QPalette.Disabled, QPalette.ButtonText, disabled_color)
    qpal.setColor(QPalette.Disabled, QPalette.Base, disabled_bg)
    qpal.setColor(QPalette.Disabled, QPalette.Window, disabled_bg)

    # ---- 选中态 ----
    qpal.setColor(QPalette.Highlight, QColor(pal.selection_bg))
    qpal.setColor(QPalette.HighlightedText, QColor(pal.selection_text))

    # ---- 超链接 ----
    qpal.setColor(QPalette.Link, QColor(pal.primary))
    qpal.setColor(QPalette.LinkVisited, QColor(pal.primary_pressed))

    return qpal


# ---------------------------------------------------------------------------
# Layer 2: QSS Fragment 聚合
# ---------------------------------------------------------------------------


def load_qss_fragments() -> str:
    """扫描 ``fragments/*.qss`` 按文件名自然序拼接为完整样式表.

    文件名约定前缀决定顺序：
    - ``01_base.qss``      全局基础、字体、滚动条、选区
    - ``10_controls.qss``  按钮、输入控件、菜单、勾选框、进度条
    - ``20_containers.qss`` 分组框、TabWidget/QTabBar、QSplitter、状态栏
    - ``30_components.qss`` 组件级：header、sidebar、tree、progressbar
    - ``40_domain.qss``    业务级：notebook、result、docs、template、about

    这样拆分的好处：每类独立维护，覆盖顺序明确（后面的 fragment 可以
    覆盖前面的同名选择器规则），新增组件样式只需追加一个文件。
    """
    if not FRAGMENTS_DIR.is_dir():
        # 兜底：fragments/ 不存在时回退单文件 style.qss（兼容旧安装）
        fallback = Path(__file__).parent / "style.qss"
        if fallback.is_file():
            return fallback.read_text(encoding="utf-8")
        logger.warning("QSS fragment 目录与单文件 style.qss 均不存在")
        return ""

    fragments: list[str] = []
    for path in sorted(FRAGMENTS_DIR.glob("*.qss")):
        text = path.read_text(encoding="utf-8")
        fragments.append(f"/* ==== {path.name} ==== */\n{text}")

    return "\n\n".join(fragments)


def load_stylesheet(
    palette: theme.Palette | None = None,
    *,
    svg_tokens: dict[str, str] | None = None,
) -> str:
    """加载完整样式表：聚合 QSS fragment + 替换主题令牌 + 注入 SVG 资源.

    Args:
        palette: 主题色板；None 取当前 theme.current_palette()。
        svg_tokens: 箭头/close/chevron SVG 路径令牌；None 表示已由调用方预生成并
                    合并。app.py 负责生成 SVG 后传入，style.py 不做文件 I/O。

    Returns:
        令牌替换后的完整 QSS 字符串，可直接 ``app.setStyleSheet()``。
    """
    pal = palette if palette is not None else theme.current_palette()
    tokens = {**theme.qss_tokens(pal)}
    if svg_tokens is not None:
        tokens.update(svg_tokens)

    qss_text = load_qss_fragments()
    return Template(qss_text).substitute(tokens)
