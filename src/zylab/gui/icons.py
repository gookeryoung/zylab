"""界面图标：assets/icons 单色 SVG 按主题令牌着色（源码注入法）.

SVG 源为单色剪影（path 无 fill 属性，默认黑）。着色方式为在 <svg> 根元素
注入 ``fill="颜色"``（SVG fill 可继承到 path），再经 loadFromData 渲染 ——
背景天然透明，不依赖 QPainter 合成模式的平台行为。
主题切换后由调用方重新生成。
"""

from __future__ import annotations

from pathlib import Path

from . import theme
from .qt_compat import QByteArray, QIcon, QPixmap

__all__ = ["NAV_ICON_NAMES", "nav_icon"]

_ICONS_DIR = Path(__file__).resolve().parent.parent / "assets" / "icons"

#: 侧边栏页序对应的图标文件基名（与 MainWindow 页序一致）
NAV_ICON_NAMES = ("console", "analysis", "about")


def nav_icon(name: str, color: str | None = None) -> QIcon:
    """加载单色 SVG 并以主题导航色着色.

    Args:
        name: 图标文件基名（如 ``console``）。
        color: 着色十六进制串；缺省用当前主题 ``nav_text``，
            选中态建议传 ``nav_accent``。

    Returns:
        着色后的 QIcon（背景透明）；SVG 缺失或渲染失败时返回空 QIcon
        （界面退化为纯文字导航，不抛错）。
    """
    svg_path = _ICONS_DIR / f"{name}.svg"
    try:
        text = svg_path.read_text(encoding="utf-8")
    except OSError:
        return QIcon()
    pal = theme.current_palette()
    tint = color if color is not None else pal.nav_text
    # 根元素注入 fill（可继承）；仅首个 <svg 出现处插入，path 无显式 fill 时生效
    tinted = text.replace("<svg ", f'<svg fill="{tint}" ', 1)
    pixmap = QPixmap()
    if not pixmap.loadFromData(QByteArray(tinted.encode("utf-8")), "SVG"):
        return QIcon()
    return QIcon(pixmap)
