"""界面图标：assets/icons 单色 SVG 按主题令牌着色（源码注入法）.

SVG 源为单色剪影（path 无 fill 属性，默认黑）。着色方式为在 <svg> 根元素
注入 ``fill="颜色"``（SVG fill 可继承到 path），再经 loadFromData 渲染 ——
背景天然透明，不依赖 QPainter 合成模式的平台行为。
主题切换后由调用方重新生成。
"""

from __future__ import annotations

from pathlib import Path

from . import theme
from .qt_compat import QByteArray, QIcon, QPixmap, Qt

__all__ = ["NAV_ICON_NAMES", "load_icon", "nav_icon", "tinted_pixmap"]

_ICONS_DIR = Path(__file__).resolve().parent.parent / "assets" / "icons"

#: 像素图缓存（name, tint, size）-> QPixmap；主题色随主题变化天然分键
_PIXMAP_CACHE: dict[tuple[str, str, int], QPixmap] = {}

#: 侧边栏页序对应的图标文件基名（与 MainWindow 页序一致）
NAV_ICON_NAMES = ("console", "analysis", "about")


def load_icon(name: str, size: int = 16) -> QIcon:
    """加载 SVG 图标原色渲染并缩放到指定尺寸.

    与 :func:`nav_icon` 不同，本函数不做主题着色（根元素注入 fill
    只对无显式 fill 的单色剪影生效，彩色图标 path 自带 fill）。

    Args:
        name: 图标文件基名（如 ``save_project``）。
        size: 目标渲染尺寸（像素）。

    Returns:
        原色 QIcon；SVG 缺失或渲染失败时返回空 QIcon（界面退化为纯文字）。
    """
    svg_path = _ICONS_DIR / f"{name}.svg"
    try:
        text = svg_path.read_text(encoding="utf-8")
    except OSError:
        return QIcon()
    pixmap = QPixmap()
    if not pixmap.loadFromData(QByteArray(text.encode("utf-8")), "SVG"):
        return QIcon()
    if pixmap.width() != size or pixmap.height() != size:
        pixmap = pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return QIcon(pixmap)


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


def tinted_pixmap(name: str, color: str, size: int = 16) -> QPixmap:
    """加载单色 SVG 着色为指定尺寸 QPixmap（QGraphicsItem.paint 直接绘制用）.

    与 :func:`nav_icon` 同源码注入着色，返回 QPixmap 而非 QIcon；
    结果按 ``(name, color, size)`` 缓存（画布每帧重绘高频调用）。

    Args:
        name: 图标文件基名（如 ``check``/``cross``/``question``）。
        color: 着色十六进制串（建议传主题语义色）。
        size: 目标边长（像素，正方形）。

    Returns:
        着色 QPixmap；SVG 缺失或渲染失败时返回空 QPixmap（绘制为空）。
    """
    key = (name, color, size)
    cached = _PIXMAP_CACHE.get(key)
    if cached is not None:
        return cached
    svg_path = _ICONS_DIR / f"{name}.svg"
    try:
        text = svg_path.read_text(encoding="utf-8")
    except OSError:
        return QPixmap()
    tinted = text.replace("<svg ", f'<svg fill="{color}" ', 1)
    pixmap = QPixmap()
    if not pixmap.loadFromData(QByteArray(tinted.encode("utf-8")), "SVG"):
        return QPixmap()
    if pixmap.width() != size or pixmap.height() != size:
        pixmap = pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    _PIXMAP_CACHE[key] = pixmap
    return pixmap
