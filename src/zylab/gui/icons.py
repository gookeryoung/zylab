"""界面图标：assets/icons 单色 SVG 按主题令牌着色（alpha 掩码法）.

SVG 源为单色剪影（无 fill，默认黑），通过 CompositionMode_DestinationIn
以源图 alpha 为掩码将纯色图层裁出图形，实现随主题换色；
主题切换后由调用方重新生成。
"""

from __future__ import annotations

from pathlib import Path

from . import theme
from .qt_compat import QColor, QIcon, QPainter, QPixmap, Qt

__all__ = ["NAV_ICON_NAMES", "nav_icon"]

_ICONS_DIR = Path(__file__).resolve().parent.parent / "assets" / "icons"

#: 侧边栏页序对应的图标文件基名（与 MainWindow 页序一致）
NAV_ICON_NAMES = ("console", "plot", "analysis", "script", "about")

_ICON_SIZE = 48  # 高分辨率源，Qt 按控件 iconSize 自动降采样


def nav_icon(name: str, color: str | None = None) -> QIcon:
    """加载单色 SVG 并以主题导航色着色.

    Args:
        name: 图标文件基名（如 ``console``）。
        color: 着色十六进制串；缺省用当前主题 ``nav_text``，
            选中态建议传 ``nav_accent``。

    Returns:
        着色后的 QIcon；SVG 缺失或渲染失败时返回空 QIcon
        （界面退化为纯文字导航，不抛错）。
    """
    source = QPixmap(str(_ICONS_DIR / f"{name}.svg"))
    if source.isNull():
        return QIcon()
    pal = theme.current_palette()
    tinted = QPixmap(source.size())
    tinted.fill(QColor(color if color is not None else pal.nav_text))
    painter = QPainter(tinted)
    painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
    painter.drawPixmap(0, 0, source)
    painter.end()
    return QIcon(tinted.scaled(_ICON_SIZE, _ICON_SIZE, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
