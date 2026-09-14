"""simple_heatmap - 纯 QPainter 热力图（无需 matplotlib / pyqtgraph）.

将 2D 标量数组渲染为矩形格子色带。支持：
- 线性颜色映射（自定义 colormap 或内置 viridis 渐变）；
- 数值范围裁剪（clip_min / clip_max）；
- 可选数值标注；
- 颜色条（colorbar）自动绘制；
- 颜色从 theme.current_palette() 取，也可显式覆盖。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import theme
from ..qt_compat import (
    QColor,
    QLinearGradient,
    QPainter,
    QRectF,
    QSizePolicy,
    Qt,
    QWidget,
)

__all__ = ["SimpleHeatmap"]


def _viridis_colormap() -> list[tuple[float, str]]:
    """内置 viridis-like 渐变色控制点（位置 0~1，颜色 ``#RRGGBB``）."""
    return [
        (0.00, "#440154"),
        (0.25, "#3B528B"),
        (0.50, "#21918C"),
        (0.75, "#5EC962"),
        (1.00, "#FDE725"),
    ]


class SimpleHeatmap(QWidget):
    """轻量 2D 热力图.

    :param parent: 父 QWidget
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data: np.ndarray | None = None  # 2D 数组
        self._clip_min: float | None = None
        self._clip_max: float | None = None
        self._show_values = False
        self._title = ""
        self._colormap: list[tuple[float, str]] = _viridis_colormap()
        self._margins = (12, 24, 60, 24)  # 左/下/右(给colorbar)/上
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # -------------------------------------------------------------- 公共 API

    def set_data(  # noqa: PLR0913  热力图数据设置参数多为正常
        self,
        data: Sequence[Sequence[float]],
        *,
        clip_min: float | None = None,
        clip_max: float | None = None,
        show_values: bool = False,
        title: str = "",
        colormap: list[tuple[float, str]] | None = None,
    ) -> None:
        """设置热力图数据.

        :param data: 二维标量数组（行 × 列）
        :param clip_min: 颜色映射下界（None 取数组最小值）
        :param clip_max: 颜色映射上界（None 取数组最大值）
        :param show_values: 是否在格子上绘制数值文本
        :param title: 图标题
        :param colormap: 自定义渐变色 ``[(position, "#RRGGBB"), ...]``；position 须递增 0~1
        """
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 2 or arr.size == 0:
            self._data = None
        else:
            self._data = arr
        self._clip_min = clip_min
        self._clip_max = clip_max
        self._show_values = show_values
        self._title = title
        if colormap is not None:
            self._colormap = list(colormap)
        self.update()

    def clear(self) -> None:
        """清空数据."""
        self._data = None
        self._title = ""
        self.update()

    # -------------------------------------------------------------- 绘制

    def paintEvent(self, _event) -> None:
        if self._data is None or self._data.size == 0:
            self._paint_placeholder()
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        pal = theme.current_palette()
        self._draw(painter, pal)
        painter.end()

    def _paint_placeholder(self) -> None:
        """空数据占位."""
        pal = theme.current_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QColor(pal.text_secondary))
        painter.drawText(self.rect(), 0, "（暂无数据）")
        painter.end()

    def _draw(self, painter: QPainter, pal: theme.Palette) -> None:
        data = self._data
        assert data is not None
        rows, cols = data.shape
        w, h = self.width(), self.height()
        ml, mb, mr, mt = self._margins
        # colorbar 占右侧 32px
        plot_rect = QRectF(ml, mt, max(10, w - ml - mr - 36), max(10, h - mt - mb))
        cb_rect = QRectF(plot_rect.right() + 8, plot_rect.top(), 20, plot_rect.height())

        # --- 背景 ---
        bg = QColor(pal.bg_app)
        painter.fillRect(self.rect(), bg)
        painter.fillRect(plot_rect, bg)

        # --- 数据范围 ---
        vmin = self._clip_min if self._clip_min is not None else float(np.min(data))
        vmax = self._clip_max if self._clip_max is not None else float(np.max(data))
        if vmin == vmax:
            vmax = vmin + 1.0

        # --- 网格尺寸 ---
        cell_w = plot_rect.width() / cols
        cell_h = plot_rect.height() / rows

        # --- 颜色映射 ---
        def val_to_color(val: float) -> QColor:
            t = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
            # 在 colormap 控制点间线性插值
            cm = self._colormap
            for i in range(len(cm) - 1):
                t0, c0 = cm[i]
                t1, c1 = cm[i + 1]
                if t0 <= t <= t1:
                    frac = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
                    return _interp_color(QColor(c0), QColor(c1), frac)
            return QColor(cm[-1][1])

        # --- 绘制格子 ---
        for r in range(rows):
            for c in range(cols):
                val = float(data[r, c])
                color = val_to_color(val)
                x = plot_rect.left() + c * cell_w
                y = plot_rect.top() + r * cell_h
                rect = QRectF(x + 0.5, y + 0.5, cell_w - 1, cell_h - 1)
                painter.fillRect(rect, color)
                if self._show_values and cell_w > 12 and cell_h > 10:
                    painter.setPen(QColor(pal.text_primary))
                    painter.drawText(rect, Qt.AlignCenter, f"{val:.2g}")

        # --- colorbar ---
        gradient = QLinearGradient(cb_rect.bottom(), 0, cb_rect.top(), 0)
        for t_pos, hex_color in self._colormap:
            gradient.setColorAt(1.0 - t_pos, QColor(hex_color))  # 反转：下小上大
        painter.fillRect(cb_rect, gradient)
        painter.setPen(QColor(pal.border_strong))
        painter.drawRect(cb_rect)
        # colorbar 刻度
        painter.setPen(QColor(pal.text_secondary))
        cb_fm = painter.fontMetrics()
        for i in range(5):
            frac = i / 4
            val = vmin + (vmax - vmin) * frac
            y = cb_rect.bottom() - frac * cb_rect.height()
            painter.drawText(
                int(cb_rect.right()) + 4,
                int(y) + cb_fm.ascent() // 2,
                f"{val:.3g}",
            )
        # colorbar 标题
        if self._title:
            painter.setPen(QColor(pal.text_primary))
            title_font = painter.font()
            title_font.setBold(True)
            painter.setFont(title_font)
            painter.drawText(int(plot_rect.left()), 20, int(plot_rect.width()), 20, Qt.AlignCenter, self._title)
            painter.setFont(painter.font())


def _interp_color(c0: QColor, c1: QColor, frac: float) -> QColor:
    """两色线性插值（frac: 0~1）."""
    r = int(c0.red() + (c1.red() - c0.red()) * frac)
    g = int(c0.green() + (c1.green() - c0.green()) * frac)
    b = int(c0.blue() + (c1.blue() - c0.blue()) * frac)
    return QColor(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
