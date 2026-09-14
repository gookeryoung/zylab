"""simple_line_plot - 纯 QPainter 折线图（无需 pyqtgraph / matplotlib）.

作为 pyqtgraph 不可用时的轻量 fallback，或嵌入式小图场景。
- 仅依赖 PySide QPainter + QPainterPath；
- 自动计算坐标变换（线性映射 x/y 数据范围到 widget 矩形）；
- 支持网格线、坐标轴标签、图例；
- 颜色从 theme.current_palette() 取，支持显式覆盖。
"""

from __future__ import annotations

from collections.abc import Sequence

from .. import theme
from ..qt_compat import (
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
    QRectF,
    QSizePolicy,
    Qt,
    QWidget,
)

__all__ = ["SimpleLinePlot"]


class SimpleLinePlot(QWidget):
    """轻量折线图（单条或多条曲线）.

    :param parent: 父 QWidget
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._series: list[dict[str, object]] = []
        self._x_label = ""
        self._y_label = ""
        self._title = ""
        # 外边距（左/下/右/上）
        self._margins = (56, 36, 16, 28)
        self.setMinimumSize(200, 120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # -------------------------------------------------------------- 公共 API

    def set_data(
        self,
        series: Sequence[tuple[Sequence[float], Sequence[float], str]],
        *,
        x_label: str = "",
        y_label: str = "",
        title: str = "",
        colors: Sequence[str] | None = None,
    ) -> None:
        """设置折线数据.

        :param series: 元组列表 ``[(x_values, y_values, label), ...]``
        :param x_label: X 轴标签
        :param y_label: Y 轴标签
        :param title: 图标题
        :param colors: 可选颜色覆盖（长度须与 series 等长；每个值 ``#RRGGBB``）
        """
        pal = theme.current_palette()
        fallback_colors = [pal.primary, pal.nav_accent, pal.success_text, pal.warning_text, pal.danger_text]
        self._series = []
        for i, (xs, ys, label) in enumerate(series):
            color = colors[i] if colors and i < len(colors) else fallback_colors[i % len(fallback_colors)]
            self._series.append({"x": list(xs), "y": list(ys), "label": label, "color": color})
        self._x_label = x_label
        self._y_label = y_label
        self._title = title
        self.update()

    def clear(self) -> None:
        """清空全部数据."""
        self._series.clear()
        self._x_label = ""
        self._y_label = ""
        self._title = ""
        self.update()

    # -------------------------------------------------------------- 绘制

    def paintEvent(self, _event) -> None:
        if not self._series:
            self._paint_placeholder()
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        pal = theme.current_palette()
        self._draw(painter, pal)
        painter.end()

    def _paint_placeholder(self) -> None:
        """空数据时显示占位."""
        pal = theme.current_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QColor(pal.text_secondary))
        painter.drawText(self.rect(), 0, "（暂无数据）")
        painter.end()

    def _draw(self, painter: QPainter, pal: theme.Palette) -> None:  # noqa: PLR0912  绘制分支多为正常
        # --- 布局 ---
        w, h = self.width(), self.height()
        ml, mb, mr, mt = self._margins
        plot_rect = QRectF(ml, mt, max(10, w - ml - mr), max(10, h - mt - mb))

        # --- 计算数据范围 ---
        x_min = min(min(s["x"]) for s in self._series)
        x_max = max(max(s["x"]) for s in self._series)
        y_min = min(min(s["y"]) for s in self._series)
        y_max = max(max(s["y"]) for s in self._series)
        # 单点时避免零范围
        if x_min == x_max:
            x_min, x_max = x_min - 1, x_max + 1
        if y_min == y_max:
            y_min, y_max = y_min - 1, y_max + 1

        # --- 颜色 ---
        bg = QColor(pal.bg_app)
        fg = QColor(pal.text_primary)
        fg2 = QColor(pal.text_secondary)
        grid_color = QColor(pal.border)
        grid_color.setAlpha(128)

        # --- 背景 ---
        painter.fillRect(self.rect(), bg)
        painter.fillRect(plot_rect, bg)

        # --- 网格（X 向 + Y 向各 5 条） ---
        painter.setPen(QPen(grid_color, 1, Qt.DotLine))
        n_grid = 5
        for i in range(n_grid + 1):
            px = plot_rect.left() + plot_rect.width() * i / n_grid
            painter.drawLine(int(px), int(plot_rect.top()), int(px), int(plot_rect.bottom()))
            py = plot_rect.top() + plot_rect.height() * i / n_grid
            painter.drawLine(int(plot_rect.left()), int(py), int(plot_rect.right()), int(py))

        # --- 坐标轴 ---
        painter.setPen(QPen(QColor(pal.border_strong), 1))
        painter.drawLine(
            int(plot_rect.left()), int(plot_rect.bottom()), int(plot_rect.right()), int(plot_rect.bottom())
        )
        painter.drawLine(int(plot_rect.left()), int(plot_rect.top()), int(plot_rect.left()), int(plot_rect.bottom()))

        # --- 坐标变换 ---
        def tx(val: float) -> float:
            return plot_rect.left() + (val - x_min) / (x_max - x_min) * plot_rect.width()

        def ty(val: float) -> float:
            return plot_rect.bottom() - (val - y_min) / (y_max - y_min) * plot_rect.height()

        # --- 刻度 ---
        painter.setPen(fg2)
        fm = QFontMetrics(painter.font())
        for i in range(n_grid + 1):
            xv = x_min + (x_max - x_min) * i / n_grid
            yv = y_min + (y_max - y_min) * i / n_grid
            px = tx(xv)
            py = ty(yv)
            label_x = f"{xv:.3g}"
            label_y = f"{yv:.3g}"
            painter.drawText(int(px) - fm.horizontalAdvance(label_x) // 2, int(plot_rect.bottom()) + 16, label_x)
            painter.drawText(4, int(py) + fm.ascent() // 2, label_y)

        # --- 轴标签 ---
        painter.setPen(fg)
        if self._x_label:
            painter.drawText(
                int(plot_rect.left()),
                int(h - 4),
                int(plot_rect.width()),
                20,
                Qt.AlignCenter,
                self._x_label,
            )
        if self._y_label:
            painter.save()
            painter.translate(12, int(plot_rect.top() + plot_rect.height() / 2))
            painter.rotate(-90)
            painter.drawText(-60, 4, 120, 20, Qt.AlignCenter, self._y_label)
            painter.restore()

        # --- 标题 ---
        if self._title:
            title_font = QFont(painter.font())
            title_font.setBold(True)
            painter.setFont(title_font)
            painter.drawText(int(plot_rect.left()), 20, int(plot_rect.width()), 24, Qt.AlignCenter, self._title)
            painter.setFont(QFont())

        # --- 折线 ---
        for s in self._series:
            xs: list[float] = s["x"]  # type: ignore[assignment]
            ys: list[float] = s["y"]  # type: ignore[assignment]
            color = QColor(str(s["color"]))
            painter.setPen(QPen(color, 2))
            path = QPainterPath()
            if xs and ys:
                path.moveTo(tx(xs[0]), ty(ys[0]))
                for xv, yv in zip(xs[1:], ys[1:], strict=False):
                    path.lineTo(tx(xv), ty(yv))
            painter.drawPath(path)
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            for xv, yv in zip(xs, ys, strict=False):
                painter.drawEllipse(tx(xv) - 2, ty(yv) - 2, 4, 4)

        # --- 图例（右上角，最多 4 条） ---
        legend_limit = min(len(self._series), 4)
        if legend_limit > 0:
            painter.setPen(fg)
            legend_x = plot_rect.right() - 120
            legend_y = plot_rect.top() + 8
            for i in range(legend_limit):
                s = self._series[i]
                color = QColor(str(s["color"]))
                painter.setBrush(color)
                painter.setPen(Qt.NoPen)
                painter.drawRect(int(legend_x), int(legend_y + i * 18), 14, 3)
                painter.setPen(fg)
                label = str(s["label"])
                painter.drawText(int(legend_x) + 18, int(legend_y + i * 18) + 10, label)
