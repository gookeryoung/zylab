"""绘图页：pyqtgraph 渲染 REPL 的 plot 请求事件."""

from __future__ import annotations

import pyqtgraph as pg

from zylab.core import EventBus
from zylab.sci import TOPIC_PLOT_REQUESTED, PlotRequest

from .. import theme
from ..qt_compat import QVBoxLayout, QWidget, Signal

__all__ = ["PlotPage"]


class PlotPage(QWidget):
    """绘图页：订阅 ``sci.plot.requested`` 事件并渲染曲线，渲染后发出 :attr:`plot_shown`."""

    plot_shown = Signal()

    def __init__(self, bus: EventBus, parent: QWidget | None = None) -> None:
        """初始化绘图页并订阅绘图事件."""
        super().__init__(parent)
        pal = theme.current_palette()
        self._plot = pg.PlotWidget(background=pal.bg_app)
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._pen = pg.mkPen(pal.primary, width=2)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD)
        layout.addWidget(self._plot)
        bus.subscribe(TOPIC_PLOT_REQUESTED, self._on_plot_request)

    def _on_plot_request(self, request: PlotRequest) -> None:
        """渲染绘图请求（REPL 在主线程执行，直接渲染安全）."""
        if request.clear:
            self._plot.clear()
        self._plot.plot(request.x, request.y, pen=self._pen)
        if request.title:
            self._plot.setTitle(request.title)
        if request.xlabel:
            self._plot.setLabel("bottom", request.xlabel)
        if request.ylabel:
            self._plot.setLabel("left", request.ylabel)
        self.plot_shown.emit()
