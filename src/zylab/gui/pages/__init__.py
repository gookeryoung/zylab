"""zylab.gui.pages - 主窗口内容页（控制台/绘图）."""

from __future__ import annotations

from .console_page import ConsolePage, ReplInput, VarTableModel
from .plot_page import PlotPage

__all__ = ["ConsolePage", "PlotPage", "ReplInput", "VarTableModel"]
