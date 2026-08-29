"""zylab.gui.pages - 主窗口内容页（控制台/绘图/工作台/脚本）."""

from __future__ import annotations

from .console_page import ConsolePage, ReplInput, VarTableModel
from .plot_page import PlotPage
from .script_page import ScriptPage
from .studio_page import StudioPage

__all__ = ["ConsolePage", "PlotPage", "ReplInput", "ScriptPage", "StudioPage", "VarTableModel"]
