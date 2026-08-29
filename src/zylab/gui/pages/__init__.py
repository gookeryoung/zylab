"""zylab.gui.pages - 主窗口内容页（控制台/绘图/分析/脚本）."""

from __future__ import annotations

from .console_page import ConsolePage, ReplInput, VarTableModel
from .fea_page import FeaPage
from .plot_page import PlotPage
from .script_page import ScriptPage

__all__ = ["ConsolePage", "FeaPage", "PlotPage", "ReplInput", "ScriptPage", "VarTableModel"]
