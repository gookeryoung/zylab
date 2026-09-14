"""zylab.gui - 桌面 GUI 层（PySide2/PySide6 双兼容，依赖 Qt 的唯一层级）."""

from __future__ import annotations

from .app import create_app, load_stylesheet, main
from .main_window import MainWindow
from .perf import PerfReport, PerfStats, render_startup_summary, timed
from .qt_compat import QT_API

__all__ = [
    "QT_API",
    "MainWindow",
    "PerfReport",
    "PerfStats",
    "create_app",
    "load_stylesheet",
    "main",
    "render_startup_summary",
    "timed",
]
