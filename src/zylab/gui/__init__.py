"""zylab.gui - 桌面 GUI 层（PySide2/PySide6 双兼容，依赖 Qt 的唯一层级）."""

from __future__ import annotations

from .app import create_app, load_stylesheet, main
from .main_window import MainWindow
from .qt_compat import QT_API

__all__ = ["QT_API", "MainWindow", "create_app", "load_stylesheet", "main"]
