"""Qt 兼容层：PySide2（Python<=3.10 / Win7）与 PySide6（>=3.11）统一导入.

GUI 模块一律从这里导入 Qt 符号，版本差异（exec_/exec、枚举路径）在此收拢。
"""

from __future__ import annotations

try:
    from PySide6.QtCore import (
        QAbstractTableModel,
        QModelIndex,
        QObject,
        QSize,
        Qt,
        QTimer,
        Signal,
        Slot,
    )
    from PySide6.QtGui import QColor, QFont, QIcon, QKeyEvent, QPainter, QPixmap, QTextCursor
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSplitter,
        QStackedWidget,
        QTableView,
        QVBoxLayout,
        QWidget,
    )

    QT_API = "pyside6"
except ImportError:  # pragma: no cover（3.8 环境走此分支）
    from PySide2.QtCore import (  # type: ignore[missing-import]
        QAbstractTableModel,
        QModelIndex,
        QObject,
        QSize,
        Qt,
        QTimer,
        Signal,
        Slot,
    )
    from PySide2.QtGui import (  # type: ignore[missing-import]
        QColor,
        QFont,
        QIcon,
        QKeyEvent,
        QPainter,
        QPixmap,
        QTextCursor,
    )
    from PySide2.QtWidgets import (  # type: ignore[missing-import]
        QAbstractItemView,
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSplitter,
        QStackedWidget,
        QTableView,
        QVBoxLayout,
        QWidget,
    )

    QT_API = "pyside2"

__all__ = [
    "QT_API",
    "QAbstractItemView",
    "QAbstractTableModel",
    "QApplication",
    "QColor",
    "QComboBox",
    "QDoubleSpinBox",
    "QFont",
    "QFormLayout",
    "QFrame",
    "QGroupBox",
    "QHBoxLayout",
    "QHeaderView",
    "QIcon",
    "QKeyEvent",
    "QLabel",
    "QLineEdit",
    "QListWidget",
    "QListWidgetItem",
    "QMainWindow",
    "QModelIndex",
    "QObject",
    "QPainter",
    "QPixmap",
    "QPlainTextEdit",
    "QProgressBar",
    "QPushButton",
    "QSize",
    "QSplitter",
    "QStackedWidget",
    "QTableView",
    "QTextCursor",
    "QTimer",
    "QVBoxLayout",
    "QWidget",
    "Qt",
    "Signal",
    "Slot",
]


def exec_app(app: QApplication) -> int:
    """启动事件循环（兼容 PySide2 的 exec_ 与 PySide6 的 exec）."""
    run = app.exec if hasattr(app, "exec") else app.exec_  # type: ignore[union-attr]
    return int(run())
