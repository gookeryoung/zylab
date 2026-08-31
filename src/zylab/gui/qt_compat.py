"""Qt 兼容层：PySide2（Python<=3.10 / Win7）与 PySide6（>=3.11）统一导入.

GUI 模块一律从这里导入 Qt 符号，版本差异（exec_/exec、枚举路径）在此收拢。
"""

from __future__ import annotations

try:
    from PySide6.QtCore import (
        QAbstractTableModel,
        QByteArray,
        QEvent,
        QModelIndex,
        QObject,
        QPoint,
        QPointF,
        QRect,
        QRectF,
        QSize,
        Qt,
        QTimer,
        Signal,
        Slot,
    )
    from PySide6.QtGui import (
        QBrush,
        QColor,
        QContextMenuEvent,
        QFont,
        QFontMetrics,
        QIcon,
        QKeyEvent,
        QKeySequence,
        QMouseEvent,
        QPainter,
        QPainterPath,
        QPen,
        QPixmap,
        QSyntaxHighlighter,
        QTextCharFormat,
        QTextCursor,
    )
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGraphicsItem,
        QGraphicsPathItem,
        QGraphicsScene,
        QGraphicsView,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QInputDialog,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMenu,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QShortcut,
        QSlider,
        QSpinBox,
        QSplitter,
        QStackedWidget,
        QStyle,
        QStyledItemDelegate,
        QStyleOptionViewItem,
        QTableView,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    QT_API = "pyside6"
except ImportError:  # pragma: no cover（3.8 环境走此分支）
    from PySide2.QtCore import (  # type: ignore[missing-import]
        QAbstractTableModel,
        QByteArray,
        QEvent,
        QModelIndex,
        QObject,
        QPoint,
        QPointF,
        QRect,
        QRectF,
        QSize,
        Qt,
        QTimer,
        Signal,
        Slot,
    )
    from PySide2.QtGui import (  # type: ignore[missing-import]
        QBrush,
        QColor,
        QContextMenuEvent,
        QFont,
        QFontMetrics,  # type: ignore[missing-import]
        QIcon,
        QKeyEvent,
        QKeySequence,  # type: ignore[missing-import]
        QMouseEvent,
        QPainter,
        QPainterPath,
        QPen,
        QPixmap,
        QSyntaxHighlighter,  # type: ignore[missing-import]
        QTextCharFormat,  # type: ignore[missing-import]
        QTextCursor,
    )
    from PySide2.QtWidgets import (  # type: ignore[missing-import]
        QAbstractItemView,
        QApplication,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGraphicsItem,
        QGraphicsPathItem,
        QGraphicsScene,
        QGraphicsView,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QInputDialog,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMenu,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QShortcut,
        QSlider,
        QSpinBox,
        QSplitter,
        QStackedWidget,
        QStyle,
        QStyledItemDelegate,
        QStyleOptionViewItem,
        QTableView,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    QT_API = "pyside2"

__all__ = [
    "QT_API",
    "QAbstractItemView",
    "QAbstractTableModel",
    "QApplication",
    "QBrush",
    "QByteArray",
    "QColor",
    "QComboBox",
    "QContextMenuEvent",
    "QDialog",
    "QDialogButtonBox",
    "QDoubleSpinBox",
    "QEvent",
    "QFileDialog",
    "QFont",
    "QFontMetrics",
    "QFormLayout",
    "QFrame",
    "QGraphicsItem",
    "QGraphicsPathItem",
    "QGraphicsScene",
    "QGraphicsView",
    "QGroupBox",
    "QHBoxLayout",
    "QHeaderView",
    "QIcon",
    "QInputDialog",
    "QKeyEvent",
    "QKeySequence",
    "QLabel",
    "QLineEdit",
    "QListWidget",
    "QListWidgetItem",
    "QMainWindow",
    "QMenu",
    "QMessageBox",
    "QModelIndex",
    "QMouseEvent",
    "QObject",
    "QPainter",
    "QPainterPath",
    "QPen",
    "QPixmap",
    "QPlainTextEdit",
    "QPoint",
    "QPointF",
    "QProgressBar",
    "QPushButton",
    "QRect",
    "QRectF",
    "QScrollArea",
    "QShortcut",
    "QSize",
    "QSlider",
    "QSpinBox",
    "QSplitter",
    "QStackedWidget",
    "QStyle",
    "QStyleOptionViewItem",
    "QStyledItemDelegate",
    "QSyntaxHighlighter",
    "QTabWidget",
    "QTableView",
    "QTableWidget",
    "QTableWidgetItem",
    "QTextCharFormat",
    "QTextCursor",
    "QTimer",
    "QTreeWidget",
    "QTreeWidgetItem",
    "QVBoxLayout",
    "QWidget",
    "Qt",
    "Signal",
    "Slot",
    "exec_app",
    "exec_menu",
    "mouse_event_pos",
]


def exec_app(app: QApplication) -> int:
    """启动事件循环（兼容 PySide2 的 exec_ 与 PySide6 的 exec）."""
    run = app.exec if hasattr(app, "exec") else app.exec_  # type: ignore[union-attr]
    return int(run())


def mouse_event_pos(event: QMouseEvent) -> QPoint:
    """鼠标事件视图坐标（Qt6 的 position / Qt5 的 pos 兼容）."""
    pos = event.position() if hasattr(event, "position") else event.pos()  # type: ignore[union-attr]
    # PySide2 的 pos() 已是 QPoint（无 toPoint），Qt6 的 position() 是 QPointF 须转换
    return pos.toPoint() if hasattr(pos, "toPoint") else pos  # type: ignore[union-attr]


def exec_menu(menu: QMenu, pos: QPoint) -> object:
    """弹出上下文菜单并返回所选 action（兼容 PySide2 的 exec_ 与 PySide6 的 exec）."""
    popup = menu.exec if hasattr(menu, "exec") else menu.exec_  # type: ignore[union-attr]
    return popup(pos)


def exec_dialog(dialog: QDialog) -> int:
    """模态执行对话框（兼容 PySide2 的 exec_ 与 PySide6 的 exec）."""
    run = dialog.exec if hasattr(dialog, "exec") else dialog.exec_  # type: ignore[union-attr]
    return int(run())
