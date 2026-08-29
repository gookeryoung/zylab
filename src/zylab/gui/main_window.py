"""主窗口：头部条 + 侧边栏导航 + 内容区（QStackedWidget）+ 状态栏."""

from __future__ import annotations

import platform

from zylab import __version__
from zylab.console import CommandHistory, ReplKernel
from zylab.core import EventBus, default_data_dir

from . import theme
from .pages.console_page import ConsolePage
from .pages.fea_page import FeaPage
from .pages.plot_page import PlotPage
from .pages.script_page import ScriptPage
from .qt_compat import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSplitter,
    QStackedWidget,
    Qt,
    QVBoxLayout,
    QWidget,
)

__all__ = ["MainWindow"]

_PAGE_CONSOLE = 0
_PAGE_PLOT = 1
_PAGE_FEA = 2
_PAGE_SCRIPT = 3
_PAGE_ABOUT = 4


class MainWindow(QMainWindow):
    """zylab 主窗口（WSLDASHBOARD 式 SIDEBAR + CONTENT）."""

    def __init__(self) -> None:
        """初始化主窗口：装配内核、页面与导航."""
        super().__init__()
        self.setWindowTitle(f"zylab {__version__}")
        self.setMinimumSize(960, 640)
        self.resize(1280, 800)

        self._bus = EventBus()
        self._kernel = ReplKernel(self._bus)
        self._history = CommandHistory(default_data_dir() / "history.json")
        self._history.load()

        self._build_ui()
        self._connect()
        self.statusBar().showMessage("就绪")

    @property
    def kernel(self) -> ReplKernel:
        """REPL 内核（测试与外部集成用）."""
        return self._kernel

    def _build_ui(self) -> None:
        """组装四区布局."""
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())

        splitter = QSplitter(Qt.Horizontal)
        self._sidebar = QListWidget(objectName="sidebar")
        for label in ("控制台", "绘图", "分析", "脚本", "关于"):
            QListWidgetItem(label, self._sidebar)
        self._sidebar.setFixedWidth(theme.SIDEBAR_WIDTH)
        self._sidebar.setCurrentRow(_PAGE_CONSOLE)

        self._stack = QStackedWidget()
        self._console_page = ConsolePage(self._kernel, self._history)
        self._plot_page = PlotPage(self._bus)
        self._fea_page = FeaPage()
        self._script_page = ScriptPage(self._kernel)
        self._about_page = self._build_about_page()
        self._stack.addWidget(self._console_page)
        self._stack.addWidget(self._plot_page)
        self._stack.addWidget(self._fea_page)
        self._stack.addWidget(self._script_page)
        self._stack.addWidget(self._about_page)

        splitter.addWidget(self._sidebar)
        splitter.addWidget(self._stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([theme.SIDEBAR_WIDTH, 1080])
        root.addWidget(splitter, stretch=1)
        self.setCentralWidget(central)

    def _build_header(self) -> QFrame:
        """构建头部条：左侧标题，右侧运行环境信息."""
        bar = QFrame(objectName="headerBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(theme.SPACING_MD, 0, theme.SPACING_MD, 0)
        layout.addWidget(QLabel("zylab", objectName="headerTitle"))
        layout.addStretch()
        meta = QLabel(f"Python {platform.python_version()} · v{__version__}", objectName="headerMeta")
        layout.addWidget(meta)
        return bar

    def _build_about_page(self) -> QWidget:
        """构建关于页."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(theme.SPACING_LG, theme.SPACING_LG, theme.SPACING_LG, theme.SPACING_LG)
        title = QLabel("zylab 通用科学计算仿真分析平台")
        title.setStyleSheet(f"font-size: {theme.FONT_TITLE}; font-weight: bold;")
        layout.addWidget(title)
        layout.addWidget(QLabel(f"版本 {__version__} · Python {platform.python_version()} · 离线可用"))
        layout.addStretch()
        return page

    def _connect(self) -> None:
        """连接导航与跨页信号."""
        self._sidebar.currentRowChanged.connect(self._stack.setCurrentIndex)
        # 绘图请求渲染后自动切换到绘图页
        self._plot_page.plot_shown.connect(lambda: self._sidebar.setCurrentRow(_PAGE_PLOT))

    def closeEvent(self, event) -> None:  # Qt 命名约定
        """关闭时持久化命令历史并终止后台求解执行器."""
        self._history.save()
        self._fea_page.shutdown()
        super().closeEvent(event)
