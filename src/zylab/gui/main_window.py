"""主窗口：头部条 + 侧边栏导航 + 内容区（QStackedWidget）+ 状态栏."""

from __future__ import annotations

import platform

from zylab import __version__
from zylab.console import ReplKernel
from zylab.core import EventBus, default_data_dir

from . import theme
from .app import apply_theme, save_theme_name
from .icons import NAV_ICON_NAMES, nav_icon
from .pages.notebook_page import NotebookPage
from .pages.studio_page import StudioPage
from .qt_compat import (
    QEvent,
    QFrame,
    QHBoxLayout,
    QKeySequence,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QShortcut,
    QSize,
    QSplitter,
    QStackedWidget,
    Qt,
    QVBoxLayout,
    QWidget,
)
from .widgets.command_palette import Command, CommandPalette

__all__ = ["MainWindow"]

_PAGE_CONSOLE = 0
_PAGE_FEA = 1
_PAGE_ABOUT = 2

#: 侧边栏图标显示尺寸（像素）
_NAV_ICON_SIZE = QSize(18, 18)


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

        self._build_ui()
        self._setup_command_palette()
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
        for label in ("笔记本", "分析", "关于"):
            QListWidgetItem(label, self._sidebar)
        self._sidebar.setIconSize(_NAV_ICON_SIZE)
        self._sidebar.setFixedWidth(theme.SIDEBAR_WIDTH)
        self._sidebar.setCurrentRow(_PAGE_CONSOLE)
        self._refresh_sidebar_icons()

        self._stack = QStackedWidget()
        self._notebook_page = NotebookPage(self._kernel, self._bus)
        self._studio_page = StudioPage()
        self._about_page = self._build_about_page()
        self._stack.addWidget(self._notebook_page)
        self._stack.addWidget(self._studio_page)
        self._stack.addWidget(self._about_page)

        splitter.addWidget(self._sidebar)
        splitter.addWidget(self._stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([theme.SIDEBAR_WIDTH, 1080])
        root.addWidget(splitter, stretch=1)
        self.setCentralWidget(central)

    def _build_header(self) -> QFrame:
        """构建头部条：左侧标题 + 居中命令搜索框 + 右侧运行环境信息."""
        bar = QFrame(objectName="headerBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(theme.SPACING_MD, 0, theme.SPACING_MD, 0)
        layout.addWidget(QLabel("zylab", objectName="headerTitle"))
        # 命令搜索框（VS Code 命令面板入口：点击或 Ctrl+Shift+P 弹出）
        self._command_search = QLineEdit(objectName="commandSearch")
        self._command_search.setReadOnly(True)
        self._command_search.setPlaceholderText("搜索功能 (Ctrl+Shift+P)")
        self._command_search.setFixedWidth(300)
        self._command_search.setFixedHeight(26)
        self._command_search.installEventFilter(self)
        layout.addWidget(self._command_search, stretch=1, alignment=Qt.AlignHCenter)
        meta = QLabel(f"Python {platform.python_version()} · v{__version__}", objectName="headerMeta")
        layout.addWidget(meta)
        return bar

    def _set_theme(self, name: str, persist: bool) -> None:
        """应用主题并刷新全部页面（persist 时持久化并提示状态栏）.

        命令面板主题预览（persist=False）与确认（persist=True）共用；
        预览只切样式不落盘，Esc 取消由面板发还原主题信号。
        """
        from .qt_compat import QApplication

        if name != theme.current_palette().name:
            apply_theme(QApplication.instance(), name)
        self._refresh_sidebar_icons()
        self._notebook_page.refresh_theme()
        self._studio_page.refresh_theme()
        if persist:
            save_theme_name(default_data_dir(), name)
            self.statusBar().showMessage(f"主题已切换: {theme.current_palette().display_name}")

    def _refresh_sidebar_icons(self) -> None:
        """按当前主题色重绘侧边栏图标（选中行用强调色）."""
        pal = theme.current_palette()
        for row, name in enumerate(NAV_ICON_NAMES):
            item = self._sidebar.item(row)
            if item is not None:
                color = pal.nav_accent if row == self._sidebar.currentRow() else pal.nav_text
                item.setIcon(nav_icon(name, color))

    def _build_about_page(self) -> QWidget:
        """构建关于页."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(theme.SPACING_LG, theme.SPACING_LG, theme.SPACING_LG, theme.SPACING_LG)
        title = QLabel("zylab 通用科学计算仿真分析平台", objectName="pageTitle")
        layout.addWidget(title)
        layout.addWidget(QLabel(f"版本 {__version__} · Python {platform.python_version()} · 离线可用"))
        layout.addStretch()
        return page

    def _connect(self) -> None:
        """连接导航与跨页信号."""
        self._sidebar.currentRowChanged.connect(self._stack.setCurrentIndex)
        self._sidebar.currentRowChanged.connect(lambda _row: self._refresh_sidebar_icons())
        # 笔记本页状态提示（已打开/已保存/保存失败等）统一进主窗口状态栏
        self._notebook_page.status_message.connect(self.statusBar().showMessage)

    def _setup_command_palette(self) -> None:
        """装配命令面板：注册全局命令 + Ctrl+Shift+P 快捷键."""
        self._palette = CommandPalette(self)
        self._palette.theme_previewed.connect(lambda name: self._set_theme(name, persist=False))
        self._palette.theme_confirmed.connect(lambda name: self._set_theme(name, persist=True))
        self._register_commands()
        QShortcut(QKeySequence("Ctrl+Shift+P"), self, self._palette.open_commands)

    def _register_commands(self) -> None:
        """注册全局命令（页面导航 / 笔记本操作 / 主题切换）."""
        page = self._notebook_page
        register = self._palette.register
        register(
            Command(
                "go.notebook",
                "转到：笔记本",
                lambda: self._sidebar.setCurrentRow(_PAGE_CONSOLE),
                keywords="goto notebook",
            )
        )
        register(
            Command(
                "go.analysis",
                "转到：分析",
                lambda: self._sidebar.setCurrentRow(_PAGE_FEA),
                keywords="goto analysis fea",
            )
        )
        register(
            Command("go.about", "转到：关于", lambda: self._sidebar.setCurrentRow(_PAGE_ABOUT), keywords="goto about")
        )
        register(Command("notebook.new", "新建笔记本", page.new_notebook, keywords="new notebook", shortcut="Ctrl+N"))
        register(
            Command("notebook.open", "打开笔记本", page.open_notebook, keywords="open notebook", shortcut="Ctrl+O")
        )
        register(
            Command("notebook.save", "保存笔记本", page.save_notebook, keywords="save notebook", shortcut="Ctrl+S")
        )
        register(Command("notebook.run_all", "全部运行（出错不中断）", page.run_all, keywords="run all"))
        register(
            Command("notebook.restart", "重启内核（清空变量与输出）", page.restart_kernel, keywords="restart kernel")
        )
        register(
            Command("theme.select", "选择主题（上下键实时预览）", self._palette.open_theme_picker, keywords="theme")
        )

    def eventFilter(self, obj, event) -> bool:  # Qt 命名约定
        """点击头部命令搜索框弹出命令面板（只读框仅作入口）."""
        if obj is self._command_search and event.type() == QEvent.MouseButtonPress:
            self._palette.open_commands()
            return True
        return super().eventFilter(obj, event)

    def closeEvent(self, event) -> None:  # Qt 命名约定
        """关闭前询问保存笔记本，确认后终止后台求解执行器."""
        if not self._notebook_page.maybe_save():
            event.ignore()
            return
        self._studio_page.shutdown()
        super().closeEvent(event)
