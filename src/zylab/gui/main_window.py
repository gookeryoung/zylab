"""主窗口：头部条 + 侧边栏导航 + 内容区（QStackedWidget）+ 状态栏."""

from __future__ import annotations

import platform

from zylab import __version__
from zylab.console import ReplKernel
from zylab.core import EventBus, default_data_dir
from zylab.sci import TOPIC_WORKSPACE_CHANGED, WorkspaceInfo, WorkspaceManager

from . import theme
from .app import apply_theme, save_theme_name
from .icons import NAV_ICON_NAMES, nav_icon
from .pages.notebook_page import NotebookPage
from .pages.studio_page import StudioPage
from .pages.template_page import TemplatePage
from .qt_compat import (
    QEvent,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QKeySequence,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
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
_PAGE_TEMPLATE = 2
_PAGE_ABOUT = 3

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
        # 工作区管理器先于内核：kernel.set_workspace_manager() 需要 WM 已构造好；
        # WM.load() 会触发 TOPIC_WORKSPACE_CHANGED 事件，内核已订阅后自动同步 namespace.cwd
        self._workspace_manager = WorkspaceManager(self._bus)
        self._workspace_manager.load()
        self._kernel = ReplKernel(self._bus)
        self._kernel.set_workspace_manager(self._workspace_manager)

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
        for label in ("笔记本", "分析", "模板", "关于"):
            QListWidgetItem(label, self._sidebar)
        self._sidebar.setIconSize(_NAV_ICON_SIZE)
        self._sidebar.setFixedWidth(theme.SIDEBAR_WIDTH)
        self._sidebar.setCurrentRow(_PAGE_CONSOLE)
        self._refresh_sidebar_icons()

        self._stack = QStackedWidget()
        self._notebook_page = NotebookPage(self._kernel, self._bus)
        self._studio_page = StudioPage()
        self._template_page = TemplatePage()
        self._about_page = self._build_about_page()
        self._stack.addWidget(self._notebook_page)
        self._stack.addWidget(self._studio_page)
        self._stack.addWidget(self._template_page)
        self._stack.addWidget(self._about_page)

        splitter.addWidget(self._sidebar)
        splitter.addWidget(self._stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([theme.SIDEBAR_WIDTH, 1080])
        root.addWidget(splitter, stretch=1)
        self.setCentralWidget(central)

    def _build_header(self) -> QFrame:
        """构建头部条：左侧标题 + 居中命令搜索框 + 工作区分组 + 分隔 + 环境版本分组."""
        bar = QFrame(objectName="headerBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(theme.SPACING_MD, 0, theme.SPACING_MD, 0)
        # 左：标题
        layout.addWidget(QLabel("zylab", objectName="headerTitle"), alignment=Qt.AlignVCenter)
        # 命令搜索框（VS Code 命令面板入口：点击或 Ctrl+Shift+P 弹出）
        self._command_search = QLineEdit(objectName="commandSearch")
        self._command_search.setReadOnly(True)
        self._command_search.setPlaceholderText("搜索功能 (Ctrl+Shift+P)")
        self._command_search.setFixedWidth(300)
        self._command_search.setFixedHeight(26)
        self._command_search.installEventFilter(self)
        layout.addWidget(self._command_search, stretch=1, alignment=Qt.AlignVCenter)

        # 工作区分组：tag + 路径 label + 切换按钮（包进 headerGroup 容器）
        workspace_group = QFrame(objectName="headerGroup")
        ws_layout = QHBoxLayout(workspace_group)
        ws_layout.setContentsMargins(theme.SPACING_XS, 2, theme.SPACING_XS, 2)
        ws_layout.setSpacing(theme.SPACING_XS)
        ws_tag = QLabel("工作区", objectName="headerTag")
        self._workspace_label = QLabel(objectName="workspaceLabel")
        self._workspace_label.setToolTip("当前工作区（MATLAB 风格 cwd）")
        self._workspace_label.setFixedHeight(26)
        self._workspace_btn = QPushButton(objectName="workspaceBtn")
        self._workspace_btn.setToolTip("切换工作区目录")
        self._workspace_btn.setFixedSize(26, 26)
        self._workspace_btn.setIconSize(QSize(14, 14))
        self._workspace_btn.clicked.connect(self._on_switch_workspace)
        ws_layout.addWidget(ws_tag)
        ws_layout.addWidget(self._workspace_label)
        ws_layout.addWidget(self._workspace_btn)
        layout.addWidget(workspace_group, alignment=Qt.AlignVCenter)

        # 竖向分隔线
        separator = QLabel(objectName="headerSeparator")
        separator.setFixedHeight(20)
        separator.setFixedWidth(1)
        layout.addWidget(separator, alignment=Qt.AlignVCenter)

        # 环境版本分组：tag + Python 版本 + 应用版本
        env_group = QFrame(objectName="headerGroup")
        env_layout = QHBoxLayout(env_group)
        env_layout.setContentsMargins(theme.SPACING_XS, 2, theme.SPACING_XS, 2)
        env_layout.setSpacing(theme.SPACING_XS)
        env_tag = QLabel("环境", objectName="headerTag")
        meta_py = QLabel(f"Python {platform.python_version()}", objectName="headerMeta")
        meta_ver = QLabel(f"v{__version__}", objectName="headerVersion")
        env_layout.addWidget(env_tag)
        env_layout.addWidget(meta_py)
        env_layout.addWidget(meta_ver)
        layout.addWidget(env_group, alignment=Qt.AlignVCenter)

        self._refresh_workspace_ui()
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
        self._refresh_workspace_ui()
        self._notebook_page.refresh_theme()
        self._studio_page.refresh_theme()
        self._template_page.refresh_theme()
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
        # 工作区切换按钮（header 色系）
        self._workspace_btn.setIcon(nav_icon("open_project", pal.nav_text))

    def _refresh_workspace_ui(self) -> None:
        """刷新头部和状态栏的工作区路径显示（只读 self._workspace_manager）."""
        pal = theme.current_palette()
        wm = self._workspace_manager
        path_str = str(wm.cwd)
        # 头部 label：显示目录名 + 父目录（如 "zylab · F:/Dev"），过长用省略
        name = wm.cwd.name or wm.cwd.parent.name  # 根目录兜底
        parent = wm.cwd.parent.name if wm.cwd.parent.name else wm.cwd.parent
        display = f"{name} · {parent}"
        self._workspace_label.setText(display)
        self._workspace_label.setToolTip(path_str)
        self._workspace_label.setStyleSheet(f"color: {pal.nav_text};")
        # 状态栏 widget：始终显示完整 cwd（等宽字体更易读）
        if hasattr(self, "_status_cwd_label"):
            self._status_cwd_label.setText(f"  📁 {path_str}")

    def _on_switch_workspace(self) -> None:
        """弹出目录选择对话框，确认后经 WorkspaceManager 切换工作区."""
        current = str(self._workspace_manager.cwd)
        target = QFileDialog.getExistingDirectory(
            self,
            "选择工作区目录",
            current,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if not target:
            return  # 用户取消
        info = self._workspace_manager.set_workspace(target)
        if info.source == "invalid":
            self.statusBar().showMessage(f"切换失败：目录不存在 — {target}")
            return
        self._workspace_manager.save()
        self.statusBar().showMessage(f"工作区已切换：{info.path}")

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
        """连接导航与跨页信号；订阅工作区变更事件同步 UI；状态栏常驻工作区路径."""
        self._sidebar.currentRowChanged.connect(self._stack.setCurrentIndex)
        self._sidebar.currentRowChanged.connect(lambda _row: self._refresh_sidebar_icons())
        # 笔记本/模板页状态提示统一进主窗口状态栏；模板声明的主题按预览语义应用
        self._notebook_page.status_message.connect(self.statusBar().showMessage)
        self._template_page.status_message.connect(self.statusBar().showMessage)
        self._template_page.theme_requested.connect(lambda name: self._set_theme(name, persist=False))
        # 工作区变更事件 → 头部/状态栏刷新
        self._bus.subscribe(TOPIC_WORKSPACE_CHANGED, self._on_workspace_changed)
        # 状态栏永久 widget：完整路径（左对齐，点击等价工作区切换）
        self._status_cwd_label = QLabel(objectName="statusCwdLabel")
        self._status_cwd_label.setToolTip("当前工作区（MATLAB cwd），点击切换")
        self._status_cwd_label.mouseDoubleClickEvent = lambda _e: self._on_switch_workspace()
        self.statusBar().addPermanentWidget(self._status_cwd_label, 1)
        self._refresh_workspace_ui()

    def _on_workspace_changed(self, info: object) -> None:
        """WorkspaceManager 切路径后刷新头部与状态栏的工作区显示."""
        if isinstance(info, WorkspaceInfo):
            self._refresh_workspace_ui()

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
            Command(
                "go.template",
                "转到：模板",
                lambda: self._sidebar.setCurrentRow(_PAGE_TEMPLATE),
                keywords="goto template dsl",
            )
        )
        register(Command("template.load", "加载 DSL 模板", self._open_template_page, keywords="load template dsl yaml"))
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
        register(
            Command(
                "workspace.switch",
                "切换工作区目录（MATLAB cd 等价）",
                self._on_switch_workspace,
                keywords="workspace cwd cd",
                shortcut="Ctrl+Shift+D",
            )
        )

    def _open_template_page(self) -> None:
        """跳转模板页并直接弹出模板文件选择（命令面板一键加载）."""
        self._sidebar.setCurrentRow(_PAGE_TEMPLATE)
        self._template_page.load_template_file()

    def eventFilter(self, obj, event) -> bool:  # Qt 命名约定
        """点击头部命令搜索框弹出命令面板（只读框仅作入口）."""
        if obj is self._command_search and event.type() == QEvent.MouseButtonPress:
            self._palette.open_commands()
            return True
        return super().eventFilter(obj, event)

    def closeEvent(self, event) -> None:  # Qt 命名约定
        """关闭前询问保存笔记本，持久化工作区路径，终止后台求解执行器."""
        if not self._notebook_page.maybe_save():
            event.ignore()
            return
        self._workspace_manager.save()
        self._studio_page.shutdown()
        super().closeEvent(event)
