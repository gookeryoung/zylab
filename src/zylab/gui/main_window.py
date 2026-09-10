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
    QGridLayout,
    QHBoxLayout,
    QKeySequence,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSize,
    QSplitter,
    QStackedWidget,
    Qt,
    QToolButton,
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
_NAV_ICON_SIZE = QSize(14, 14)


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
        """构建头部条：左侧标题 + 居中 MATLAB 风格工作区地址栏 + 右侧功能搜索框."""
        bar = QFrame(objectName="headerBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(theme.SPACING_SM, 0, theme.SPACING_SM, 0)
        layout.setSpacing(theme.SPACING_XS)

        # 左：应用标题
        layout.addWidget(QLabel("zylab", objectName="headerTitle"), alignment=Qt.AlignVCenter)

        # 中：工作区地址栏（标签 + 路径 + 操作按钮组）
        workspace_bar = QFrame(objectName="workspaceBar")
        ws_layout = QHBoxLayout(workspace_bar)
        ws_layout.setContentsMargins(theme.SPACING_SM, 0, theme.SPACING_XS, 0)
        ws_layout.setSpacing(theme.SPACING_XS)

        # 标签：【工作区】
        ws_tag = QLabel("工作区", objectName="workspaceTag")
        ws_tag.setFixedWidth(40)

        # 竖向分隔线
        ws_sep = QLabel(objectName="workspaceSeparator")

        # 路径地址
        self._workspace_label = QLabel(objectName="workspaceLabel")
        self._workspace_label.setToolTip("当前工作区（MATLAB 风格 cwd）")
        self._workspace_label.setMinimumWidth(100)
        self._workspace_label.setMaximumWidth(400)

        # 弹簧填充：让标签 + 路径居左，按钮组居右
        ws_layout.addWidget(ws_tag)
        ws_layout.addWidget(ws_sep)
        ws_layout.addWidget(self._workspace_label, stretch=1)

        # 操作按钮组（右侧）：下拉历史 + 打开新工作区
        self._workspace_history_btn = QToolButton(objectName="workspaceHistoryBtn")
        self._workspace_history_btn.setToolTip("切换到最近的工作区")
        self._workspace_history_btn.setFixedSize(24, 22)
        self._workspace_history_btn.setIconSize(QSize(10, 10))
        self._workspace_history_menu = QMenu(self)
        self._workspace_history_btn.setMenu(self._workspace_history_menu)
        self._workspace_history_btn.setPopupMode(QToolButton.InstantPopup)

        self._workspace_open_btn = QPushButton(objectName="workspaceOpenBtn")
        self._workspace_open_btn.setToolTip("打开新工作区目录")
        self._workspace_open_btn.setFixedSize(24, 22)
        self._workspace_open_btn.setIconSize(QSize(12, 12))
        self._workspace_open_btn.clicked.connect(self._on_switch_workspace)

        ws_layout.addWidget(self._workspace_history_btn)
        ws_layout.addWidget(self._workspace_open_btn)
        layout.addWidget(workspace_bar, stretch=1, alignment=Qt.AlignVCenter)

        # 右：功能搜索框（VS Code 命令面板入口：点击或 Ctrl+Shift+P 弹出）
        self._command_search = QLineEdit(objectName="commandSearch")
        self._command_search.setReadOnly(True)
        self._command_search.setPlaceholderText("搜索功能 (Ctrl+Shift+P)")
        self._command_search.setFixedWidth(240)
        self._command_search.setFixedHeight(24)
        self._command_search.installEventFilter(self)
        layout.addWidget(self._command_search, alignment=Qt.AlignVCenter)

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
        """按当前主题色重绘侧边栏图标（选中行用强调色）+ 工作区操作按钮."""
        pal = theme.current_palette()
        for row, name in enumerate(NAV_ICON_NAMES):
            item = self._sidebar.item(row)
            if item is not None:
                color = pal.nav_accent if row == self._sidebar.currentRow() else pal.nav_text
                item.setIcon(nav_icon(name, color))

        # 工作区下拉历史按钮（箭头）+ 打开文件夹按钮
        self._workspace_history_btn.setIcon(nav_icon("arrow_down", pal.nav_text))
        self._workspace_open_btn.setIcon(nav_icon("open_file", pal.nav_text))

    def _refresh_workspace_ui(self) -> None:
        """刷新头部和状态栏的工作区路径显示（只读 self._workspace_manager）."""
        wm = self._workspace_manager
        path_str = str(wm.cwd)
        # MATLAB 风格地址栏：显示完整路径（等宽字体自然对齐），过长自动省略
        self._workspace_label.setText(path_str)
        self._workspace_label.setToolTip(f"当前工作区：{path_str}")
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
        self._switch_workspace_to(target)

    def _refresh_and_show_workspace_menu(self) -> None:
        """刷新历史下拉菜单并立即弹出（供按钮点击或 setMenu 自动触发）."""
        self._refresh_workspace_menu()

    def _refresh_workspace_menu(self) -> None:
        """重建历史工作区菜单（最近 10 条，点击即切换）."""
        menu = self._workspace_history_menu
        menu.clear()
        history = self._workspace_manager.recent_workspaces(limit=10)
        if not history:
            # 无历史：显示禁用占位项
            empty = menu.addAction("（暂无历史）")
            empty.setEnabled(False)
            return
        for path in history:
            action = menu.addAction(str(path))
            action.setData(str(path))
            action.triggered.connect(lambda _checked=False, p=str(path): self._switch_workspace_to(p))
        menu.addSeparator()
        open_action = menu.addAction("选择其他目录…")
        open_action.triggered.connect(self._on_switch_workspace)

    def _switch_workspace_to(self, target: str) -> None:
        """切换到指定工作区路径（校验 + 应用 + 持久化 + 提示）."""
        target_path = target
        info = self._workspace_manager.set_workspace(target_path)
        if info.source == "invalid":
            self.statusBar().showMessage(f"切换失败：目录不存在 — {target_path}")
            return
        self._workspace_manager.save()
        self._refresh_workspace_menu()
        self.statusBar().showMessage(f"工作区已切换：{info.path}")

    def _build_about_page(self) -> QWidget:
        """构建关于页：卡片式布局，包含版本/环境/技术栈/许可证完整信息."""
        # 外层滚动区：内容较多时可滚动，保持页面一致
        scroll = QScrollArea(objectName="aboutScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget(objectName="aboutContainer")
        root = QVBoxLayout(container)
        root.setContentsMargins(theme.SPACING_XL, theme.SPACING_XL, theme.SPACING_XL, theme.SPACING_XL)
        root.setSpacing(theme.SPACING_LG)

        # --- 头部：品牌区 ---
        brand = QFrame(objectName="aboutBrand")
        brand_layout = QVBoxLayout(brand)
        brand_layout.setContentsMargins(theme.SPACING_LG, theme.SPACING_LG, theme.SPACING_LG, theme.SPACING_LG)
        brand_layout.setSpacing(theme.SPACING_XS)
        app_name = QLabel("zylab", objectName="aboutAppName")
        app_desc = QLabel("通用科学计算仿真分析平台", objectName="aboutAppDesc")
        app_desc.setWordWrap(True)
        brand_layout.addWidget(app_name)
        brand_layout.addWidget(app_desc)
        root.addWidget(brand)

        # --- 信息卡片网格 ---
        grid = QGridLayout()
        grid.setHorizontalSpacing(theme.SPACING_MD)
        grid.setVerticalSpacing(theme.SPACING_MD)

        grid.addWidget(self._build_info_card("产品版本", f"v{__version__}", None), 0, 0)
        grid.addWidget(self._build_info_card("Python", platform.python_version(), None), 0, 1)
        grid.addWidget(self._build_info_card("Qt 框架", self._qt_version(), None), 1, 0)
        grid.addWidget(self._build_info_card("操作系统", f"{platform.system()} {platform.release()}", None), 1, 1)
        root.addLayout(grid)

        # --- 技术栈卡片 ---
        tech_card = QFrame(objectName="aboutCard")
        tech_layout = QVBoxLayout(tech_card)
        tech_layout.setContentsMargins(theme.SPACING_LG, theme.SPACING_LG, theme.SPACING_LG, theme.SPACING_LG)
        tech_layout.setSpacing(theme.SPACING_SM)
        tech_title = QLabel("技术栈", objectName="aboutCardTitle")
        tech_desc = QLabel(
            "PySide2/PySide6 · NumPy · SciPy · matplotlib · 离线可用的 FEA 求解内核",
            objectName="aboutBody",
        )
        tech_desc.setWordWrap(True)
        tech_layout.addWidget(tech_title)
        tech_layout.addWidget(tech_desc)
        root.addWidget(tech_card)

        # --- 开源信息 ---
        license_card = QFrame(objectName="aboutCard")
        lic_layout = QVBoxLayout(license_card)
        lic_layout.setContentsMargins(theme.SPACING_LG, theme.SPACING_LG, theme.SPACING_LG, theme.SPACING_LG)
        lic_layout.setSpacing(theme.SPACING_SM)
        lic_title = QLabel("开源许可", objectName="aboutCardTitle")
        lic_body = QLabel(
            "zylab 采用 MIT License 开源发布。\n使用 Python 标准库与第三方开源库，各库保留其原始许可。",
            objectName="aboutBody",
        )
        lic_body.setWordWrap(True)
        lic_layout.addWidget(lic_title)
        lic_layout.addWidget(lic_body)
        root.addWidget(license_card)

        root.addStretch()
        scroll.setWidget(container)
        return scroll

    @staticmethod
    def _build_info_card(title: str, value: str, _subtitle: str | None) -> QFrame:
        """构建单条信息卡片（标题 + 值）."""
        card = QFrame(objectName="aboutCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD)
        layout.setSpacing(theme.SPACING_XS)
        t = QLabel(title, objectName="aboutInfoTitle")
        v = QLabel(value, objectName="aboutInfoValue")
        v.setWordWrap(True)
        layout.addWidget(t)
        layout.addWidget(v)
        return card

    @staticmethod
    def _qt_version() -> str:
        """运行时 Qt 版本（PySide6 用 __version__，PySide2 无此属性时退回 qt_version_tag）."""
        try:
            from PySide6.QtCore import __version__  # type: ignore[attr-defined]

            return f"PySide6 {__version__}"
        except (ImportError, AttributeError):
            try:
                import PySide2  # type: ignore[import-not-found]

                return f"PySide2 {getattr(PySide2, '__version__', 'unknown')}"
            except ImportError:
                return "unknown"

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
        self._refresh_workspace_menu()

    def _on_workspace_changed(self, info: object) -> None:
        """WorkspaceManager 切路径后刷新头部工作区显示、状态栏与历史菜单."""
        if isinstance(info, WorkspaceInfo):
            self._refresh_workspace_ui()
            self._refresh_workspace_menu()

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
