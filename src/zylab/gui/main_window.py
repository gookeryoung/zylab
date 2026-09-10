"""主窗口：头部条 + 侧边栏导航 + 内容区（QStackedWidget）+ 状态栏."""

from __future__ import annotations

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
    QMenu,
    QPushButton,
    QShortcut,
    QSize,
    QSizePolicy,
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
_NAV_LABELS = ("笔记本", "工作台", "计算模板")

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

        # 侧边栏折叠状态（默认展开；恢复上次会话）
        self._sidebar_folded = False

        self._build_ui()
        self._load_gui_state()
        self._apply_sidebar_folded()
        self._install_page_shortcuts()
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

        self._splitter = QSplitter(Qt.Horizontal)
        self._sidebar = QListWidget(objectName="sidebar")
        for label in _NAV_LABELS:
            QListWidgetItem(label, self._sidebar)
        self._sidebar.setIconSize(_NAV_ICON_SIZE)
        self._sidebar.setFixedWidth(theme.SIDEBAR_WIDTH)
        # QListWidget 默认垂直 Expanding 会强制填充整个容器，导致 item 下方大片空白
        self._sidebar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        # 侧边栏折叠手柄（包在容器底部）
        from .qt_compat import QPushButton as _QPB

        self._sidebar_handle = _QPB(objectName="sidebarHandle")
        self._sidebar_handle.setFixedHeight(36)  # 与导航 item 同高（QListWidget#sidebar::item height: 36px）
        self._sidebar_handle.setCursor(Qt.PointingHandCursor)
        self._sidebar_handle.setToolTip("折叠/展开侧边栏 (Ctrl+B)")
        self._sidebar_handle.setText("«")
        self._sidebar_handle.clicked.connect(self._toggle_sidebar)
        self._sidebar_container = QWidget(objectName="sidebarContainer")
        _sb_layout = QVBoxLayout(self._sidebar_container)
        _sb_layout.setContentsMargins(0, 0, 0, 0)
        _sb_layout.setSpacing(0)
        _sb_layout.addWidget(self._sidebar)
        _sb_layout.addStretch()  # 弹簧填充导航项与手柄之间的空白
        _sb_layout.addWidget(self._sidebar_handle)
        self._sidebar.setCurrentRow(_PAGE_CONSOLE)
        self._refresh_sidebar_icons()

        self._stack = QStackedWidget()
        self._notebook_page = NotebookPage(self._kernel, self._bus)
        self._studio_page = StudioPage()
        self._template_page = TemplatePage()
        self._stack.addWidget(self._notebook_page)
        self._stack.addWidget(self._studio_page)
        self._stack.addWidget(self._template_page)

        self._splitter.addWidget(self._sidebar_container)
        self._splitter.addWidget(self._stack)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([theme.SIDEBAR_WIDTH, 1080])
        root.addWidget(self._splitter, stretch=1)
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

        # 右：帮助按钮（关于 zylab，降级自侧边栏导航项）
        self._help_btn = QPushButton(objectName="headerHelpBtn")
        self._help_btn.setToolTip("关于 zylab")
        self._help_btn.setFixedSize(24, 24)
        self._help_btn.setIconSize(QSize(12, 12))
        self._help_btn.setIcon(nav_icon("question", theme.current_palette().nav_text))
        self._help_btn.clicked.connect(self._open_about_dialog)
        layout.addWidget(self._help_btn, alignment=Qt.AlignVCenter)

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

        # 手柄文字同步折叠状态
        self._sidebar_handle.setText("»" if self._sidebar_folded else "«")
        self._sidebar_handle.setToolTip("展开侧边栏 (Ctrl+B)" if self._sidebar_folded else "折叠侧边栏 (Ctrl+B)")

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
        self._save_gui_state()
        self._refresh_workspace_menu()
        self.statusBar().showMessage(f"工作区已切换：{info.path}")

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

    def _open_about_dialog(self) -> None:
        """弹出关于对话框（独立 QDialog，而非侧边栏页面）."""
        from .qt_compat import QDialog

        dlg = QDialog(self)
        dlg.setWindowTitle("关于 zylab")
        dlg.setMinimumWidth(420)
        root = QVBoxLayout(dlg)
        root.setContentsMargins(theme.SPACING_LG, theme.SPACING_LG, theme.SPACING_LG, theme.SPACING_LG)
        root.setSpacing(theme.SPACING_MD)

        brand = QLabel("zylab", objectName="aboutAppName")
        desc = QLabel("通用科学计算仿真分析平台", objectName="aboutAppDesc")
        desc.setWordWrap(True)
        version_label = QLabel(f"版本 v{__version__}")
        tech = QLabel("技术栈：PySide2/PySide6 · NumPy · SciPy · matplotlib · 离线 FEA 求解内核")
        tech.setWordWrap(True)
        lic = QLabel("开源许可：MIT License。\nzylab 采用 MIT License 开源发布，使用 Python 标准库与第三方开源库。")
        lic.setWordWrap(True)

        for w in (brand, desc, version_label, tech, lic):
            root.addWidget(w)
        root.addStretch()

        dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec()

    def _install_page_shortcuts(self) -> None:
        """页面级快捷键：Ctrl+1/2/3 直达，Ctrl+PgDn/PgUp 循环切换."""
        from .qt_compat import QKeySequence

        bindings = (
            ("Ctrl+1", lambda: self._sidebar.setCurrentRow(_PAGE_CONSOLE)),
            ("Ctrl+2", lambda: self._sidebar.setCurrentRow(_PAGE_FEA)),
            ("Ctrl+3", lambda: self._sidebar.setCurrentRow(_PAGE_TEMPLATE)),
            ("Ctrl+PageDown", self._cycle_next_page),
            ("Ctrl+PageUp", self._cycle_prev_page),
            ("Ctrl+B", self._toggle_sidebar),
            ("F5", self._global_run),
        )
        for key, handler in bindings:
            QShortcut(QKeySequence(key), self, activated=handler)  # type: ignore[arg-type]

    def _cycle_next_page(self) -> None:
        """Ctrl+PgDn：切换到下一页（循环回到首页）."""
        current = self._sidebar.currentRow()
        total = self._sidebar.count()
        self._sidebar.setCurrentRow((current + 1) % total)

    def _cycle_prev_page(self) -> None:
        """Ctrl+PgUp：切换到上一页（循环到末页）."""
        current = self._sidebar.currentRow()
        total = self._sidebar.count()
        self._sidebar.setCurrentRow((current - 1) % total)

    def _global_run(self) -> None:
        """F5 全局运行：按当前激活页分发到对应 run 方法.

        - 笔记本页：run_all()（顺序执行全部单元）
        - 工作台页：暂不支持直接运行
        - 模板页：run()（运行当前 DSL 模板）
        """
        row = self._sidebar.currentRow()
        if row == _PAGE_CONSOLE:
            self.statusBar().showMessage("笔记本：运行全部单元（F5）…")
            self._notebook_page.run_all()
        elif row == _PAGE_FEA:
            self.statusBar().showMessage("工作台：请通过模板或笔记本 F5 触发运行")
        elif row == _PAGE_TEMPLATE:
            self.statusBar().showMessage("计算模板：运行当前 DSL（F5）…")
            self._template_page.run()

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
                "转到：工作台",
                lambda: self._sidebar.setCurrentRow(_PAGE_FEA),
                keywords="goto analysis fea",
            )
        )
        register(
            Command(
                "go.template",
                "转到：计算模板",
                lambda: self._sidebar.setCurrentRow(_PAGE_TEMPLATE),
                keywords="goto template dsl",
            )
        )
        register(Command("template.load", "加载 DSL 模板", self._open_template_page, keywords="load template dsl yaml"))
        register(Command("go.about", "关于 zylab", self._open_about_dialog, keywords="goto about help"))
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
        register(
            Command(
                "run.global",
                "F5：运行（按当前页自动分发）",
                self._global_run,
                keywords="run execute f5 运行",
                shortcut="F5",
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

    def _toggle_sidebar(self) -> None:
        """切换侧边栏折叠状态."""
        self._sidebar_folded = not self._sidebar_folded
        self._apply_sidebar_folded()

    def _apply_sidebar_folded(self) -> None:
        """应用折叠状态：改变宽度 + 隐藏/显示导航项文字 + 刷新手柄."""
        theme.current_palette()
        if self._sidebar_folded:
            self._sidebar.setFixedWidth(48)
            self._sidebar_container.setFixedWidth(48)
            for row in range(self._sidebar.count()):
                item = self._sidebar.item(row)
                item.setText("")
                item.setToolTip(item.toolTip() if item.toolTip() else _NAV_LABELS[row])
        else:
            self._sidebar.setFixedWidth(theme.SIDEBAR_WIDTH)
            self._sidebar_container.setFixedWidth(theme.SIDEBAR_WIDTH)
            labels = _NAV_LABELS
            for row in range(self._sidebar.count()):
                item = self._sidebar.item(row)
                item.setText(labels[row])
                item.setToolTip("")
        self._refresh_sidebar_icons()
        w = self._splitter.width()
        h = self._splitter.handleWidth()
        sw = self._sidebar_container.width()
        self._splitter.setSizes([sw, w - sw - h])

    def _load_gui_state(self) -> None:
        """加载 gui_state.json（侧边栏折叠、窗口几何等）."""
        import json

        try:
            path = default_data_dir() / "gui_state.json"
            if path.is_file():
                state = json.loads(path.read_text(encoding="utf-8"))
                self._sidebar_folded = bool(state.get("sidebar_folded", False))
        except (OSError, ValueError):
            pass  # 文件不存在或损坏，忽略

    def _save_gui_state(self) -> None:
        """保存 gui_state.json."""
        import json

        try:
            path = default_data_dir() / "gui_state.json"
            path.write_text(
                json.dumps({"sidebar_folded": self._sidebar_folded}, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    def resizeEvent(self, event) -> None:  # Qt 命名约定
        """窗口宽度 <1000px 自动折叠侧边栏（窄屏响应）."""
        super().resizeEvent(event)
        if self.width() < 1000 and not self._sidebar_folded:
            self._sidebar_folded = True
            self._apply_sidebar_folded()

    def closeEvent(self, event) -> None:  # Qt 命名约定
        """关闭前询问保存笔记本，持久化工作区路径，终止后台求解执行器."""
        if not self._notebook_page.maybe_save():
            event.ignore()
            return
        self._workspace_manager.save()
        self._save_gui_state()
        self._studio_page.shutdown()
        super().closeEvent(event)
