"""主窗口：头部条 + 侧边栏导航 + 内容区（QStackedWidget）+ 状态栏."""

from __future__ import annotations

from zylab import __version__
from zylab.console import ReplKernel
from zylab.core import EventBus, default_data_dir
from zylab.sci import TOPIC_WORKSPACE_CHANGED, WorkspaceInfo, WorkspaceManager

from . import theme
from .app import apply_theme, save_theme_name
from .icons import nav_icon
from .pages.flowchart_page import FlowchartPage
from .pages.notebook_page import NotebookPage
from .pages.template_page import TemplatePage
from .qt_compat import (
    QDockWidget,
    QEvent,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QKeySequence,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPushButton,
    QShortcut,
    QSize,
    QStackedWidget,
    Qt,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from .widgets.command_palette import Command, CommandPalette

__all__ = ["MainWindow"]

_PAGE_CONSOLE = 0
_PAGE_FEA = 1
_PAGE_TEMPLATE = 2
_NAV_LABELS = ("笔记本", "流程图", "参数化计算")


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
        self._project_dock_visible = True

        self._build_ui()
        self._load_gui_state()
        self._install_page_shortcuts()
        self._setup_command_palette()
        self._connect()
        self.statusBar().showMessage("就绪")

    @property
    def kernel(self) -> ReplKernel:
        """REPL 内核（测试与外部集成用）."""
        return self._kernel

    def _build_ui(self) -> None:
        """组装 Dock 窗口布局：中央 QStackedWidget + 三个 QDockWidget."""
        # ---- 中央区：Header + QStackedWidget ----
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())

        self._stack = QStackedWidget()
        self._notebook_page = NotebookPage(self._kernel, self._bus)
        self._flowchart_page = FlowchartPage()
        self._template_page = TemplatePage()
        self._stack.addWidget(self._notebook_page)
        self._stack.addWidget(self._flowchart_page)
        self._stack.addWidget(self._template_page)
        root.addWidget(self._stack, stretch=1)
        self.setCentralWidget(central)

        # ---- 左侧 Dock：项目树 ----
        self._project_tree = QTreeWidget(objectName="projectTree")
        # 表头：项目浏览器不需要显式表头，item 左对齐顶满
        self._project_tree.setHeaderHidden(True)
        # 缩进量：Qt 默认 20px 过大，VS Code 风格紧凑树形取 12px
        self._project_tree.setIndentation(12)
        # 层级装饰：branch indicator 由 QSS 自定义（右/下三角）
        self._project_tree.setRootIsDecorated(True)
        # 统一行高：渲染加速，配合 QSS min-height: 22px
        self._project_tree.setUniformRowHeights(True)
        # 展开/折叠：禁用动画，快速切换（大量节点时性能友好）
        self._project_tree.setAnimated(False)
        self._build_default_project_tree()
        self._project_dock = QDockWidget("项目", self)
        self._project_dock.setObjectName("projectDock")
        self._project_dock.setWidget(self._project_tree)
        self._project_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._project_dock)

        # ---- 右侧 Dock：属性 ----
        self._prop_dock = QDockWidget("属性", self)
        self._prop_dock.setObjectName("propertyDock")
        self._prop_dock.setWidget(QWidget(objectName="propertyPlaceholder"))
        self._prop_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, self._prop_dock)

        # ---- 底部 Dock：日志 ----
        from .qt_compat import QPlainTextEdit

        self._log_view = QPlainTextEdit(objectName="logView")
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(500)
        self._log_dock = QDockWidget("日志", self)
        self._log_dock.setObjectName("logDock")
        self._log_dock.setWidget(self._log_view)
        self._log_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._log_dock)

        # 初始 Dock 可见性
        self._project_dock.setVisible(self._project_dock_visible)
        self._prop_dock.setVisible(False)  # 默认隐藏，有选中项时再打开
        self._log_dock.setVisible(False)

    def _build_default_project_tree(self) -> None:
        """构建默认项目树根节点（工作区目录扫描 + 页面快捷入口）."""
        from .icons import NAV_ICON_NAMES

        pal = theme.current_palette()
        # 图标尺寸：与 QTreeWidget 行高 22px 匹配，留 2px 间隙
        self._project_tree.setIconSize(QSize(16, 16))

        root = QTreeWidgetItem(self._project_tree, ["当前工作区"])
        root.setData(0, Qt.UserRole, ("workspace", ""))
        root.setExpanded(True)

        # 页面快捷入口（三种类型各有专属图标，区分视觉语义）
        for idx, label in enumerate(_NAV_LABELS):
            item = QTreeWidgetItem(root, [label])
            item.setData(0, Qt.UserRole, ("page", idx))
            # NAV_ICON_NAMES 与 _NAV_LABELS 顺序一致：notebook / analysis / template
            icon = nav_icon(NAV_ICON_NAMES[idx], pal.text_primary)
            item.setIcon(0, icon)

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
        self._refresh_project_tree_icons()
        self._notebook_page.refresh_theme()
        self._flowchart_page.refresh_theme()
        self._template_page.refresh_theme()
        if persist:
            save_theme_name(default_data_dir(), name)
            self.statusBar().showMessage(f"主题已切换: {theme.current_palette().display_name}")

    def _refresh_sidebar_icons(self) -> None:
        """按当前主题色重绘头部工作区操作按钮（侧边栏已改为 Dock）."""
        pal = theme.current_palette()
        self._workspace_history_btn.setIcon(nav_icon("arrow_down", pal.nav_text))
        self._workspace_open_btn.setIcon(nav_icon("open_file", pal.nav_text))

    def _refresh_project_tree_icons(self) -> None:
        """按当前主题色重绘项目浏览器树节点图标（主题切换时联动）."""
        from .icons import NAV_ICON_NAMES
        from .qt_compat import Qt

        pal = theme.current_palette()
        root = self._project_tree.topLevelItem(0)
        if root is None:
            return
        for idx in range(root.childCount()):
            child = root.child(idx)
            if child is None:
                continue
            data = child.data(0, Qt.UserRole)
            if isinstance(data, tuple) and data[0] == "page":
                page_idx = int(data[1])
                if 0 <= page_idx < len(NAV_ICON_NAMES):
                    child.setIcon(0, nav_icon(NAV_ICON_NAMES[page_idx], pal.text_primary))

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
        """连接导航与跨页信号；订阅工作区变更事件同步 UI；状态栏常驻工作区路径与运行状态."""

        self._project_tree.itemDoubleClicked.connect(self._on_project_tree_double_clicked)
        # 笔记本/参数化计算页状态提示统一进主窗口状态栏；参数化计算声明的主题按预览语义应用
        self._notebook_page.status_message.connect(self.statusBar().showMessage)
        self._template_page.status_message.connect(self.statusBar().showMessage)
        self._flowchart_page.status_message.connect(self.statusBar().showMessage)
        self._template_page.theme_requested.connect(lambda name: self._set_theme(name, persist=False))
        # 参数化计算运行生命周期 → 主窗口右下 indicator（加载/运行/完成/失败统一承载）
        self._template_page.run_state_changed.connect(self.set_run_status)
        # 流程图运行成功/失败 → 主窗口右下 indicator
        self._flowchart_page.run_status_changed.connect(self.set_run_status)
        # 工作区变更事件 → 头部/状态栏刷新
        self._bus.subscribe(TOPIC_WORKSPACE_CHANGED, self._on_workspace_changed)
        # 状态栏永久 widget：完整路径（左对齐，双击切换）
        self._status_cwd_label = QLabel(objectName="statusCwdLabel")
        self._status_cwd_label.setToolTip("当前工作区（MATLAB cwd），双击切换")
        self._status_cwd_label.mouseDoubleClickEvent = lambda event: self._on_switch_workspace()  # noqa: ARG005  参数名须匹配 PySide stub
        self.statusBar().addPermanentWidget(self._status_cwd_label, 1)
        # 状态栏永久 widget：运行状态 indicator（右对齐，icon + 颜色 + 文字）
        # PySide2 QLabel 无 setIcon，用 pixmap + text 两个 QLabel 组合
        pal = theme.current_palette()
        self._run_indicator_widget = QWidget(objectName="runStatusIndicator")
        indicator_layout = QHBoxLayout(self._run_indicator_widget)
        indicator_layout.setContentsMargins(12, 0, 12, 0)
        indicator_layout.setSpacing(4)
        self._indicator_icon = QLabel()
        self._indicator_icon.setFixedSize(16, 16)
        self._indicator_text = QLabel("就绪")
        self._indicator_text.setAlignment(Qt.AlignCenter)
        self._indicator_text.setStyleSheet(f"color: {pal.text_secondary};")
        self._run_indicator_widget.setToolTip("运行状态（流程图/参数化计算运行完成后在此统一显示）")
        indicator_layout.addWidget(self._indicator_icon)
        indicator_layout.addWidget(self._indicator_text)
        self.set_run_status("idle")  # 初始化 icon + 颜色
        self.statusBar().addPermanentWidget(self._run_indicator_widget, 0)
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

    def _open_settings_dialog(self) -> None:
        """弹出设置对话框（SettingsPanel 嵌入 QDialog）.

        保存时应用主题等副作用：持久化 → set_theme 切换 → 状态栏提示。
        """
        from .qt_compat import QDialog
        from .widgets.settings_panel import SettingsPanel

        dlg = QDialog(self)
        dlg.setWindowTitle("设置")
        dlg.setMinimumSize(480, 360)

        panel = SettingsPanel()
        from .qt_compat import QVBoxLayout

        root = QVBoxLayout(dlg)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(panel)

        import contextlib

        def _on_save() -> None:
            cfg = panel.save()
            theme_name = cfg.get("theme")
            if theme_name and theme_name != theme.current_palette().name:
                self._set_theme(theme_name, persist=True)
            self.statusBar().showMessage("设置已保存", 3000)
            dlg.accept()

        with contextlib.suppress(RuntimeError, TypeError):
            panel._save_btn.clicked.disconnect()
        panel._save_btn.clicked.connect(_on_save)
        with contextlib.suppress(RuntimeError, TypeError):
            panel._cancel_btn.clicked.disconnect()
        panel._cancel_btn.clicked.connect(dlg.reject)

        dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec()

    def _install_page_shortcuts(self) -> None:
        """页面级快捷键：Ctrl+1/2/3 直达，Ctrl+PgDn/PgUp 循环切换."""
        from .qt_compat import QKeySequence

        bindings = (
            ("Ctrl+1", lambda: self._stack.setCurrentIndex(_PAGE_CONSOLE)),
            ("Ctrl+2", lambda: self._stack.setCurrentIndex(_PAGE_FEA)),
            ("Ctrl+3", lambda: self._stack.setCurrentIndex(_PAGE_TEMPLATE)),
            ("Ctrl+PageDown", self._cycle_next_page),
            ("Ctrl+PageUp", self._cycle_prev_page),
            ("Ctrl+B", self._toggle_project_dock),
            ("F5", self._global_run),
        )
        for key, handler in bindings:
            QShortcut(QKeySequence(key), self, activated=handler)  # type: ignore[arg-type]

    def _cycle_next_page(self) -> None:
        """Ctrl+PgDn：切换到下一页（循环回到首页）."""
        current = self._stack.currentIndex()
        total = self._stack.count()
        self._stack.setCurrentIndex((current + 1) % total)

    def _cycle_prev_page(self) -> None:
        """Ctrl+PgUp：切换到上一页（循环到末页）."""
        current = self._stack.currentIndex()
        total = self._stack.count()
        self._stack.setCurrentIndex((current - 1) % total)

    def _global_run(self) -> None:
        """F5 全局运行：按当前激活页分发到对应 run 方法.

        - 笔记本页：run_all()（顺序执行全部单元）
        - 流程图页：暂不支持直接运行
        - 参数化计算页：run()（运行当前 DSL 参数化计算）
        """
        row = self._stack.currentIndex()
        if row == _PAGE_CONSOLE:
            self.statusBar().showMessage("笔记本：运行全部单元（F5）…")
            self._notebook_page.run_all()
        elif row == _PAGE_FEA:
            self.statusBar().showMessage("流程图：请通过参数化计算或笔记本 F5 触发运行")
        elif row == _PAGE_TEMPLATE:
            self.statusBar().showMessage("参数化计算：运行当前 DSL（F5）…")
            self._template_page.run()

    def _register_commands(self) -> None:
        """注册全局命令（页面导航 / 笔记本操作 / 主题切换）."""
        page = self._notebook_page
        register = self._palette.register
        register(
            Command(
                "go.notebook",
                "转到：笔记本",
                lambda: self._stack.setCurrentIndex(_PAGE_CONSOLE),
                keywords="goto notebook",
            )
        )
        register(
            Command(
                "go.analysis",
                "转到：流程图",
                lambda: self._stack.setCurrentIndex(_PAGE_FEA),
                keywords="goto analysis fea",
            )
        )
        register(
            Command(
                "go.template",
                "转到：参数化计算",
                lambda: self._stack.setCurrentIndex(_PAGE_TEMPLATE),
                keywords="goto template dsl",
            )
        )
        register(
            Command("template.load", "加载 DSL 参数化计算", self._open_template_page, keywords="load template dsl yaml")
        )
        register(Command("go.about", "关于 zylab", self._open_about_dialog, keywords="goto about help"))
        register(Command("go.settings", "设置", self._open_settings_dialog, keywords="goto settings preferences"))
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
        """跳转参数化计算页并直接弹出参数化计算文件选择（命令面板一键加载）."""
        self._stack.setCurrentIndex(_PAGE_TEMPLATE)
        self._template_page.load_template_file()

    def eventFilter(self, obj, event) -> bool:  # Qt 命名约定
        """点击头部命令搜索框弹出命令面板（只读框仅作入口）."""
        if obj is self._command_search and event.type() == QEvent.MouseButtonPress:
            self._palette.open_commands()
            return True
        return super().eventFilter(obj, event)

    def _on_project_tree_double_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        """项目树双击：page 类型切页，workspace 类型后续扩展."""
        data = item.data(0, Qt.UserRole)
        if isinstance(data, tuple) and data[0] == "page":
            self._stack.setCurrentIndex(int(data[1]))

    def _toggle_project_dock(self) -> None:
        """切换项目树 Dock 可见性（Ctrl+B）."""
        self._project_dock_visible = not self._project_dock_visible
        self._project_dock.setVisible(self._project_dock_visible)
        self._save_gui_state()

    def _load_gui_state(self) -> None:
        """加载 gui_state.json（项目树 Dock 可见性等）."""
        import json

        try:
            path = default_data_dir() / "gui_state.json"
            if path.is_file():
                state = json.loads(path.read_text(encoding="utf-8"))
                self._project_dock_visible = bool(state.get("project_dock_visible", True))
        except (OSError, ValueError):
            pass  # 文件不存在或损坏，忽略

    def _save_gui_state(self) -> None:
        """保存 gui_state.json."""
        import json

        try:
            path = default_data_dir() / "gui_state.json"
            path.write_text(
                json.dumps({"project_dock_visible": self._project_dock_visible}, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    def resizeEvent(self, event) -> None:  # Qt 命名约定
        """窗口宽度 <1000px 自动隐藏项目 Dock（窄屏响应）."""
        super().resizeEvent(event)
        if self.width() < 1000 and self._project_dock_visible:
            self._project_dock_visible = False
            self._project_dock.setVisible(False)
            self._save_gui_state()

    # ------------------------------------------------------------------ 运行状态 indicator

    def set_run_status(self, state: str, detail: str = "") -> None:
        """设置右下角运行状态 indicator（流程图/参数化计算运行完成后由主窗口统一呈现）.

        :param state: "idle"（就绪）/ "running"（运行中）/ "success"（成功）/ "error"（失败）.
        :param detail: 失败时的错误详情（tooltip 承载，不超过 200 字）。
        """
        from .icons import tinted_pixmap

        pal = theme.current_palette()
        state_map = {
            "idle": ("question", "就绪", pal.text_secondary),
            "running": ("play", "计算中…", pal.primary),
            "success": ("check", "运行完成", pal.success_text),
            "error": ("cross", "运行失败", pal.danger_text),
        }
        icon_name, label, color = state_map.get(state, state_map["idle"])
        # PySide2 QLabel 无 setIcon，用 tinted_pixmap 渲染为 QPixmap 再 setPixmap
        self._indicator_icon.setPixmap(tinted_pixmap(icon_name, color, 16))
        self._indicator_text.setText(label)
        self._indicator_text.setStyleSheet(f"color: {color};")
        if detail and state == "error":
            self._run_indicator_widget.setToolTip(f"运行失败：{detail[:200]}")
        elif detail:
            self._run_indicator_widget.setToolTip(detail[:200])
        else:
            self._run_indicator_widget.setToolTip("运行状态（流程图/参数化计算运行完成后在此统一显示）")
