"""笔记本页：Jupyter 风格单元流（代码 + inline 输出 + 内嵌绘图）+ 可折叠变量侧栏.

- :class:`CellEditor`：单元代码编辑器（Ctrl+Enter 运行本格、Shift+Enter
  运行并推进下格、Tab 插入 4 空格缩进）；
- :class:`CellWidget`：单格卡片（``In [n]:`` 序号栏 + 编辑器 + 输出区 +
  悬停浮现的单元级工具条：运行/上移/下移/插入/删除），
  输出按类型分发：流文本/结果 repr/错误回溯/pyqtgraph 内嵌绘图；
- :class:`NotebookPage`：顶部文档级工具栏（新建/打开/保存/全部运行/重启内核/
  变量面板切换）+ 单元流 + 可折叠变量侧栏（默认收起，展开 4:1）；
  ``.znbk`` 打开/保存/另存、dirty 跟踪与关闭前保存询问。
"""

from __future__ import annotations

import html
from pathlib import Path

import pyqtgraph as pg

from zylab.console import ReplKernel
from zylab.core import EventBus
from zylab.sci import (
    ErrorOutput,
    Notebook,
    NotebookCell,
    NotebookError,
    PlotOutput,
    ResultOutput,
    StreamOutput,
    load_notebook,
    new_cell,
    save_notebook,
    whos,
)

from .. import theme
from ..highlight import PythonHighlighter
from ..icons import nav_icon
from ..qt_compat import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QKeyEvent,
    QKeySequence,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSize,
    QSplitter,
    Qt,
    QTableView,
    QVBoxLayout,
    QWidget,
    Signal,
    exec_dialog,
)
from .var_browser import VarDetailDialog, VarTableModel, VarTagDelegate, mono_font

__all__ = ["CellEditor", "CellWidget", "NotebookPage", "VarTableModel"]  # VarTableModel 经 var_browser re-export

# 新笔记本默认首格（演示单元执行与内嵌绘图）
_WELCOME_SOURCE = """\
# zylab 笔记本：Ctrl+Enter 运行本格，Shift+Enter 运行并新建下格
# Alt+Enter 下方插入，Ctrl+Shift+Enter 从此运行，工具栏可切换变量面板
x = linspace(0, 4 * pi, 200)
y = sin(x) * exp(-x / 10)
plot(x, y, title="衰减振荡", xlabel="x", ylabel="y")
y[:5]
"""

#: 绘图多曲线着色循环（主题色板取色）
_CURVE_KEYS = ("primary", "success_text", "warning_text", "danger_text", "border_strong")

#: 单元级工具条按钮提示文案（按动作名）
_TOOL_TIPS = {
    "run": "运行本格 (Ctrl+Enter)",
    "up": "上移 (Ctrl+Shift+Up)",
    "down": "下移 (Ctrl+Shift+Down)",
    "insert": "下方插入 (Alt+Enter)",
    "delete": "删除本格",
}


class CellEditor(QPlainTextEdit):
    """单元代码编辑器（Ctrl+Enter 运行 / Shift+Enter 运行并推进 / Tab 缩进）."""

    #: 参数 advance：True = 运行后推进下格（Shift+Enter）
    run_requested = Signal(bool)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # Qt 命名约定
        """按键分发：运行快捷键拦截、Tab 插入 4 空格、其余默认."""
        key = event.key()
        if key in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ControlModifier:
                self.run_requested.emit(False)
                return
            if event.modifiers() & Qt.ShiftModifier:
                self.run_requested.emit(True)
                return
        if key == Qt.Key_Tab and not self.textCursor().hasSelection():
            self.insertPlainText("    ")
            return
        super().keyPressEvent(event)


class CellWidget(QFrame):
    """单格卡片：序号栏 + 代码编辑器 + 输出区 + 悬停浮现的单元级工具条."""

    #: 单元级动作请求（"run"/"up"/"down"/"insert"/"delete"，页面解析执行）
    action_requested = Signal(str)

    #: 工具条按钮（图标基名, 动作名）——悬停卡片时浮现
    _TOOL_ITEMS = (("play", "run"), ("arrow_up", "up"), ("arrow_down", "down"), ("plus", "insert"), ("trash", "delete"))

    def __init__(self, cell: NotebookCell, parent: QWidget | None = None) -> None:
        """初始化卡片并装载单元源码与既有输出.

        Args:
            cell: 绑定的笔记本单元（编辑与运行双向往 cell 同步）。
        """
        super().__init__(objectName="notebookCell", parent=parent)
        self._cell = cell
        self._editor = CellEditor(objectName="cellEditor")
        self._editor.setPlainText(cell.source)
        self._editor.setFont(mono_font())
        self._highlighter = PythonHighlighter(self._editor.document())
        self._editor.textChanged.connect(self._on_text_changed)
        self._output_host = QWidget()
        self._output_layout = QVBoxLayout(self._output_host)
        self._output_layout.setContentsMargins(0, 0, 0, 0)
        self._output_layout.setSpacing(theme.SPACING_XS)

        body = QVBoxLayout()
        body.setContentsMargins(theme.SPACING_SM, theme.SPACING_SM, theme.SPACING_SM, theme.SPACING_SM)
        body.setSpacing(theme.SPACING_XS)
        body.addWidget(self._editor)
        body.addWidget(self._output_host)

        self._count_label = QLabel(objectName="cellCount")
        self._count_label.setFont(mono_font())
        self._count_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self._count_label.setFixedWidth(64)
        # 顶部偏移对齐 cellEditor 边框(1px)+内边距(8px)，使 In[n] 与代码首行基线一致
        self._count_label.setContentsMargins(0, 9, 0, 0)

        # 单元级工具条：固定占位宽度，悬停时按钮可见（不挤压格宽）
        self._tool_buttons: list[QPushButton] = []
        toolbar = QWidget(objectName="cellToolbar")
        toolbar.setFixedWidth(30)
        tool_layout = QVBoxLayout(toolbar)
        tool_layout.setContentsMargins(0, 0, 0, 0)
        tool_layout.setSpacing(2)
        for _icon_name, action in self._TOOL_ITEMS:
            button = QPushButton(objectName="cellToolButton")
            button.setFixedSize(26, 22)
            button.setIconSize(QSize(12, 12))
            button.setToolTip(_TOOL_TIPS[action])
            button.setVisible(False)
            button.clicked.connect(lambda _checked=False, name=action: self.action_requested.emit(name))
            tool_layout.addWidget(button)
            self._tool_buttons.append(button)
        tool_layout.addStretch(1)
        self.refresh_icons()

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACING_SM)
        row.addWidget(self._count_label)
        row.addLayout(body, stretch=1)
        row.addWidget(toolbar)

        self._editor.setMinimumHeight(56)
        self._update_count_label()
        self.render_outputs()

    def refresh_icons(self) -> None:
        """按当前主题重绘单元级工具条图标（删除动作用危险色区分）."""
        pal = theme.current_palette()
        for (icon_name, _action), button in zip(self._TOOL_ITEMS, self._tool_buttons):
            color = pal.danger_text if icon_name == "trash" else pal.text_secondary
            button.setIcon(nav_icon(icon_name, color))

    def enterEvent(self, event) -> None:  # Qt 命名约定
        """鼠标进入卡片：浮现单元级工具条."""
        for button in self._tool_buttons:
            button.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # Qt 命名约定
        """鼠标离开卡片：隐藏单元级工具条."""
        for button in self._tool_buttons:
            button.setVisible(False)
        super().leaveEvent(event)

    @property
    def cell(self) -> NotebookCell:
        """绑定的笔记本单元."""
        return self._cell

    @property
    def editor(self) -> CellEditor:
        """代码编辑器（页面接焦/安装运行信号用）."""
        return self._editor

    @property
    def highlighter(self) -> PythonHighlighter:
        """编辑器语法高亮器."""
        return self._highlighter

    def sync_to_model(self) -> None:
        """编辑器文本回写单元模型（运行/保存前调用）."""
        self._cell.source = self._editor.toPlainText()

    def reload_from_model(self) -> None:
        """单元模型回填编辑器（外部加载/结构变化后重建视图）."""
        self._editor.setPlainText(self._cell.source)
        self.render_outputs()

    def render_outputs(self) -> None:
        """按 cell.outputs 重建输出区（流/结果/错误/绘图分发渲染）."""
        self._clear_outputs()
        for out in self._cell.outputs:
            if isinstance(out, StreamOutput):
                self._add_stream(out)
            elif isinstance(out, ResultOutput):
                self._add_result(out)
            elif isinstance(out, ErrorOutput):
                self._add_error(out)
            else:
                self._add_plot(out)
        self._update_count_label()

    def focus_editor(self) -> None:
        """聚焦编辑器（运行推进/格切换）."""
        self._editor.setFocus()
        self._editor.moveCursor(self._editor.textCursor().End)

    def _on_text_changed(self) -> None:
        """编辑即置未保存状态并清本格输出（旧结果失效）."""
        self._cell.source = self._editor.toPlainText()
        if self._cell.outputs or self._cell.execution_count is not None:
            self._cell.outputs.clear()
            self._cell.execution_count = None
            self._update_count_label()

    def _update_count_label(self) -> None:
        """刷新 ``In [n]:`` 序号栏（未运行为 ``In [ ]:``；已执行态主色高亮）."""
        count = self._cell.execution_count
        executed = count is not None
        text = f"In [{count}]:" if executed else "In [ ]:"
        if self._count_label.text() != text:
            self._count_label.setText(text)
        # 动态属性供 QSS 属性选择器着色，变更后需重 polish 生效
        if self._count_label.property("executed") != executed:
            self._count_label.setProperty("executed", executed)
            style = self._count_label.style()
            style.unpolish(self._count_label)
            style.polish(self._count_label)

    def _clear_outputs(self) -> None:
        """销毁输出区全部子控件（重建式渲染前置）."""
        while True:
            item = self._output_layout.takeAt(0)
            if item is None:
                break
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _add_stream(self, out: StreamOutput) -> None:
        """流输出：等宽文本，stderr 用危险色."""
        pal = theme.current_palette()
        color = pal.danger_text if out.name == "stderr" else pal.text_primary
        label = QLabel(objectName="cellStream")
        label.setFont(mono_font())
        label.setTextFormat(Qt.RichText)
        label.setWordWrap(True)
        label.setText(f'<pre style="margin:0">{html.escape(out.text.rstrip())}</pre>')
        label.setStyleSheet(f"color: {color}; padding: 4px 8px; background-color: {pal.bg_input}; border-radius: 4px;")
        self._output_layout.addWidget(label)

    def _add_result(self, out: ResultOutput) -> None:
        """表达式结果：Out[n] 前缀 + repr + 类型/形状摘要."""
        pal = theme.current_palette()
        label = QLabel(objectName="cellResult")
        label.setFont(mono_font())
        label.setTextFormat(Qt.RichText)
        label.setWordWrap(True)
        summary = f" · {out.type_name} {out.shape}" if out.type_name else ""
        label.setText(
            f'<span style="color: {pal.text_secondary}">Out[{self._cell.execution_count}]:</span> '
            f'<pre style="margin:0; color: {pal.success_text}">{html.escape(out.repr_text)}</pre>'
            f'<span style="color: {pal.text_secondary}; font-size: 11px">{html.escape(summary)}</span>'
        )
        label.setStyleSheet(f"padding: 4px 8px; background-color: {pal.bg_input}; border-radius: 4px;")
        self._output_layout.addWidget(label)

    def _add_error(self, out: ErrorOutput) -> None:
        """错误输出：ename 粗体 + 回溯等宽文本（危险色）."""
        pal = theme.current_palette()
        label = QLabel(objectName="cellError")
        label.setFont(mono_font())
        label.setTextFormat(Qt.RichText)
        label.setWordWrap(True)
        label.setText(
            f'<pre style="margin:0; color: {pal.danger_text}">'
            f"<b>{html.escape(out.ename)}</b>\n{html.escape(out.traceback_text.rstrip())}</pre>"
        )
        label.setStyleSheet(f"padding: 4px 8px; background-color: {pal.bg_input}; border-radius: 4px;")
        self._output_layout.addWidget(label)

    def _add_plot(self, out: PlotOutput) -> None:
        """绘图输出：内嵌 pyqtgraph（多 series 同图、主题色循环、轴色随主题）."""
        pal = theme.current_palette()
        plot = pg.PlotWidget(background=pal.bg_app, parent=self)
        plot.showGrid(x=True, y=True, alpha=0.3)
        # 轴线/刻度/文字随主题前景色（深色主题下默认黑不可见）
        axis_pen = pg.mkPen(pal.border_strong, width=1)
        for axis_name in ("bottom", "left"):
            axis = plot.getAxis(axis_name)
            axis.setPen(axis_pen)
            axis.setTextPen(pg.mkPen(pal.text_primary, width=1))
        legend = plot.addLegend(offset=(8, 8)) if any(s.label for s in out.series) else None
        if legend is not None:
            legend.setLabelTextColor(pal.text_primary)
        plot.setMinimumHeight(260)
        for index, series in enumerate(out.series):
            color = getattr(pal, _CURVE_KEYS[index % len(_CURVE_KEYS)])
            plot.plot(
                series.x,
                series.y,
                pen=pg.mkPen(color, width=2),
                name=series.label or None,
            )
        if out.title:
            plot.setTitle(out.title, color=pal.text_primary)
        if out.xlabel:
            plot.setLabel("bottom", out.xlabel)
        if out.ylabel:
            plot.setLabel("left", out.ylabel)
        self._output_layout.addWidget(plot)


class NotebookPage(QWidget):
    """笔记本页：单元流 + 工具栏 + 右侧变量侧栏（.znbk 打开/保存）."""

    #: 状态提示（打开/保存/重启结果等，主窗口状态栏承接显示）
    status_message = Signal(str)

    def __init__(self, kernel: ReplKernel, bus: EventBus, parent: QWidget | None = None) -> None:
        """初始化笔记本页.

        Args:
            kernel: REPL 内核（单元执行与变量命名空间来源）。
            bus: 事件总线（cls()/clc() 清屏事件清全部单元输出）。
        """
        super().__init__(parent)
        self._kernel = kernel
        self._notebook = Notebook(cells=[new_cell(_WELCOME_SOURCE)])
        self._path: Path | None = None
        self._dirty = False
        self._widgets: list[CellWidget] = []
        self._current = 0
        self._build_ui()
        self._rebuild_cells()
        bus.subscribe(ReplKernel.TOPIC_CONSOLE_CLEAR, self._clear_all_outputs)

    # ------------------------------------------------------------ 公开接口

    @property
    def path(self) -> Path | None:
        """当前笔记本文件路径（未保存过为 None）."""
        return self._path

    def is_dirty(self) -> bool:
        """是否有未保存修改."""
        return self._dirty

    def maybe_save(self) -> bool:
        """关闭前保存询问：接受/保存返回 True（可放弃），取消返回 False."""
        if not self._dirty:
            return True
        ret = QMessageBox.question(
            self,
            "保存笔记本",
            "笔记本有未保存的修改，是否保存？",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
        )
        if ret == QMessageBox.Cancel:
            return False
        if ret == QMessageBox.Save:
            return self.save()
        return True  # Discard

    def refresh_vars(self) -> None:
        """按命名空间当前状态刷新变量浏览器."""
        self._var_model.set_vars(whos(self._kernel.namespace, self._kernel.builtin_names))

    def refresh_theme(self) -> None:
        """主题切换后重绘工具栏图标与全部单元输出（内嵌绘图随主题换色）."""
        self._refresh_tool_icons()
        self._refresh_var_icon()
        for widget in self._widgets:
            widget.refresh_icons()
            widget.render_outputs()

    def restart_kernel(self) -> None:
        """重启内核：确认后清空命名空间/计数并失效全部单元输出（jupyter Restart Kernel 语义）."""
        ret = QMessageBox.question(
            self,
            "重启内核",
            "重启将清空全部变量与单元输出，确认重启？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return
        self._kernel.restart_kernel()
        for widget in self._widgets:
            widget.cell.outputs.clear()
            widget.cell.execution_count = None
            widget.render_outputs()
        self._dirty = True
        self.refresh_vars()
        self._set_status("内核已重启")

    # ------------------------------------------------------------ 文档操作

    def new_document(self) -> None:
        """新建笔记本（含演示首格）."""
        self._notebook = Notebook(cells=[new_cell(_WELCOME_SOURCE)])
        self._path = None
        self._dirty = False
        self._rebuild_cells()

    def open_path(self, path: Path) -> None:
        """打开 .znbk 笔记本（解析失败置状态提示，不破坏当前文档）."""
        try:
            notebook = load_notebook(path)
        except NotebookError as exc:
            self._set_status(f"打开失败: {exc}")
            return
        if not notebook.cells:
            notebook.cells.append(new_cell())
        self._notebook = notebook
        self._path = Path(path)
        self._dirty = False
        self._rebuild_cells()
        self._set_status(f"已打开: {path.name}")

    def save(self) -> bool:
        """保存笔记本（无路径时转另存为）；返回是否成功."""
        if self._path is None:
            return self.save_as()
        return self._save_to(self._path)

    def save_as(self) -> bool:
        """另存为（文件对话框选路径）."""
        target, _filter = QFileDialog.getSaveFileName(
            self, "另存笔记本", str(self._path or "notebook.znbk"), "zylab 笔记本 (*.znbk)"
        )
        if not target:
            return False
        return self._save_to(Path(target))

    # ------------------------------------------------------------ 单元操作

    def run_current(self) -> None:
        """运行当前焦点格."""
        self._run_at(self._current)

    def run_all(self) -> None:
        """顺序运行全部单元（某格出错仍继续，同 jupyter Run All）."""
        for index in range(len(self._widgets)):
            self._run_at(index)

    def run_from_current(self) -> None:
        """从当前焦点格运行到末尾."""
        for index in range(self._current, len(self._widgets)):
            self._run_at(index)

    def insert_after_current(self) -> None:
        """在当前格下方插入空格并聚焦."""
        self.insert_at(self._current + 1)

    def insert_at(self, index: int) -> None:
        """在指定位置插入空格并聚焦."""
        index = max(0, min(index, len(self._notebook.cells)))
        cell = new_cell()
        self._notebook.cells.insert(index, cell)
        self._dirty = True
        self._rebuild_cells()
        self._focus_index(index)

    def delete_current(self) -> None:
        """删除当前格（删空则自动补一空格）."""
        if not self._notebook.cells:
            return
        del self._notebook.cells[self._current]
        self._dirty = True
        if not self._notebook.cells:
            self._notebook.cells.append(new_cell())
            self._current = 0
        else:
            self._current = min(self._current, len(self._notebook.cells) - 1)
        self._rebuild_cells()
        self._focus_index(self._current)

    def move_current(self, offset: int) -> None:
        """当前格上移/下移（offset = -1 / +1）."""
        target = self._current + offset
        if not 0 <= target < len(self._notebook.cells):
            return
        cells = self._notebook.cells
        cells[self._current], cells[target] = cells[target], cells[self._current]
        self._dirty = True
        self._rebuild_cells()
        self._focus_index(target)

    # ------------------------------------------------------------ 内部实现

    def _build_ui(self) -> None:
        """组装文档级工具栏 + 单元流滚动区 + 可折叠变量侧栏 + 页面级快捷键."""
        # 文档级操作（单元级操作在每格悬停工具条，jupyter 语义两极分化）；
        # 纯图标按钮（tooltip 承载语义），与分析页工具行同款规格
        bar = QHBoxLayout()
        bar.setSpacing(theme.SPACING_SM)
        icon_size = QSize(16, 16)
        self._btn_new = QPushButton(objectName="toolBtn")
        self._btn_new.setToolTip("新建笔记本")
        self._btn_open = QPushButton(objectName="toolBtn")
        self._btn_open.setToolTip("打开笔记本 (.znbk)")
        self._btn_save = QPushButton(objectName="toolBtn")
        self._btn_save.setToolTip("保存笔记本 (Ctrl+S)，未保存过自动弹出另存对话框")
        self._btn_run_all = QPushButton(objectName="toolBtn")
        self._btn_run_all.setToolTip("全部运行（出错不中断）")
        self._btn_restart = QPushButton(objectName="toolBtn")
        self._btn_restart.setToolTip("重启内核：清空全部变量与单元输出，执行计数归零")
        self._refresh_tool_icons()
        for button in (self._btn_new, self._btn_open, self._btn_save, self._btn_run_all, self._btn_restart):
            button.setIconSize(icon_size)
            bar.addWidget(button)
        bar.addStretch(1)
        # 变量面板切换独立分组靠右（与其他文档操作区分，选中态主色高亮）
        self._btn_vars = QPushButton(objectName="notebookVarToggle")
        self._btn_vars.setCheckable(True)
        self._btn_vars.setToolTip("切换变量面板显示 (Ctrl+Shift+V)")
        self._btn_vars.setIconSize(icon_size)
        self._refresh_var_icon()
        bar.addWidget(self._btn_vars)
        self._btn_new.clicked.connect(self._on_new)
        self._btn_open.clicked.connect(self._on_open)
        self._btn_save.clicked.connect(self._on_save)
        self._btn_run_all.clicked.connect(self.run_all)
        self._btn_restart.clicked.connect(self.restart_kernel)
        self._btn_vars.toggled.connect(self._on_vars_toggled)

        self._cells_host = QWidget()
        self._cells_layout = QVBoxLayout(self._cells_host)
        self._cells_layout.setContentsMargins(theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_SM, theme.SPACING_MD)
        self._cells_layout.setSpacing(theme.SPACING_MD)
        self._cells_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._cells_host)
        scroll.setFrameShape(QFrame.NoFrame)

        # 变量侧栏：默认收起，单元流占满全宽；展开时 4:1（jupyterlab 侧栏语义）
        self._var_model = VarTableModel(self)
        self._var_view = QTableView(objectName="varBrowser")
        self._var_view.setModel(self._var_model)
        self._var_view.setItemDelegate(VarTagDelegate(self._var_view))
        self._var_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._var_view.verticalHeader().setVisible(False)
        self._var_view.setVisible(False)
        # 双击打开 MATLAB 风格变量详情对话框
        self._var_view.doubleClicked.connect(self._open_var_detail)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(self._var_view)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        self._splitter = splitter

        root = QVBoxLayout(self)
        root.setContentsMargins(0, theme.SPACING_SM, theme.SPACING_MD, theme.SPACING_MD)
        root.setSpacing(theme.SPACING_SM)
        root.addLayout(bar)
        root.addWidget(splitter, stretch=1)
        self._install_shortcuts()
        self.refresh_vars()

    def _install_shortcuts(self) -> None:
        """页面级快捷键：从此运行/下方插入/上下移格/另存/切换变量面板."""
        bindings = (
            ("Ctrl+Shift+Return", self.run_from_current),
            ("Ctrl+Shift+Enter", self.run_from_current),
            ("Alt+Return", self.insert_after_current),
            ("Alt+Enter", self.insert_after_current),
            ("Ctrl+Shift+Up", lambda: self.move_current(-1)),
            ("Ctrl+Shift+Down", lambda: self.move_current(1)),
            ("Ctrl+Shift+S", self.save_as),
            ("Ctrl+Shift+V", self._btn_vars.toggle),
        )
        for key, handler in bindings:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(handler)  # type: ignore[arg-type]（信号签名差异）

    def _rebuild_cells(self) -> None:
        """按 notebook.cells 重建单元控件列表（结构变化后调用）."""
        for widget in self._widgets:
            widget.deleteLater()
        self._widgets = []
        while self._cells_layout.count() > 1:  # 保留末尾 stretch
            item = self._cells_layout.takeAt(0)
            widget = item.widget() if item else None
            if widget is not None:
                widget.deleteLater()
        for index, cell in enumerate(self._notebook.cells):
            widget = CellWidget(cell)
            widget.editor.run_requested.connect(lambda advance, w=widget: self._on_run_shortcut(w, advance))
            widget.action_requested.connect(lambda action, w=widget: self._on_cell_action(w, action))
            widget.editor.focusInEvent = self._wrap_focus_in(widget, index)
            self._cells_layout.insertWidget(self._cells_layout.count() - 1, widget)
            self._widgets.append(widget)
        self._current = min(self._current, max(0, len(self._widgets) - 1))

    def _on_cell_action(self, widget: CellWidget, action: str) -> None:
        """单元级工具条动作分发：运行/上移/下移/下方插入/删除."""
        index = self._widgets.index(widget)
        if action == "run":
            self._run_at(index)
        elif action == "up":
            self._current = index
            self.move_current(-1)
        elif action == "down":
            self._current = index
            self.move_current(1)
        elif action == "insert":
            self.insert_at(index + 1)
        elif action == "delete":
            self._delete_at(index)

    def _delete_at(self, index: int) -> None:
        """删除指定格（不改变当前焦点格语义，删空自动补格）."""
        if not 0 <= index < len(self._notebook.cells):
            return
        del self._notebook.cells[index]
        self._dirty = True
        if not self._notebook.cells:
            self._notebook.cells.append(new_cell())
            self._current = 0
        else:
            self._current = min(
                self._current if self._current < index else self._current - 1, len(self._notebook.cells) - 1
            )
        self._rebuild_cells()
        self._focus_index(max(0, min(self._current, len(self._widgets) - 1)))

    def _on_vars_toggled(self, visible: bool) -> None:
        """变量按钮切换：显示侧栏并联动图标强调色."""
        self._toggle_vars(visible)
        self._refresh_var_icon()

    def _toggle_vars(self, visible: bool) -> None:
        """切换变量侧栏显示（jupyterlab 侧栏收起/展开；展开宽度不超过内容区 1/4）."""
        self._update_var_max_width()
        self._var_view.setVisible(visible)
        if visible:
            total = self._splitter.width()
            quarter = min(total // 4, self._var_view.maximumWidth())
            self._splitter.setSizes([total - quarter, quarter])

    def _update_var_max_width(self) -> None:
        """变量侧栏最宽不超过页面 1/4（保底 220px，避免过窄不可用）."""
        self._var_view.setMaximumWidth(max(220, self.width() // 4))

    def resizeEvent(self, event) -> None:  # Qt 命名约定
        """页面尺寸变化时同步变量侧栏宽度上限（≤1/4）."""
        super().resizeEvent(event)
        self._update_var_max_width()

    def _open_var_detail(self, index) -> None:
        """双击变量行：弹出 MATLAB 风格详情对话框（只读）."""
        info = self._var_model.info_at(index.row())
        if info is None:
            return
        value = self._kernel.namespace.get(info.name)
        exec_dialog(VarDetailDialog(info, value, self))

    def _refresh_tool_icons(self) -> None:
        """按当前主题重绘文档级工具按钮图标（单色剪影随按钮文字色着色）."""
        pal = theme.current_palette()
        for name, button in (
            ("new", self._btn_new),
            ("open_file", self._btn_open),
            ("save", self._btn_save),
            ("run_all", self._btn_run_all),
            ("rerun", self._btn_restart),
        ):
            button.setIcon(nav_icon(name, pal.text_on_primary))

    def _refresh_var_icon(self) -> None:
        """变量面板切换按钮图标（checked 选中态用强调色）."""
        pal = theme.current_palette()
        color = pal.nav_accent if self._btn_vars.isChecked() else pal.text_on_primary
        self._btn_vars.setIcon(nav_icon("variable", color))

    def _wrap_focus_in(self, widget: CellWidget, index: int):
        """包装编辑器 focusInEvent：记录当前焦点格并保留默认聚焦行为."""
        original = widget.editor.focusInEvent

        def _focus_in(event) -> None:  # Qt 命名约定
            self._current = index
            original(event)

        return _focus_in

    def _on_run_shortcut(self, widget: CellWidget, advance: bool) -> None:
        """编辑器运行快捷键：运行本格；advance 时推进/新建下格."""
        index = self._widgets.index(widget)
        self._run_at(index)
        if advance:
            if index + 1 < len(self._widgets):
                self._focus_index(index + 1)
            else:
                self.insert_at(index + 1)

    def _run_at(self, index: int) -> None:
        """执行指定格：同步源码 → 内核执行 → 回写输出 → 刷新变量."""
        if not 0 <= index < len(self._widgets):
            return
        widget = self._widgets[index]
        widget.sync_to_model()
        execution = self._kernel.execute_cell(widget.cell.source)
        widget.cell.outputs = list(execution.outputs)
        widget.cell.execution_count = execution.count
        widget.render_outputs()
        self._dirty = True
        self.refresh_vars()

    def _focus_index(self, index: int) -> None:
        """聚焦指定格编辑器（滚动到可见由 QScrollArea 自动处理）."""
        self._current = index
        if 0 <= index < len(self._widgets):
            self._widgets[index].focus_editor()

    def _clear_all_outputs(self, _payload: object = None) -> None:
        """cls()/clc() 清空全部单元输出（MATLAB clc 清屏语义，变量保留）."""
        for widget in self._widgets:
            widget.cell.outputs.clear()
            widget.render_outputs()
        self._dirty = True

    def _set_status(self, text: str) -> None:
        """状态提示经信号送主窗口状态栏显示（避免页面内挤占工具栏空间）."""
        self.status_message.emit(text)

    def _on_new(self) -> None:
        """新建前保存询问."""
        if not self.maybe_save():
            return
        self.new_document()

    def _on_save(self, _checked: bool = False) -> None:
        """工具栏保存（clicked 信号带 checked 参数，显式丢弃）."""
        self.save()

    def _on_save_as(self, _checked: bool = False) -> None:
        """工具栏另存为（clicked 信号带 checked 参数，显式丢弃）."""
        self.save_as()

    def _on_open(self) -> None:
        """打开前保存询问."""
        if not self.maybe_save():
            return
        target, _filter = QFileDialog.getOpenFileName(self, "打开笔记本", "", "zylab 笔记本 (*.znbk);;所有文件 (*)")
        if target:
            self.open_path(Path(target))

    def _save_to(self, path: Path) -> bool:
        """保存到指定路径（失败置状态提示，成功清 dirty）."""
        for widget in self._widgets:
            widget.sync_to_model()
        try:
            save_notebook(path, self._notebook)
        except NotebookError as exc:
            self._set_status(f"保存失败: {exc}")
            return False
        self._path = path
        self._dirty = False
        self._set_status(f"已保存: {path.name}")
        return True
