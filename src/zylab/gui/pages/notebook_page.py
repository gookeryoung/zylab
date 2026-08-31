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
    VarInfo,
    load_notebook,
    new_cell,
    save_notebook,
    whos,
)

from .. import theme
from ..highlight import PythonHighlighter
from ..qt_compat import (
    QAbstractTableModel,
    QColor,
    QFileDialog,
    QFont,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QKeyEvent,
    QKeySequence,
    QLabel,
    QMessageBox,
    QModelIndex,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSplitter,
    Qt,
    QTableView,
    QVBoxLayout,
    QWidget,
    Signal,
)

__all__ = ["CellEditor", "CellWidget", "NotebookPage", "VarTableModel"]

_INVALID_INDEX = QModelIndex()

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


class VarTableModel(QAbstractTableModel):
    """变量浏览器数据模型（列：名称/类型/形状/元素类型/字节数/预览）."""

    _HEADERS = ("名称", "类型", "形状", "元素类型", "字节数", "预览")

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化空模型."""
        super().__init__(parent)
        self._vars: list[VarInfo] = []

    def set_vars(self, infos: list[VarInfo]) -> None:
        """整体替换变量列表并刷新视图."""
        self.beginResetModel()
        self._vars = list(infos)
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = _INVALID_INDEX) -> int:
        """行数（顶层）."""
        return 0 if parent.isValid() else len(self._vars)

    def columnCount(self, _parent: QModelIndex = _INVALID_INDEX) -> int:
        """列数."""
        return len(self._HEADERS)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> object:
        """单元格数据：内置符号行用次级色区分用户变量."""
        if not index.isValid():
            return None
        info = self._vars[index.row()]
        if role == Qt.DisplayRole:
            return (info.name, info.type_name, info.shape, info.dtype, str(info.nbytes), info.preview)[index.column()]
        if role == Qt.ForegroundRole:
            pal = theme.current_palette()
            return QColor(pal.text_secondary if info.builtin else pal.text_primary)
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> object:
        """表头数据."""
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self._HEADERS[section]
        return None


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

    #: 工具条按钮（符号, 动作名）——悬停卡片时浮现
    _TOOL_ITEMS = (("\u25b6", "run"), ("\u2191", "up"), ("\u2193", "down"), ("\uff0b", "insert"), ("\u2715", "delete"))

    def __init__(self, cell: NotebookCell, parent: QWidget | None = None) -> None:
        """初始化卡片并装载单元源码与既有输出.

        Args:
            cell: 绑定的笔记本单元（编辑与运行双向往 cell 同步）。
        """
        super().__init__(objectName="notebookCell", parent=parent)
        self._cell = cell
        self._editor = CellEditor(objectName="cellEditor")
        self._editor.setPlainText(cell.source)
        self._editor.setFont(_mono_font())
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
        self._count_label.setFont(_mono_font())
        self._count_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self._count_label.setFixedWidth(64)

        # 单元级工具条：固定占位宽度，悬停时按钮可见（不挤压格宽）
        self._tool_buttons: list[QPushButton] = []
        toolbar = QWidget(objectName="cellToolbar")
        toolbar.setFixedWidth(30)
        tool_layout = QVBoxLayout(toolbar)
        tool_layout.setContentsMargins(0, 0, 0, 0)
        tool_layout.setSpacing(2)
        for symbol, action in self._TOOL_ITEMS:
            button = QPushButton(symbol, objectName="cellToolButton")
            button.setFixedSize(26, 22)
            button.setToolTip(_TOOL_TIPS[action])
            button.setVisible(False)
            button.clicked.connect(lambda _checked=False, name=action: self.action_requested.emit(name))
            tool_layout.addWidget(button)
            self._tool_buttons.append(button)
        tool_layout.addStretch(1)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACING_SM)
        row.addWidget(self._count_label)
        row.addLayout(body, stretch=1)
        row.addWidget(toolbar)

        self._editor.setMinimumHeight(56)
        self._update_count_label()
        self.render_outputs()

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
        """刷新 ``In [n]:`` 序号栏（未运行为 ``In [ ]:``）."""
        count = self._cell.execution_count
        self._count_label.setText(f"In [{count}]:" if count is not None else "In [ ]:")

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
        label.setFont(_mono_font())
        label.setTextFormat(Qt.RichText)
        label.setWordWrap(True)
        label.setText(f'<pre style="margin:0">{html.escape(out.text.rstrip())}</pre>')
        label.setStyleSheet(f"color: {color}; padding: 4px 8px; background-color: {pal.bg_input}; border-radius: 4px;")
        self._output_layout.addWidget(label)

    def _add_result(self, out: ResultOutput) -> None:
        """表达式结果：Out[n] 前缀 + repr + 类型/形状摘要."""
        pal = theme.current_palette()
        label = QLabel(objectName="cellResult")
        label.setFont(_mono_font())
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
        label.setFont(_mono_font())
        label.setTextFormat(Qt.RichText)
        label.setWordWrap(True)
        label.setText(
            f'<pre style="margin:0; color: {pal.danger_text}">'
            f"<b>{html.escape(out.ename)}</b>\n{html.escape(out.traceback_text.rstrip())}</pre>"
        )
        label.setStyleSheet(f"padding: 4px 8px; background-color: {pal.bg_input}; border-radius: 4px;")
        self._output_layout.addWidget(label)

    def _add_plot(self, out: PlotOutput) -> None:
        """绘图输出：内嵌 pyqtgraph（多 series 同图、主题色循环）."""
        pal = theme.current_palette()
        plot = pg.PlotWidget(background=pal.bg_app, parent=self)
        plot.showGrid(x=True, y=True, alpha=0.3)
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
            plot.setTitle(out.title)
        if out.xlabel:
            plot.setLabel("bottom", out.xlabel)
        if out.ylabel:
            plot.setLabel("left", out.ylabel)
        if any(s.label for s in out.series):
            plot.addLegend()
        self._output_layout.addWidget(plot)


class NotebookPage(QWidget):
    """笔记本页：单元流 + 工具栏 + 右侧变量侧栏（.znbk 打开/保存）."""

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
        # 文档级操作（单元级操作在每格悬停工具条，jupyter 语义两极分化）
        bar = QHBoxLayout()
        bar.setSpacing(theme.SPACING_SM)
        self._btn_new = QPushButton("新建")
        self._btn_open = QPushButton("打开")
        self._btn_save = QPushButton("保存")
        self._btn_save.setToolTip("保存笔记本 (Ctrl+S)，未保存过自动弹出另存对话框")
        self._btn_run_all = QPushButton("全部运行")
        self._btn_restart = QPushButton("重启内核")
        self._btn_restart.setToolTip("清空全部变量与单元输出，执行计数归零")
        self._btn_vars = QPushButton("变量", objectName="notebookVarToggle")
        self._btn_vars.setCheckable(True)
        self._btn_vars.setToolTip("切换变量面板显示 (Ctrl+Shift+V)")
        for button in (
            self._btn_new,
            self._btn_open,
            self._btn_save,
            self._btn_run_all,
            self._btn_restart,
            self._btn_vars,
        ):
            bar.addWidget(button)
        self._status_label = QLabel(objectName="notebookStatus")
        self._status_label.setStyleSheet(f"color: {theme.current_palette().text_secondary};")
        bar.addWidget(self._status_label)
        bar.addStretch(1)
        self._btn_new.clicked.connect(self._on_new)
        self._btn_open.clicked.connect(self._on_open)
        self._btn_save.clicked.connect(self._on_save)
        self._btn_run_all.clicked.connect(self.run_all)
        self._btn_restart.clicked.connect(self.restart_kernel)
        self._btn_vars.toggled.connect(self._toggle_vars)

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
        self._var_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._var_view.verticalHeader().setVisible(False)
        self._var_view.setVisible(False)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(self._var_view)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

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

    def _toggle_vars(self, visible: bool) -> None:
        """切换变量侧栏显示（jupyterlab 侧栏收起/展开）."""
        self._var_view.setVisible(visible)

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
        """工具栏尾部状态提示（打开/保存结果等轻量反馈，避免弹窗）."""
        self._status_label.setText(text)

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


def _mono_font() -> QFont:
    """等宽字体（与脚本页一致）."""
    return QFont(theme.FONT_MONO.strip('"').split(",")[0], 10)
