"""控制台页：左侧选项卡（交互 / 脚本）+ 右侧选项卡（变量 / 绘图）."""

from __future__ import annotations

import html

from zylab.console import CommandHistory, ExecResult, ReplKernel
from zylab.core import EventBus
from zylab.sci import VarInfo, whos

from .. import theme
from ..qt_compat import (
    QAbstractTableModel,
    QHBoxLayout,
    QHeaderView,
    QKeyEvent,
    QModelIndex,
    QPlainTextEdit,
    QSplitter,
    Qt,
    QTableView,
    QTabWidget,
    QTextCursor,
    QVBoxLayout,
    QWidget,
    Signal,
)
from .plot_page import PlotPage
from .script_page import ScriptPage

__all__ = ["ConsolePage", "ReplInput", "VarTableModel"]


_INVALID_INDEX = QModelIndex()


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
        """单元格数据."""
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        info = self._vars[index.row()]
        return (info.name, info.type_name, info.shape, info.dtype, str(info.nbytes), info.preview)[index.column()]

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> object:
        """表头数据."""
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self._HEADERS[section]
        return None


class ReplInput(QPlainTextEdit):
    """REPL 输入框.

    - Enter：执行完整代码；语法不完整（如 ``for`` 未闭合）自动续行；
    - Shift+Enter：软换行；
    - Up/Down：光标位于首行/末行时浏览命令历史。
    """

    submitted = Signal(object)  # ExecResult

    def __init__(self, kernel: ReplKernel, history: CommandHistory, parent: QWidget | None = None) -> None:
        """初始化输入框."""
        super().__init__(objectName="replInput", parent=parent)
        self._kernel = kernel
        self._history = history
        self.setPlaceholderText("输入命令，Enter 执行（Shift+Enter 换行，↑/↓ 浏览历史）")
        self.setMaximumHeight(96)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # Qt 命名约定
        """按键分发."""
        key = event.key()
        if key in (Qt.Key_Return, Qt.Key_Enter) and not (event.modifiers() & Qt.ShiftModifier):
            self._submit_or_continue(event)
            return
        if key == Qt.Key_Up and self.textCursor().blockNumber() == 0:
            prev = self._history.previous(self.toPlainText())
            if prev is not None:
                self.setPlainText(prev)
                self.moveCursor(QTextCursor.End)
            return
        if key == Qt.Key_Down and self.textCursor().blockNumber() == self.blockCount() - 1:
            nxt = self._history.next()
            if nxt is not None:
                self.setPlainText(nxt)
                self.moveCursor(QTextCursor.End)
            return
        super().keyPressEvent(event)

    def _submit_or_continue(self, event: QKeyEvent) -> None:
        """完整则执行并发出 submitted；不完整则换行续写."""
        text = self.toPlainText().strip()
        if not text:
            return
        result = self._kernel.execute(text)
        if result.incomplete:
            super().keyPressEvent(event)  # 插入换行，继续收集
            return
        self._history.add(text)
        self.clear()
        self.submitted.emit(result)


class ConsolePage(QWidget):
    """控制台页：左侧「交互 / 脚本」选项卡，右侧「变量 / 绘图」选项卡.

    变量浏览器随每条命令执行自动刷新（常态显示）；
    plot 渲染后自动切换右侧到绘图选项卡 —— 左侧脚本与右侧绘图可同屏查看。
    """

    def __init__(
        self,
        kernel: ReplKernel,
        history: CommandHistory,
        bus: EventBus,
        parent: QWidget | None = None,
    ) -> None:
        """初始化控制台页."""
        super().__init__(parent)
        self._kernel = kernel
        self._build_ui(kernel, history, bus)
        self._append_html(
            f'<span style="color:{theme.current_palette().text_secondary}">'
            f"zylab 控制台就绪 · 右侧选项卡实时查看变量 · plot(x, y) 绘图 · 左侧「脚本」页可编辑运行多行脚本</span>"
        )

    def _build_ui(self, kernel: ReplKernel, history: CommandHistory, bus: EventBus) -> None:
        """组装布局."""
        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD)
        left_layout.setSpacing(theme.SPACING_SM)
        self._output = QPlainTextEdit(objectName="replOutput", readOnly=True)
        self._input = ReplInput(kernel, history)
        self._input.submitted.connect(self._on_result)
        left_layout.addWidget(self._output, stretch=1)
        left_layout.addWidget(self._input)

        # 左侧：交互 / 脚本（脚本与右侧绘图同屏）
        self._work_tabs = QTabWidget(objectName="consoleSideTabs")
        self._work_tabs.addTab(left, "交互")
        self._script_page = ScriptPage(kernel, self)
        self._work_tabs.addTab(self._script_page, "脚本")

        self._var_model = VarTableModel(self)
        var_view = QTableView(objectName="varBrowser")
        var_view.setModel(self._var_model)
        var_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        var_view.verticalHeader().setVisible(False)

        # 右侧：变量 / 绘图
        self._side_tabs = QTabWidget(objectName="consoleSideTabs")
        self._side_tabs.addTab(var_view, "变量")
        self._plot_page = PlotPage(bus, self)
        self._side_tabs.addTab(self._plot_page, "绘图")
        # plot 渲染后自动切到绘图选项卡（REPL 与脚本共用内核，脚本绘图同样触发）
        self._plot_page.plot_shown.connect(lambda: self._side_tabs.setCurrentIndex(1))
        # 脚本运行后变量表同步刷新（脚本与 REPL 共享命名空间）
        self._script_page.run_finished.connect(lambda: self._var_model.set_vars(whos(self._kernel.namespace)))

        splitter.addWidget(self._work_tabs)
        splitter.addWidget(self._side_tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def _on_result(self, result: ExecResult) -> None:
        """渲染执行结果并刷新变量浏览器（常态显示）."""
        pal = theme.current_palette()
        self._append_html(f'<span style="color:{pal.primary}">&gt;&gt;&gt; {html.escape(result.source)}</span>')
        if result.stdout:
            self._append_html(f"<pre>{html.escape(result.stdout.rstrip())}</pre>")
        if result.result_repr is not None:
            self._append_html(f'<span style="color:{pal.success_text}">ans = {html.escape(result.result_repr)}</span>')
        error_text = result.error or (result.stderr if result.stderr else "")
        if error_text:
            self._append_html(f'<pre style="color:{pal.error_text}">{html.escape(error_text.rstrip())}</pre>')
        self._var_model.set_vars(whos(self._kernel.namespace))
        self._output.moveCursor(QTextCursor.End)

    def _append_html(self, fragment: str) -> None:
        """追加 HTML 片段到输出区."""
        self._output.appendHtml(fragment)
