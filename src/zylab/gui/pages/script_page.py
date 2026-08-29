"""脚本编辑页：多行脚本编辑 + 一键运行到 REPL 内核（复用控制台命名空间）."""

from __future__ import annotations

from zylab.console import ReplKernel

from .. import theme
from ..qt_compat import QFont, QPlainTextEdit, QPushButton, QSplitter, Qt, QVBoxLayout, QWidget, Signal

__all__ = ["ScriptPage"]

# 初始示例脚本（演示与工作区联动）
_DEFAULT_SCRIPT = """\
# zylab 脚本示例：与控制台共享工作区
x = linspace(0, 4 * pi, 200)
y = sin(x) * exp(-x / 10)
plot(x, y, title="衰减振荡", xlabel="x", ylabel="y")
"""


class ScriptPage(QWidget):
    """脚本页：上方编辑器 + 下方输出，运行结果写入共享 REPL 工作区."""

    #: 脚本运行完成后发出（宿主页据此刷新变量浏览器等）
    run_finished = Signal()

    def __init__(self, kernel: ReplKernel, parent: QWidget | None = None) -> None:
        """初始化脚本页.

        Args:
            kernel: REPL 内核（与控制台页共享命名空间）。
        """
        super().__init__(parent)
        self._kernel = kernel
        self._build_ui()

    def _build_ui(self) -> None:
        """组装编辑器/输出分栏与运行按钮."""
        self._editor = QPlainTextEdit()
        self._editor.setPlainText(_DEFAULT_SCRIPT)
        self._editor.setFont(QFont(theme.FONT_MONO.strip('"').split(",")[0], 10))

        self._output = QPlainTextEdit(readOnly=True)
        self._output.setFont(QFont(theme.FONT_MONO.strip('"').split(",")[0], 10))

        self._run_button = QPushButton("运行 (Ctrl+Enter)")
        self._run_button.setMinimumHeight(34)
        self._run_button.setShortcut("Ctrl+Return")

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self._editor)
        splitter.addWidget(self._output)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        root = QVBoxLayout(self)
        root.setContentsMargins(theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD)
        root.setSpacing(theme.SPACING_SM)
        root.addWidget(self._run_button)
        root.addWidget(splitter, stretch=1)
        self._run_button.clicked.connect(self._on_run)

    def _on_run(self) -> None:
        """执行编辑器整段脚本并渲染输出."""
        source = self._editor.toPlainText()
        result = self._kernel.run_script(source)
        parts: list[str] = []
        if result.stdout:
            parts.append(result.stdout.rstrip())
        if result.stderr:
            parts.append(result.stderr.rstrip())
        if result.result_repr is not None:
            parts.append(f"ans = {result.result_repr}")
        if result.error:
            parts.append(result.error.rstrip())
        text = "\n".join(parts) if parts else "(无输出)"
        self._output.setPlainText(text)
        self.run_finished.emit()
