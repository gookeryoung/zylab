"""zylab.console REPL 内核（MATLAB 式交互计算环境）.

职责：执行代码片段、管理命名空间（注入 NumPy 常用符号与 whos/plot/run 命令）、
捕获 stdout/stderr 与表达式结果（``ans``）、支持多行块续行检测与脚本执行。

安全说明：REPL 本质是本地代码执行器（同 MATLAB 命令窗口），仅面向本地交互输入，
不接收远程/不受信来源代码。
"""

from __future__ import annotations

import code
import contextlib
import io
import logging
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zylab.core.events import EventBus
from zylab.sci import format_whos, make_plot_function, whos

__all__ = ["ExecResult", "ReplKernel"]

logger = logging.getLogger(__name__)

# 注入命名空间的 NumPy 常用符号（MATLAB 风格直接可用）
_NP_SYMBOLS = (
    "arange",
    "array",
    "cos",
    "diag",
    "e",
    "exp",
    "eye",
    "inf",
    "linspace",
    "log",
    "log10",
    "nan",
    "ones",
    "pi",
    "sin",
    "sqrt",
    "tan",
    "zeros",
)


@dataclass(frozen=True)
class ExecResult:
    """一次执行的完整结果（GUI/CLI 渲染层的唯一数据源）.

    :param source: 执行的源码。
    :param stdout: 捕获的标准输出。
    :param stderr: 捕获的标准错误。
    :param result_repr: 表达式结果的 repr（无结果时为 None；结果同时写入命名空间 ``ans``/``_``）。
    :param error: 异常格式化文本（无异常为 None）。
    :param incomplete: 语法不完整（需要续行，如 ``for ...:`` 未闭合）。
    """

    source: str
    stdout: str = ""
    stderr: str = ""
    result_repr: str | None = None
    error: str | None = None
    incomplete: bool = False


class ReplKernel:
    """REPL 内核：进程内 Python 交互解释器.

    用法::

        kernel = ReplKernel()
        kernel.execute("x = linspace(0, pi, 100)")
        result = kernel.execute("sin(x) * 2")
        assert result.result_repr is not None
    """

    def __init__(self, bus: EventBus | None = None) -> None:
        """初始化内核并构建命名空间."""
        self.bus = bus or EventBus()
        self.namespace: dict[str, Any] = {}
        self._init_namespace()

    def _init_namespace(self) -> None:
        """构建 REPL 命名空间：NumPy 符号 + whos/plot/run 命令."""
        import numpy as np

        ns = self.namespace
        ns["np"] = np
        ns["numpy"] = np
        for name in _NP_SYMBOLS:
            ns[name] = getattr(np, name)

        def _whos() -> None:
            """列出当前工作区变量（MATLAB whos 风格表格）."""
            print(format_whos(whos(ns)))

        ns["whos"] = _whos
        ns["plot"] = make_plot_function(self.bus)
        ns["run"] = self.run_file

    def execute(self, source: str) -> ExecResult:
        """执行代码片段.

        - 语法不完整返回 ``incomplete=True``（调用方应继续收集输入）；
        - 表达式语句结果写入命名空间 ``ans``/``_`` 并填入 ``result_repr``；
        - 异常不抛出，格式化后填入 ``error``。
        """
        source = source.rstrip()
        if not source:
            return ExecResult(source=source)
        try:
            code_obj = code.compile_command(source, filename="<console>", symbol="single")
        except (SyntaxError, OverflowError, ValueError):
            error = traceback.format_exc(limit=0)
            return ExecResult(source=source, error=error)
        if code_obj is None:
            # 多行块末尾补空行标记块结束（compile_command 要求块以空行收尾）
            code_obj = code.compile_command(source + "\n", filename="<console>", symbol="single")
        if code_obj is None:
            return ExecResult(source=source, incomplete=True)
        return self._exec_code(code_obj, source)

    def run_script(self, source: str) -> ExecResult:
        """执行多行脚本文本（``exec`` 模式，语句结果不回显）.

        :param source: 脚本源码。
        :returns: ExecResult；语法错误时 error 非空。
        """
        try:
            code_obj = compile(source, filename="<script>", mode="exec")
        except (SyntaxError, OverflowError, ValueError):
            return ExecResult(source=source, error=traceback.format_exc(limit=0))
        return self._exec_code(code_obj, source)

    def run_file(self, path: str | Path) -> ExecResult:
        """执行脚本文件（``exec`` 模式，脚本末尾表达式不打印）.

        :param path: 脚本路径（``.py``）。
        :returns: ExecResult；文件不存在/读取失败时 error 非空。
        """
        script = Path(path)
        if not script.is_file():
            return ExecResult(source=str(path), error=f"FileNotFoundError: 脚本不存在: {script}")
        try:
            source = script.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return ExecResult(source=str(path), error=f"{type(exc).__name__}: {exc}")
        try:
            code_obj = compile(source, filename=str(script), mode="exec")
        except (SyntaxError, OverflowError, ValueError):
            return ExecResult(source=source, error=traceback.format_exc(limit=0))
        logger.info("执行脚本: %s", script)
        return self._exec_code(code_obj, source)

    def whos(self) -> str:
        """返回当前工作区变量表格文本."""
        return format_whos(whos(self.namespace))

    def _exec_code(self, code_obj: Any, source: str) -> ExecResult:
        """在命名空间内执行编译产物，捕获输出/结果/异常."""
        result_holder: dict[str, Any] = {}
        old_hook = sys.displayhook

        def _hook(value: Any) -> None:
            if value is not None:
                result_holder["value"] = value
                self.namespace["ans"] = value
                self.namespace["_"] = value

        sys.displayhook = _hook
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        error: str | None = None
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(code_obj, self.namespace)  # REPL 内核预期执行用户代码
        except KeyboardInterrupt:
            error = "KeyboardInterrupt: 用户中断"
        except Exception:
            error = traceback.format_exc()
        finally:
            sys.displayhook = old_hook
        result_repr = repr(result_holder["value"]) if "value" in result_holder else None
        return ExecResult(
            source=source,
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            result_repr=result_repr,
            error=error,
        )
