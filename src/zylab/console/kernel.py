"""zylab.console REPL 内核（MATLAB 式交互计算环境）.

职责：执行代码片段、管理命名空间（注入 NumPy 常用符号与 whos/plot/run 命令）、
捕获 stdout/stderr 与表达式结果（``ans``）、支持多行块续行检测与脚本执行、
执行笔记本单元（``execute_cell``：语句 + 末表达式回显 + 绘图请求捕获）。

安全说明：REPL 本质是本地代码执行器（同 MATLAB 命令窗口），仅面向本地交互输入，
不接收远程/不受信来源代码。
"""

from __future__ import annotations

import ast
import code
import contextlib
import io
import logging
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from zylab.core.events import EventBus
from zylab.sci import TOPIC_PLOT_REQUESTED, PlotRequest, format_whos, make_plot_function, whos
from zylab.sci.notebook import (
    CellOutput,
    ErrorOutput,
    PlotOutput,
    PlotSeries,
    ResultOutput,
    StreamOutput,
)

__all__ = ["CellExecution", "ExecResult", "ReplKernel"]

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

# help() 打印的内置命令说明
_HELP_TEXT = """内置命令:
  whos()          列出工作区变量（MATLAB 风格表格）
  plot(x, y)      绘制曲线（自动切换右侧绘图页）
  run(path)       执行 Python 脚本文件
  cls() / clc()   清空控制台输出区
  clear()         清除全部用户变量（保留内置符号与 ans）
  help()          显示本帮助
另: np/numpy 模块与 arange/linspace/pi/sin 等常用符号可直接使用, 输入 help(np) 可查其文档."""


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


@dataclass(frozen=True)
class CellExecution:
    """一次笔记本单元执行的结果（notebook 渲染层数据源）.

    :param count: 执行序号（内核单调递增；空源不执行、不递增）。
    :param outputs: 翻译后的输出列表（流/结果/错误/绘图，按发生顺序）。
    """

    count: int
    outputs: list[CellOutput]


@dataclass(frozen=True)
class _CellRun:
    """单元执行的原始结果（输出组装前的中间形态，模块私有）.

    :param stdout: 捕获的标准输出。
    :param stderr: 捕获的标准错误。
    :param requests: 执行窗口内捕获的绘图请求列表。
    :param result: 末表达式求值结果（无结果时未定义，以 has_result 为准）。
    :param has_result: 末表达式是否存在非 None 结果。
    :param error: 异常格式化文本（无异常为 None）。
    """

    stdout: str
    stderr: str
    requests: list[PlotRequest]
    result: Any
    has_result: bool
    error: str | None


class ReplKernel:
    """REPL 内核：进程内 Python 交互解释器.

    用法::

        kernel = ReplKernel()
        kernel.execute("x = linspace(0, pi, 100)")
        result = kernel.execute("sin(x) * 2")
        assert result.result_repr is not None

    内置命令（namespace 直呼即用）：``whos()`` 列变量、``plot(x, y)`` 绘图、
    ``run(path)`` 跑脚本、``cls()``/``clc()`` 清屏（发 ``console.clear`` 事件）、
    ``clear()`` 清用户变量（保留内置符号）、``help()`` 列内置命令。
    """

    #: 清屏事件主题（GUI 输出区订阅此主题清空显示）
    TOPIC_CONSOLE_CLEAR = "console.clear"

    def __init__(self, bus: EventBus | None = None) -> None:
        """初始化内核并构建命名空间."""
        self.bus = bus or EventBus()
        self.namespace: dict[str, Any] = {}
        #: 系统内置符号名集合（初始化时快照，clear() 与变量浏览器据此区分用户变量）
        self.builtin_names: frozenset[str] = frozenset()
        #: 笔记本单元执行序号（execute_cell 单调递增）
        self.execution_count = 0
        self._init_namespace()

    def _init_namespace(self) -> None:
        """构建 REPL 命名空间：NumPy 符号 + whos/plot/run/cls/clear/help 命令."""
        import numpy as np

        ns = self.namespace
        ns["np"] = np
        ns["numpy"] = np
        for name in _NP_SYMBOLS:
            ns[name] = getattr(np, name)

        def _whos() -> None:
            """列出当前工作区变量（MATLAB whos 风格表格，仅用户变量）."""
            print(format_whos([i for i in whos(ns, self.builtin_names) if not i.builtin]))

        def _cls() -> None:
            """清空控制台输出区（clc 为别名）."""
            self.bus.publish(self.TOPIC_CONSOLE_CLEAR)

        def _clear() -> None:
            """清除全部用户变量（保留内置符号与 ans）."""
            dropped = [
                name
                for name, value in ns.items()
                if not name.startswith("_") and name not in self.builtin_names and name != "ans"
            ]
            for name in dropped:
                del ns[name]
            print(f"已清除 {len(dropped)} 个变量: {', '.join(dropped)}" if dropped else "无用户变量")

        def _help() -> None:
            """打印内置命令帮助."""
            print(_HELP_TEXT)

        ns["whos"] = _whos
        ns["plot"] = make_plot_function(self.bus)
        ns["run"] = self.run_file
        ns["cls"] = _cls
        ns["clc"] = _cls
        ns["clear"] = _clear
        ns["help"] = _help
        self.builtin_names = frozenset(ns)

    def restart_kernel(self) -> None:
        """重启内核：清空命名空间并重建内置符号，执行计数归零（jupyter Restart Kernel 语义）."""
        self.namespace.clear()
        self.execution_count = 0
        self._init_namespace()

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
        """返回当前工作区用户变量表格文本（不含内置符号）."""
        return format_whos([i for i in whos(self.namespace, self.builtin_names) if not i.builtin])

    def execute_cell(self, source: str) -> CellExecution:
        """执行笔记本单元（jupyter 语义）.

        - 整体按 ``exec`` 模式执行语句；末尾独立表达式语句求值回显
          （结果 repr 进 ResultOutput，同时写入命名空间 ``ans``/``_``）；
        - 执行期间临时订阅总线收集 ``plot()`` 绘图请求（数组快照防后续变异），
          同一单元的全部请求合并为一个 PlotOutput（多 series 单图）；
        - 异常不抛出，翻译为 ErrorOutput（含 ename 与完整回溯）；
        - 空源不执行、不递增计数。

        :param source: 单元源码（多行完整代码块）。
        :returns: CellExecution（count + outputs）。
        """
        source = source.rstrip()
        if not source:
            return CellExecution(count=self.execution_count, outputs=[])
        self.execution_count += 1
        count = self.execution_count
        try:
            code_exec, code_eval = self._compile_cell(source)
        except (SyntaxError, ValueError):
            return CellExecution(count=count, outputs=[self._error_output()])
        run = self._run_cell(code_exec, code_eval)
        outputs: list[CellOutput] = []
        if run.stdout:
            outputs.append(StreamOutput(name="stdout", text=run.stdout))
        if run.requests:
            outputs.append(self._plot_output(run.requests))
        if run.has_result:
            outputs.append(self._result_output(run.result))
        if run.error:
            outputs.append(self._error_output(run.error))
        if run.stderr:
            outputs.append(StreamOutput(name="stderr", text=run.stderr))
        return CellExecution(count=count, outputs=outputs)

    def _compile_cell(self, source: str) -> tuple[Any, Any]:
        """ast 拆分单元源码并编译（jupyter 末表达式语义）.

        :param source: 单元源码。
        :returns: ``(exec 编译产物, eval 编译产物)``（无对应部分为 None）。
        :raises SyntaxError: 源码语法错误（含 ast 拆分/编译阶段）。
        :raises ValueError: 空表达式等编译期错误。
        """
        tree = ast.parse(source, filename="<cell>")
        exec_statements = tree.body
        eval_node: ast.expr | None = None
        if exec_statements:
            last = exec_statements[-1]
            if isinstance(last, ast.Expr):
                exec_statements.pop()
                eval_node = last.value
        code_exec = (
            compile(ast.Module(body=exec_statements, type_ignores=[]), "<cell>", "exec") if exec_statements else None
        )
        code_eval = compile(ast.Expression(body=eval_node), "<cell>", "eval") if eval_node is not None else None
        return code_exec, code_eval

    def _run_cell(self, code_exec: Any, code_eval: Any) -> _CellRun:
        """执行编译产物（重定向输出 + 临时订阅捕获绘图请求）.

        :param code_exec: 语句部分编译产物（可为 None）。
        :param code_eval: 末表达式编译产物（可为 None）。
        :returns: :class:`_CellRun`（stdout/stderr/绘图请求/结果/异常文本）。
        """
        # 绘图请求捕获：执行窗口内临时订阅总线（kernel 的 plot 绑定自身 bus）
        requests: list[PlotRequest] = []

        def _collect(request: PlotRequest) -> None:
            requests.append(request)

        self.bus.subscribe(TOPIC_PLOT_REQUESTED, _collect)
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        error: str | None = None
        result: Any | None = None
        has_result = False
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                if code_exec is not None:
                    exec(code_exec, self.namespace)  # notebook 内核预期执行用户代码
                if code_eval is not None:
                    result = eval(code_eval, self.namespace)  # notebook 内核预期执行用户代码
                    has_result = result is not None
                    if has_result:
                        self.namespace["ans"] = result
                        self.namespace["_"] = result
        except KeyboardInterrupt:
            error = "KeyboardInterrupt: 用户中断"
        except Exception:
            error = traceback.format_exc()
        finally:
            self.bus.unsubscribe(TOPIC_PLOT_REQUESTED, _collect)
        return _CellRun(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            requests=requests,
            result=result,
            has_result=has_result,
            error=error,
        )

    def _result_output(self, value: Any) -> ResultOutput:
        """表达式值转结果输出（类型名 + shape 摘要）."""
        shape = getattr(value, "shape", None)
        return ResultOutput(
            repr_text=repr(value),
            type_name=type(value).__name__,
            shape=str(tuple(shape)) if shape is not None else "",
        )

    def _error_output(self, error_text: str | None = None) -> ErrorOutput:
        """异常回溯文本转错误输出（ename 从末行解析）."""
        text = error_text if error_text is not None else traceback.format_exc()
        last_line = text.rstrip().splitlines()[-1] if text.strip() else ""
        ename = last_line.partition(":")[0].strip() or "Exception"
        return ErrorOutput(ename=ename, traceback_text=text)

    @staticmethod
    def _plot_output(requests: list[PlotRequest]) -> PlotOutput:
        """绘图请求列表合并为单图输出（各请求数组拷贝快照防后续变异）."""
        series = [
            PlotSeries(
                x=np.asarray(req.x, dtype=float).tolist(),
                y=np.asarray(req.y, dtype=float).tolist(),
                label=str(req.extra.get("label", "")),
            )
            for req in requests
        ]
        first = requests[0]
        return PlotOutput(
            title=first.title,
            xlabel=first.xlabel,
            ylabel=first.ylabel,
            series=series,
        )

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
