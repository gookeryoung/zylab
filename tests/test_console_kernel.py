"""console.kernel REPL 内核测试."""

from __future__ import annotations

import builtins

import pytest

from zylab.console import ReplKernel
from zylab.core import EventBus
from zylab.sci import TOPIC_PLOT_REQUESTED


def test_execute_assignment() -> None:
    """赋值语句应写入命名空间且无结果."""
    kernel = ReplKernel()
    result = kernel.execute("x = 5")
    assert result.error is None
    assert result.result_repr is None
    assert kernel.namespace["x"] == 5


def test_execute_expression_sets_ans() -> None:
    """表达式结果应写入 ans 与 _ 并填入 result_repr."""
    kernel = ReplKernel()
    kernel.execute("x = 5")
    result = kernel.execute("x * 2")
    assert result.result_repr == "10"
    assert kernel.namespace["ans"] == 10
    assert kernel.namespace["_"] == 10


def test_execute_none_result_not_recorded() -> None:
    """结果为 None 的表达式不写入 ans."""
    kernel = ReplKernel()
    result = kernel.execute("None")
    assert result.result_repr is None
    assert "ans" not in kernel.namespace


def test_execute_captures_stdout() -> None:
    """print 输出应被捕获到 stdout."""
    result = ReplKernel().execute("print('你好 zylab')")
    assert "你好 zylab" in result.stdout


def test_execute_numpy_symbols_available() -> None:
    """NumPy 常用符号应直接可用（MATLAB 风格）."""
    kernel = ReplKernel()
    result = kernel.execute("linspace(0, 1, 5)")
    assert result.error is None
    assert "array" in (result.result_repr or "")
    assert kernel.execute("np.zeros(3)").error is None


def test_execute_syntax_error() -> None:
    """语法错误应返回 error 且 incomplete 为 False."""
    result = ReplKernel().execute("def f(:")
    assert result.error is not None
    assert "SyntaxError" in result.error
    assert result.incomplete is False


def test_execute_incomplete_then_complete() -> None:
    """不完整输入返回 incomplete，补全后可执行."""
    kernel = ReplKernel()
    incomplete = kernel.execute("for i in range(3):")
    assert incomplete.incomplete is True
    assert incomplete.error is None
    complete = kernel.execute("for i in range(3):\n    y = i")
    assert complete.incomplete is False
    assert complete.error is None
    assert kernel.namespace["y"] == 2


def test_execute_empty_source() -> None:
    """空输入应返回全默认结果."""
    result = ReplKernel().execute("   ")
    assert result.stdout == ""
    assert result.error is None
    assert result.incomplete is False


def test_execute_exception_traceback() -> None:
    """运行时异常应格式化到 error 且不抛出."""
    result = ReplKernel().execute("1 / 0")
    assert result.error is not None
    assert "ZeroDivisionError" in result.error


def test_execute_whos_command() -> None:
    """whos() 命令应打印变量表格."""
    kernel = ReplKernel()
    kernel.execute("a_var = 1")
    result = kernel.execute("whos()")
    assert "a_var" in result.stdout
    assert "名称" in result.stdout


def test_execute_plot_command_publishes_event() -> None:
    """plot 命令应通过事件总线发布绘图请求."""
    bus = EventBus()
    received: list = []
    bus.subscribe(TOPIC_PLOT_REQUESTED, received.append)
    kernel = ReplKernel(bus)
    result = kernel.execute("plot([1, 2], [3, 4])")
    assert result.error is None
    assert len(received) == 1


def test_run_file_executes_script(tmp_path) -> None:
    """run 应执行脚本并共享命名空间."""
    script = tmp_path / "demo.py"
    script.write_text("y = 7\nprint('ran')\n", encoding="utf-8")
    kernel = ReplKernel()
    result = kernel.run_file(script)
    assert result.error is None
    assert kernel.namespace["y"] == 7
    assert "ran" in result.stdout


def test_run_file_not_found() -> None:
    """脚本不存在应返回 FileNotFoundError 文本."""
    result = ReplKernel().run_file("不存在的脚本.py")
    assert result.error is not None
    assert "FileNotFoundError" in result.error


def test_run_file_syntax_error(tmp_path) -> None:
    """脚本语法错误应返回 error."""
    script = tmp_path / "bad.py"
    script.write_text("def broken(:\n", encoding="utf-8")
    result = ReplKernel().run_file(script)
    assert result.error is not None
    assert "SyntaxError" in result.error


def test_run_file_read_error(tmp_path) -> None:
    """脚本编码非法应返回读取错误."""
    script = tmp_path / "binary.py"
    script.write_bytes(b"\xff\xfe\x00\x01")
    result = ReplKernel().run_file(script)
    assert result.error is not None
    assert "UnicodeDecodeError" in result.error


def test_whos_method_returns_table() -> None:
    """kernel.whos 应返回表格文本."""
    kernel = ReplKernel()
    kernel.execute("aa = 1")
    assert "aa" in kernel.whos()


def test_execute_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    """执行期 KeyboardInterrupt 应转为 error（monkeypatch 内置 exec 模拟中断）."""
    real_exec = builtins.exec

    def boom(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(builtins, "exec", boom)
    result = ReplKernel().execute("x = 1")
    assert result.error is not None
    assert "KeyboardInterrupt" in result.error
    monkeypatch.setattr(builtins, "exec", real_exec)
