"""executor 测试目标函数集（须为模块级函数，保证 spawn 子进程可导入）."""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

__all__ = ["SAMPLE_TEMPLATE", "add", "crash", "echo_report", "failing", "long_running"]

#: 模板插件 entry point 测试用示例（工厂解析为字典）
SAMPLE_TEMPLATE = {
    "id": "plugin.sample",
    "name": "插件示例模板",
    "nodes": [
        {"id": "model", "type": "example.truss2_two_bar"},
        {"id": "solve", "type": "analysis.static", "inputs": {"model": "model.model"}},
    ],
}


def add(a: int, b: int) -> int:
    """返回两数之和."""
    return a + b


def echo_report(report: Optional[Callable[[float, str], None]] = None) -> str:
    """上报两次进度后返回 ok."""
    if report is not None:
        report(0.5, "半程")
        report(1.0, "完成")
    return "ok"


def failing() -> None:
    """故意抛出 ValueError."""
    raise ValueError("目标故障")


def crash() -> None:
    """模拟进程崩溃（直接退出，不抛异常）."""
    os._exit(1)


def long_running(seconds: float = 30.0, report: Optional[Callable[[float, str], None]] = None) -> str:
    """长跑任务（用于取消测试），周期性上报进度."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if report is not None:
            report(0.1, "运行中")
        time.sleep(0.05)
    return "done"


def take_any(value: Any) -> Any:
    """原样返回入参（无 report 参数声明的目标）."""
    return value
