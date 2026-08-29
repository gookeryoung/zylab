"""core.executor 进程执行器测试.

涉及 spawn 子进程，目标函数必须来自可 import 的模块级符号（tests._targets）。
"""

from __future__ import annotations

import time

import pytest

from zylab.core.errors import TaskCancelledError, WorkerCrashError, WorkerError
from zylab.core.executor import EventKind, ProcessExecutor, TaskSpec


def test_executor_submit_and_wait() -> None:
    """正常任务应返回正确结果."""
    executor = ProcessExecutor()
    handle = executor.submit(TaskSpec(target="tests._targets:add", args=(2, 3)))
    result = handle.wait(timeout=10)
    assert result == 5
    executor.shutdown()


def test_executor_progress_events() -> None:
    """report 参数应触发 PROGRESS 事件."""
    executor = ProcessExecutor()
    progress_events = []

    def listener(event):
        if event.kind is EventKind.PROGRESS:
            progress_events.append(event.payload)

    handle = executor.submit(TaskSpec(target="tests._targets:echo_report"))
    handle.add_listener(listener)
    result = handle.wait(timeout=10)
    assert result == "ok"
    assert len(progress_events) == 2
    assert progress_events[0] == (0.5, "半程")
    assert progress_events[1] == (1.0, "完成")
    executor.shutdown()


def test_executor_failing_task() -> None:
    """worker 内抛异常应包装为 WorkerError."""
    executor = ProcessExecutor()
    handle = executor.submit(TaskSpec(target="tests._targets:failing"))
    with pytest.raises(WorkerError, match="目标故障"):
        handle.wait(timeout=10)
    assert handle.status.value == "failed"
    executor.shutdown()


def test_executor_crash() -> None:
    """worker 进程崩溃应标记为 CRASHED."""
    executor = ProcessExecutor()
    handle = executor.submit(TaskSpec(target="tests._targets:crash"))
    with pytest.raises(WorkerCrashError):
        handle.wait(timeout=10)
    assert handle.status.value == "crashed"
    executor.shutdown()


def test_executor_cancel() -> None:
    """取消应终止运行中的任务."""
    executor = ProcessExecutor()
    handle = executor.submit(TaskSpec(target="tests._targets:long_running", args=(30.0,)))
    time.sleep(0.3)  # 给 worker 启动时间
    executor.cancel(handle)
    with pytest.raises(TaskCancelledError):
        handle.wait(timeout=5)
    executor.shutdown()


def test_executor_timeout() -> None:
    """wait 超时未终态应抛 TimeoutError."""
    executor = ProcessExecutor()
    handle = executor.submit(TaskSpec(target="tests._targets:long_running", args=(10.0,)))
    with pytest.raises(TimeoutError):
        handle.wait(timeout=0.1)
    executor.cancel(handle)
    executor.shutdown()


def test_executor_reuse_shutdown() -> None:
    """shutdown 后应不再接受新任务."""
    executor = ProcessExecutor()
    executor.shutdown()
    with pytest.raises(WorkerError, match="已关闭"):
        executor.submit(TaskSpec(target="tests._targets:add", args=(1, 1)))


def test_cancel_unknown_handle_noop() -> None:
    """取消未知句柄应静默返回."""
    from zylab.core.executor import TaskHandle

    executor = ProcessExecutor()
    executor.cancel(TaskHandle("unknown"))  # 不抛异常
    executor.shutdown()


def test_cancel_finished_task_noop() -> None:
    """取消已完成任务不改变其终态."""
    executor = ProcessExecutor()
    handle = executor.submit(TaskSpec(target="tests._targets:add", args=(1, 2)))
    assert handle.wait(timeout=10) == 3
    executor.cancel(handle)
    assert handle.status.value == "finished"
    executor.shutdown()


def test_shutdown_cancels_running() -> None:
    """shutdown 应取消所有运行中的任务."""
    executor = ProcessExecutor()
    handle = executor.submit(TaskSpec(target="tests._targets:long_running", args=(30.0,)))
    executor.shutdown()
    with pytest.raises(TaskCancelledError):
        handle.wait(timeout=5)


# ---------- _worker_main 直接驱动测试（私有函数，覆盖 worker 内分支，不经 spawn 进程） ----------


class _StubQueue:
    """模拟 multiprocessing.Queue 接口的桩（put/close/join_thread）."""

    def __init__(self) -> None:
        self.items: list = []

    def put(self, item) -> None:
        self.items.append(item)

    def close(self) -> None:
        pass

    def join_thread(self) -> None:
        pass


def test_worker_main_success() -> None:
    """正常执行应回传 STARTED + RESULT 事件."""
    from zylab.core.executor import _worker_main

    q = _StubQueue()
    _worker_main(TaskSpec(target="tests._targets:add", args=(2, 3)), q)  # type: ignore[arg-type]
    kinds = [e.kind for e in q.items]
    assert kinds == [EventKind.STARTED, EventKind.RESULT]
    assert q.items[1].payload == 5


def test_worker_main_report_injection() -> None:
    """目标声明 report 参数时应注入进度回调."""
    from zylab.core.executor import _worker_main

    q = _StubQueue()
    _worker_main(TaskSpec(target="tests._targets:echo_report"), q)  # type: ignore[arg-type]
    kinds = [e.kind for e in q.items]
    assert kinds == [EventKind.STARTED, EventKind.PROGRESS, EventKind.PROGRESS, EventKind.RESULT]


def test_worker_main_target_error() -> None:
    """目标抛异常应回传 ERROR 事件且含异常类型."""
    from zylab.core.executor import _worker_main

    q = _StubQueue()
    _worker_main(TaskSpec(target="tests._targets:failing"), q)  # type: ignore[arg-type]
    assert q.items[-1].kind is EventKind.ERROR
    assert q.items[-1].payload["exc_type"] == "ValueError"
    assert "目标故障" in q.items[-1].payload["message"]


def test_worker_main_invalid_target_format() -> None:
    """非法 target 格式应回传 ERROR 事件."""
    from zylab.core.executor import _worker_main

    q = _StubQueue()
    _worker_main(TaskSpec(target="no_colon_at_all"), q)  # type: ignore[arg-type]
    assert q.items[-1].kind is EventKind.ERROR
    assert q.items[-1].payload["exc_type"] == "ValueError"


def test_worker_main_not_callable() -> None:
    """target 指向不可调用对象应回传 ERROR 事件."""
    from zylab.core.executor import _worker_main

    q = _StubQueue()
    _worker_main(TaskSpec(target="tests._targets:__all__"), q)  # type: ignore[arg-type]
    assert q.items[-1].kind is EventKind.ERROR
    assert q.items[-1].payload["exc_type"] == "TypeError"


def test_kill_process_tree_no_such_process() -> None:
    """对不存在的 pid 杀进程树应静默返回（私有函数，直接测试）."""
    from zylab.core.executor import _kill_process_tree

    _kill_process_tree(999999999)  # 不抛异常
