"""QRunnable 批量调度：QThreadPool 共享池 + 协作式取消.

QThreadPool 适合短任务（秒级）批量提交——内部维护一个线程池，避免反复创建
QThread 的开销。与 QThread Worker（见 :mod:`task_worker`）的边界：

| 维度                | QThread Worker                | QRunnable + QThreadPool         |
|---------------------|-------------------------------|---------------------------------|
| 生命周期            | 显式 start/stop，一对一绑定   | 提交即忘，自动管理              |
| 取消语义            | 协作式 request_cancel         | 协作式（检查 cancel_requested）|
| 适合任务            | 单个中等任务（如网格加载）   | 批量短任务（如参数化计算逐行）  |
| 线程创建开销        | 每次新建 QThread              | 池化复用                        |

``worker_pool`` 是模块级共享实例，默认用 QThreadPool.globalInstance()。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Optional

from ..qt_compat import QObject, QRunnable, QThreadPool, Signal

logger = logging.getLogger(__name__)

__all__ = ["WorkerTask", "WorkerThreadPool", "worker_pool"]


class WorkerTask(QRunnable):
    """单个可追踪的 Runnable，支持协作式取消与完成回调.

    QRunnable 本身不是 QObject，无法发信号——完成/失败通过 callable 回调回传。
    回调会在 worker 线程被调用；如需要在主线程执行，由 :class:`WorkerThreadPool`
    的 bridge 信号（跨线程自动排队）保证。

    Args:
        func: 要执行的 callable。若 ``capture_cancel=True``，func 会收到一个
            ``cancel_requested: Callable[[], bool]`` 参数用于查询取消状态。
        capture_cancel: 是否注入取消查询 callable。
    """

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        capture_cancel: bool = False,
    ) -> None:
        super().__init__()
        self._func = func
        self._capture_cancel = capture_cancel
        self._cancelled = False
        self._result: Any = None
        self._error: Optional[BaseException] = None
        self._on_finished: Optional[Callable[[Any], None]] = None
        self._on_failed: Optional[Callable[[BaseException], None]] = None

    def set_callbacks(
        self,
        on_finished: Callable[[Any], None] | None = None,
        on_failed: Callable[[BaseException], None] | None = None,
    ) -> None:
        """设置完成/失败回调。"""
        self._on_finished = on_finished
        self._on_failed = on_failed

    def request_cancel(self) -> None:
        """请求取消（协作式，仅当 func 检查 ``cancel_requested`` 时才生效）."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """返回取消请求状态（供 func 内查询）."""
        return self._cancelled

    @property
    def result(self) -> Any:
        """func 返回值（run() 完成后才有意义）."""
        return self._result

    @property
    def error(self) -> Optional[BaseException]:
        """捕获的异常（若有）."""
        return self._error

    def run(self) -> None:  # Qt 命名约定
        """执行 func 并捕获异常；结束后触发回调."""
        try:
            if self._capture_cancel:
                self._result = self._func(self.is_cancelled)
            else:
                self._result = self._func()
        except BaseException as exc:
            self._error = exc
            logger.debug("WorkerTask 执行失败: %s", exc, exc_info=True)
            if self._on_failed is not None:
                self._on_failed(exc)
            return
        if self._on_finished is not None:
            self._on_finished(self._result)


class _PoolBridge(QObject):
    """跨线程信号桥接：WorkerTask 在 worker 线程 emit，主线程槽接收."""

    task_finished = Signal(object)  #: WorkerTask 对象（携带 result / error）


class WorkerThreadPool:
    """WorkerTask 调度器：共享 QThreadPool + 批量提交 + 协作式取消."""

    def __init__(self, pool: QThreadPool | None = None) -> None:
        """初始化调度器.

        Args:
            pool: 底层 QThreadPool；为 None 时使用 QThreadPool.globalInstance()。
        """
        self._pool = pool if pool is not None else QThreadPool.globalInstance()
        self._bridge = _PoolBridge()
        self._active: set[WorkerTask] = set()

    @property
    def active_count(self) -> int:
        """当前活跃任务数."""
        return len(self._active)

    # --- 批量提交 ---

    def submit_many(
        self,
        tasks: list[WorkerTask],
        *,
        on_each: Callable[[WorkerTask], None] | None = None,
        on_all_done: Callable[[], None] | None = None,
    ) -> None:
        """批量提交 WorkerTask.

        Args:
            tasks: WorkerTask 列表。
            on_each: 每个任务结束（成功或失败）时调用；在主线程执行。
            on_all_done: 所有任务都结束时调用；在主线程执行。
        """
        pending = list(tasks)
        remaining = len(pending)

        def _on_bridge(task: WorkerTask) -> None:
            nonlocal remaining
            self._active.discard(task)
            if on_each is not None:
                on_each(task)
            remaining -= 1
            if remaining == 0 and on_all_done is not None:
                on_all_done()

        self._bridge.task_finished.connect(_on_bridge)

        for task in pending:
            task.setAutoDelete(False)
            self._active.add(task)
            task.set_callbacks(
                on_finished=lambda _r, _t=task: self._bridge.task_finished.emit(_t),
                on_failed=lambda _e, _t=task: self._bridge.task_finished.emit(_t),
            )
            self._pool.start(task)

    # --- 单个提交 ---

    def submit_one(
        self,
        func: Callable[..., Any],
        *,
        capture_cancel: bool = False,
        on_finished: Callable[[Any], None] | None = None,
        on_failed: Callable[[BaseException], None] | None = None,
    ) -> WorkerTask:
        """提交单个函数并返回 WorkerTask 句柄（可 ``request_cancel``）.

        Args:
            func: 要执行的 callable。
            capture_cancel: 是否注入 ``cancel_requested`` 参数。
            on_finished: 成功回调（主线程）。
            on_failed: 失败回调（主线程）。

        Returns:
            WorkerTask 句柄。
        """
        task = WorkerTask(func, capture_cancel=capture_cancel)
        task.setAutoDelete(False)
        self._active.add(task)

        def _on_bridge(t: WorkerTask) -> None:
            self._active.discard(t)
            if t.error is not None and on_failed is not None:
                on_failed(t.error)
            elif t.error is None and on_finished is not None:
                on_finished(t.result)

        self._bridge.task_finished.connect(_on_bridge)
        task.set_callbacks(
            on_finished=lambda _r: self._bridge.task_finished.emit(task),
            on_failed=lambda _e: self._bridge.task_finished.emit(task),
        )
        self._pool.start(task)
        return task

    # --- 取消 ---

    def cancel_all(self) -> None:
        """向所有活跃任务发取消请求（协作式，不会强杀）."""
        for task in list(self._active):
            task.request_cancel()


# 模块级共享实例——直接用全局线程池，大多数场景不需要自定义并发数
worker_pool: WorkerThreadPool = WorkerThreadPool()
