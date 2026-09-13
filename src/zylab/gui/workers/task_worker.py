"""QThread Worker 模式：QObject 子类 + moveToThread.

典型用法（GUI 层）::

    controller = WorkerController(MyWorker(arg1, arg2))
    controller.worker.progress.connect(self.on_progress)
    controller.worker.finished_ok.connect(self.on_finished)
    controller.start()

    # 需要取消时
    controller.stop()

与 ProcessExecutor 的边界：ProcessExecutor 用于长耗时 + 崩溃隔离的求解任务；
本模块用于短耗时 + 同进程协作式取消的轻量后台操作。

注意：Worker 的 run() 中绝对禁止操作任何 QWidget；跨线程 UI 更新一律走信号。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..qt_compat import QObject, QThread, Signal

if TYPE_CHECKING:
    pass

__all__ = ["TaskWorker", "WorkerController"]


class TaskWorker(QObject):
    """可中断的 QThread Worker 基类.

    子类须实现 :meth:`run`，通过 ``is_cancelled`` 属性查询是否已请求取消、
    发送 ``progress`` / ``message`` 信号报告进度与状态、结束时发 ``finished_ok``
    或 ``failed``。Worker 对象本身不持有线程——由 :class:`WorkerController` 负责
    moveToThread 与生命周期管理。

    信号跨线程会被 Qt 自动排队到主线程接收，无需显式 ``QueuedConnection``。
    """

    progress = Signal(int)  #: 进度百分比 0-100
    message = Signal(str)  #: 状态文字提示
    finished_ok = Signal(object)  #: 成功结果（类型由子类决定）
    failed = Signal(str)  #: 错误信息

    def __init__(self) -> None:
        super().__init__()
        self._cancelled = False

    @property
    def is_cancelled(self) -> bool:
        """是否已请求取消（Worker 协作式检查此标志提前退出）."""
        return self._cancelled

    def request_cancel(self) -> None:
        """请求 Worker 停止（线程协作式，不会强杀）."""
        self._cancelled = True

    def run(self) -> None:
        """后台线程入口；子类须在此实现实际逻辑，且须检查 ``is_cancelled``."""
        raise NotImplementedError(f"{type(self).__name__} 须实现 run()")  # pragma: no cover


class WorkerController(QObject):
    """Worker 生命周期控制器：封装 QThread + moveToThread + 信号路由.

    负责创建线程、把 Worker 移入线程、启动与清理；Worker 完成后线程自动退出，
    避免 Thread 对象泄漏（finished → quit → deleteLater 链）。
    """

    def __init__(self, worker: TaskWorker, parent: QObject | None = None) -> None:
        """初始化控制器并装配 Worker 与 QThread.

        Args:
            worker: 待执行的 Worker 实例（须为 TaskWorker 子类）。
            parent: 可选父对象（Qt 所有权）。
        """
        super().__init__(parent)
        self._worker = worker
        self._thread = QThread(self)
        worker.moveToThread(self._thread)
        # 线程启动 → worker.run
        self._thread.started.connect(worker.run)
        # worker 终态 → 发退出信号
        worker.finished_ok.connect(self._thread.quit)
        worker.failed.connect(self._thread.quit)
        # 线程退出 → 清理对象（自动 deleteLater 防泄漏）
        self._thread.finished.connect(worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

    @property
    def worker(self) -> TaskWorker:
        """底层 Worker 实例（供外部连接信号）."""
        return self._worker

    def start(self) -> None:
        """启动后台线程并执行 Worker."""
        self._thread.start()

    def stop(self, wait_ms: int = 3000) -> None:
        """请求 Worker 取消 + 等待线程退出.

        Args:
            wait_ms: 等待线程退出的超时毫秒数；超时后放弃等待（线程自行退出）.
        """
        if self._thread.isRunning():
            self._worker.request_cancel()
            self._thread.quit()
            self._thread.wait(wait_ms)

    def stop_async(self) -> None:
        """仅请求取消，不阻塞等待（适合 UI 线程内调用）."""
        if self._thread.isRunning():
            self._worker.request_cancel()
            self._thread.quit()
