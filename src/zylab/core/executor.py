"""zylab 进程执行器（求解任务进程隔离）.

设计目标（对标 MATLAB 级稳定性）：
- 任务在独立 spawn 子进程执行，worker 崩溃/死循环不影响主进程；
- 进度经队列回传，任务可取消（psutil 杀进程树，含 worker 派生的孙进程）；
- 子进程异常以结构化事件回传（含 traceback 文本），主进程原样还原语义。

任务协议：:class:`TaskSpec` 的 ``target`` 为 ``"module:func"`` 全限定名（不用函数对象，
规避 lambda/闭包不可 pickle 的限制）。目标函数若声明 ``report`` 参数，worker 会注入进度回调::

    def solve(mesh_path, report=None):
        report(0.5, "装配完成")   # report(progress: float, message: str = "")
        return result
"""

from __future__ import annotations

import inspect
import logging
import multiprocessing
import multiprocessing.queues
import queue
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum, unique
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable

import psutil

from .errors import TaskCancelledError, WorkerCrashError, WorkerError

__all__ = ["EventKind", "ProcessExecutor", "TaskEvent", "TaskHandle", "TaskSpec", "TaskStatus"]

logger = logging.getLogger(__name__)


@unique
class EventKind(Enum):
    """任务事件类别."""

    STARTED = "started"
    PROGRESS = "progress"
    RESULT = "result"
    ERROR = "error"


@unique
class TaskStatus(Enum):
    """任务状态机：PENDING → RUNNING → FINISHED/FAILED/CRASHED/CANCELLED."""

    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    CRASHED = "crashed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class TaskSpec:
    """任务描述（必须整体可 pickle）.

    :param target: 目标函数全限定名 ``"module:func"``。
    :param args: 位置参数。
    :param kwargs: 关键字参数。
    """

    target: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskEvent:
    """任务事件.

    - PROGRESS: payload 为 ``(progress: float, message: str)``；
    - RESULT: payload 为任务返回值；
    - ERROR: payload 为 ``{"exc_type", "message", "traceback"}`` 字典。
    """

    task_id: str
    kind: EventKind
    payload: Any = None


if TYPE_CHECKING:
    _EventQueue = multiprocessing.queues.Queue[TaskEvent]
else:
    _EventQueue = multiprocessing.queues.Queue


def _kill_process_tree(pid: int) -> None:
    """终止进程及其全部子孙进程（不存在的进程静默跳过）."""
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    for child in proc.children(recursive=True):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            continue
    try:
        proc.kill()
    except psutil.NoSuchProcess:
        return


def _load_target(target: str) -> Callable[..., Any]:
    """解析 ``"module:func"`` 为可调用对象."""
    module_name, _, func_name = target.partition(":")
    if not module_name or not func_name:
        raise ValueError(f"target 格式应为 'module:func'，得到 {target!r}")
    func = getattr(import_module(module_name), func_name)
    if not callable(func):
        raise TypeError(f"目标不可调用: {target!r}")
    return func


def _worker_main(spec: TaskSpec, event_queue: multiprocessing.queues.Queue[TaskEvent]) -> None:
    """worker 子进程入口（spawn 目标，必须模块级可 pickle）.

    负责执行目标函数并回传事件；任何异常都转为 ERROR 事件，不让子进程带栈退出。
    """
    task_id = ""  # 事件 task_id 由主进程分配，worker 内填空串即可
    try:
        event_queue.put(TaskEvent(task_id=task_id, kind=EventKind.STARTED))
        func = _load_target(spec.target)
        kwargs = dict(spec.kwargs)
        try:
            accepts_report = "report" in inspect.signature(func).parameters
        except (TypeError, ValueError):  # 部分 C 扩展函数无签名
            accepts_report = False
        if accepts_report:

            def report(progress: float, message: str = "") -> None:
                event_queue.put(TaskEvent(task_id=task_id, kind=EventKind.PROGRESS, payload=(progress, message)))

            kwargs["report"] = report
        result = func(*spec.args, **kwargs)
        event_queue.put(TaskEvent(task_id=task_id, kind=EventKind.RESULT, payload=result))
    except BaseException as exc:  # worker 内兜底：SystemExit/KeyboardInterrupt 也要回传
        event_queue.put(
            TaskEvent(
                task_id=task_id,
                kind=EventKind.ERROR,
                payload={
                    "exc_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        )
    finally:
        event_queue.close()
        event_queue.join_thread()  # 确保缓冲区全部送达后进程再退出


class TaskHandle:
    """任务句柄：查询状态、等待结果、取消任务、监听事件.

    事件监听回调在后台监控线程中执行，GUI 场景须自行桥接到主线程（如 Qt 信号）。
    """

    def __init__(self, task_id: str) -> None:
        """初始化句柄（由 :class:`ProcessExecutor` 内部创建）."""
        self._task_id = task_id
        self._status = TaskStatus.PENDING
        self._result: Any = None
        self._error_payload: dict[str, str] | None = None
        self._done = threading.Event()
        self._lock = threading.RLock()
        self._listeners: list[Callable[[TaskEvent], None]] = []

    @property
    def task_id(self) -> str:
        """任务唯一标识."""
        return self._task_id

    @property
    def status(self) -> TaskStatus:
        """当前任务状态."""
        with self._lock:
            return self._status

    @property
    def done(self) -> bool:
        """是否已到达终态（FINISHED/FAILED/CRASHED/CANCELLED）."""
        return self._done.is_set()

    def add_listener(self, callback: Callable[[TaskEvent], None]) -> None:
        """注册事件监听（在后台线程触发；回调异常仅记录日志）."""
        with self._lock:
            self._listeners.append(callback)

    def wait(self, timeout: float | None = None) -> Any:
        """阻塞等待终态并返回结果.

        :param timeout: 超时秒数，None 表示无限等待。
        :returns: 任务返回值。
        :raises TimeoutError: 超时未完成。
        :raises WorkerError: worker 内任务抛异常（含原 traceback 文本）。
        :raises WorkerCrashError: worker 进程崩溃。
        :raises TaskCancelledError: 任务已取消。
        """
        if not self._done.wait(timeout):
            raise TimeoutError(f"任务 {self._task_id} 等待超时（{timeout}s）")
        with self._lock:
            status = self._status
            if status is TaskStatus.FINISHED:
                return self._result
            if status is TaskStatus.CANCELLED:
                raise TaskCancelledError(f"任务 {self._task_id} 已取消")
            if status is TaskStatus.CRASHED:
                raise WorkerCrashError(f"任务 {self._task_id} 所在 worker 进程崩溃（未回传结果）")
            payload = self._error_payload or {}
            raise WorkerError(
                f"任务 {self._task_id} 执行失败: {payload.get('exc_type', '未知错误')}: "
                f"{payload.get('message', '')}\n{payload.get('traceback', '')}"
            )

    def _on_event(self, event: TaskEvent) -> None:
        """处理事件并分发监听（内部方法，由监控线程调用）."""
        with self._lock:
            if event.kind is EventKind.STARTED:
                self._status = TaskStatus.RUNNING
            elif event.kind is EventKind.RESULT:
                self._result = event.payload
                self._status = TaskStatus.FINISHED
                self._done.set()
            elif event.kind is EventKind.ERROR:
                self._error_payload = event.payload
                self._status = TaskStatus.FAILED
                self._done.set()
            listeners = list(self._listeners)
        for callback in listeners:
            try:
                callback(event)
            except Exception:
                logger.warning("任务事件监听回调失败: task=%s", self._task_id, exc_info=True)

    def _mark_terminal(self, status: TaskStatus) -> None:
        """标记终态（内部方法；已被取消的任务保持 CANCELLED 不被覆盖）."""
        with self._lock:
            if self._status is TaskStatus.CANCELLED:
                self._done.set()
                return
            self._status = status
            self._done.set()

    def _cancel(self) -> None:
        """标记取消（内部方法，由 executor 在杀进程前调用）."""
        with self._lock:
            if not self._done.is_set():
                self._status = TaskStatus.CANCELLED
                self._done.set()


class ProcessExecutor:
    """进程执行器：提交任务到独立 spawn 子进程执行.

    用法::

        executor = ProcessExecutor()
        handle = executor.submit(TaskSpec(target="mypkg.solver:solve", args=(path,)))
        handle.add_listener(lambda ev: print(ev.kind, ev.payload))
        result = handle.wait(timeout=60)
        executor.shutdown()
    """

    def __init__(self) -> None:
        """初始化执行器（spawn 上下文，跨平台行为一致）."""
        self._ctx = multiprocessing.get_context("spawn")
        self._tasks: dict[str, tuple[TaskHandle, multiprocessing.process.BaseProcess]] = {}
        self._lock = threading.RLock()
        self._shutdown = False

    def submit(self, spec: TaskSpec) -> TaskHandle:
        """提交任务，立即 spawn worker 进程并返回句柄.

        :raises WorkerError: 执行器已关闭。
        """
        with self._lock:
            if self._shutdown:
                raise WorkerError("执行器已关闭，无法提交任务")
            task_id = uuid.uuid4().hex[:12]
            handle = TaskHandle(task_id)
            event_queue: multiprocessing.queues.Queue[TaskEvent] = self._ctx.Queue()
            process = self._ctx.Process(target=_worker_main, args=(spec, event_queue), daemon=True)
            process.start()
            self._tasks[task_id] = (handle, process)
        watcher = threading.Thread(
            target=self._watch, args=(handle, process, event_queue), name=f"zylab-watch-{task_id}", daemon=True
        )
        watcher.start()
        logger.debug("任务已提交: %s -> %s (pid=%s)", task_id, spec.target, process.pid)
        return handle

    def cancel(self, handle: TaskHandle) -> None:
        """取消任务：杀进程树并标记 CANCELLED（幂等，已完成任务调用无效果）."""
        with self._lock:
            entry = self._tasks.get(handle.task_id)
        if entry is None or handle.done:
            return
        _, process = entry
        handle._cancel()
        if process.pid is not None:
            _kill_process_tree(process.pid)
        logger.info("任务已取消: %s", handle.task_id)

    def shutdown(self, *, cancel_running: bool = True) -> None:
        """关闭执行器；``cancel_running`` 为 True 时取消所有未完成任务."""
        with self._lock:
            self._shutdown = True
            entries = list(self._tasks.values())
            self._tasks.clear()
        for handle, process in entries:
            if cancel_running and not handle.done:
                handle._cancel()
                if process.pid is not None:
                    _kill_process_tree(process.pid)
            process.join(timeout=5)
        logger.debug("执行器已关闭")

    def _watch(
        self,
        handle: TaskHandle,
        process: multiprocessing.process.BaseProcess,
        event_queue: _EventQueue,
    ) -> None:
        """监控线程：消费事件队列直到任务终态 + 进程退出；进程崩溃无终态时标记 CRASHED."""
        terminated = False
        while True:
            try:
                event = event_queue.get(timeout=0.1)
            except queue.Empty:
                event = None
            if event is not None:
                handle._on_event(event)
                if event.kind in (EventKind.RESULT, EventKind.ERROR):
                    terminated = True
            if handle.done and not process.is_alive():
                break
            if terminated and not process.is_alive():
                break
            if not process.is_alive() and event_queue.empty():
                break
        process.join(timeout=1)
        exitcode = process.exitcode
        event_queue.close()
        with self._lock:
            self._tasks.pop(handle.task_id, None)
        if not handle.done:
            logger.error("worker 进程崩溃: task=%s exitcode=%s", handle.task_id, exitcode)
            handle._mark_terminal(TaskStatus.CRASHED)
        elif handle.status is TaskStatus.RUNNING:
            # 有 STARTED 但缺终态事件即退出，按崩溃处理
            logger.error("worker 进程异常退出: task=%s exitcode=%s", handle.task_id, exitcode)
            handle._mark_terminal(TaskStatus.CRASHED)
