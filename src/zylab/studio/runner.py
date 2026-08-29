"""工作流编排执行：按拓扑序驱动 ProcessExecutor 逐节点执行.

- 缓存语义：UP_TO_DATE 节点命中结果缓存自动跳过，仅运行过期（READY/FAILED）节点；
- 级联：:meth:`WorkflowRunner.run_node` 自动补齐目标节点的全部过期上游；
- 失败中止：任一节点失败后清空队列（下游依赖其输出，不可继续）；
- 线程模型：节点事件监听在 executor 监控线程触发，图状态迁移在该线程内完成
  （GIL 保证单字段赋值的读写安全）；GUI 侧经 Qt 信号桥接到主线程后仅读取状态刷新视图。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from zylab.core.errors import WorkerError
from zylab.core.executor import EventKind, ProcessExecutor, TaskEvent, TaskHandle, TaskSpec

from .errors import StudioError
from .graph import WorkflowGraph

__all__ = ["NodeRunEvent", "WorkflowRunner"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeRunEvent:
    """节点级运行事件.

    :param node_id: 节点 id。
    :param kind: STARTED / PROGRESS（payload=(progress, message)）/
        RESULT（payload=节点输出对象）/ ERROR（payload=错误消息字符串）。
    :param payload: 事件载荷。
    """

    node_id: str
    kind: EventKind
    payload: Any = None


class WorkflowRunner:
    """工作流编排执行器（与图一一对应；运行期间禁止图结构/参数变更）."""

    def __init__(self, graph: WorkflowGraph, executor: ProcessExecutor | None = None) -> None:
        """初始化编排器；``executor`` 缺省时自建进程执行器（shutdown 时一并关闭）."""
        self._graph = graph
        self._executor = executor if executor is not None else ProcessExecutor()
        self._owns_executor = executor is None
        self._queue: list[str] = []
        self._on_event: Callable[[NodeRunEvent], None] = lambda _event: None
        self._handle: TaskHandle | None = None
        self._current: str | None = None
        self._lock = threading.RLock()

    @property
    def running(self) -> bool:
        """是否有节点在执行."""
        return self._handle is not None

    @property
    def graph(self) -> WorkflowGraph:
        """编排的工作流图."""
        return self._graph

    def run_node(self, node_id: str, on_event: Callable[[NodeRunEvent], None]) -> None:
        """级联运行目标节点及其全部过期上游（拓扑序；已 UP_TO_DATE 的自动跳过）."""
        targets = {node_id, *self._graph.ancestors(node_id)}
        queue = [nid for nid in self._graph.execution_order() if nid in targets and self._graph.node(nid).needs_run]
        self._start(queue, on_event)

    def run_all(self, on_event: Callable[[NodeRunEvent], None]) -> None:
        """运行全部过期节点（拓扑序）."""
        queue = [nid for nid in self._graph.execution_order() if self._graph.node(nid).needs_run]
        self._start(queue, on_event)

    def cancel(self) -> None:
        """取消当前运行：清空队列并终止 worker 进程，运行中节点回到可运行态."""
        with self._lock:
            self._queue.clear()
            handle = self._handle
            self._handle = None
            current = self._current
            self._current = None
        if handle is not None:
            self._executor.cancel(handle)
        if current is not None:
            self._graph.mark_reset(current)

    def shutdown(self) -> None:
        """取消运行并关闭执行器（仅当执行器由本 runner 自建时）."""
        self.cancel()
        if self._owns_executor:
            self._executor.shutdown()

    # ------------------------------------------------------------------ 内部

    def _start(self, queue: list[str], on_event: Callable[[NodeRunEvent], None]) -> None:
        """登记执行队列并启动首个节点."""
        with self._lock:
            if self._handle is not None:
                raise StudioError("已有运行中的工作流，请先完成或取消")
            self._queue = queue
            self._on_event = on_event
        self._submit_next()

    def _submit_next(self) -> None:
        """弹出队首节点并提交进程执行器；队列空时结束运行.

        STARTED 先于 submit 派发：保证同步执行器（事件在 add_listener 内重入）下
        事件序仍为 STARTED -> PROGRESS -> RESULT。
        """
        with self._lock:
            if not self._queue:
                self._handle = None
                self._current = None
                return
            node_id = self._queue.pop(0)
            node = self._graph.node(node_id)
            inputs = {port: self._graph.node(ref.partition(".")[0]).result for port, ref in node.inputs.items()}
            self._graph.mark_running(node_id)
            self._current = node_id
            started_at = time.perf_counter()
            spec = TaskSpec(target=node.spec.target, args=(inputs, dict(node.params)))
            self._emit(NodeRunEvent(node_id, EventKind.STARTED))
            try:
                self._handle = self._executor.submit(spec)
            except WorkerError as exc:  # 执行器已关闭
                self._graph.mark_failed(node_id, str(exc))
                self._emit(NodeRunEvent(node_id, EventKind.ERROR, str(exc)))
                self._queue.clear()
                self._handle = None
                self._current = None
                return
            self._handle.add_listener(lambda event: self._on_task_event(node_id, started_at, event))

    def _on_task_event(self, node_id: str, started_at: float, event: TaskEvent) -> None:
        """executor 事件处理（监控线程）：PROGRESS 透传；RESULT 登记缓存并续跑；ERROR 中止队列."""
        if event.kind is EventKind.PROGRESS:
            self._emit(NodeRunEvent(node_id, EventKind.PROGRESS, event.payload))
            return
        if event.kind is EventKind.RESULT:
            self._graph.mark_result(node_id, event.payload, time.perf_counter() - started_at)
            self._emit(NodeRunEvent(node_id, EventKind.RESULT, event.payload))
            with self._lock:
                self._handle = None
            self._submit_next()
            return
        if event.kind is EventKind.ERROR:
            payload = event.payload
            message = f"{payload.get('exc_type', '错误')}: {payload.get('message', '')}"
            self._graph.mark_failed(node_id, message)
            self._emit(NodeRunEvent(node_id, EventKind.ERROR, message))
            with self._lock:
                self._queue.clear()
                self._handle = None
                self._current = None

    def _emit(self, event: NodeRunEvent) -> None:
        """派发节点事件（回调属外部代码，异常仅记录不中断编排）."""
        try:
            self._on_event(event)
        except Exception:
            logger.warning("节点事件回调失败: %s", event.node_id, exc_info=True)
