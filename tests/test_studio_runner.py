"""studio.runner 编排执行测试：拓扑序执行、缓存命中、级联运行、失败中止、取消."""

from __future__ import annotations

import importlib
import inspect
import threading

import pytest

from zylab.core.executor import EventKind, TaskEvent, TaskSpec
from zylab.studio import (
    NodeRunEvent,
    NodeState,
    StudioError,
    Template,
    WorkflowGraph,
    WorkflowRunner,
)

__all__ = []


class _SyncHandle:
    """同步任务句柄：add_listener 时立即执行目标并回放事件（不起子进程）."""

    def __init__(self, spec: TaskSpec) -> None:
        self._spec = spec
        self.task_id = "sync"

    def add_listener(self, callback) -> None:
        """执行目标函数并按序回放 STARTED/PROGRESS/RESULT|ERROR 事件."""
        module_name, _, attr = self._spec.target.partition(":")
        func = getattr(importlib.import_module(module_name), attr)
        callback(TaskEvent(task_id="sync", kind=EventKind.STARTED))
        kwargs = dict(self._spec.kwargs)
        if "report" in inspect.signature(func).parameters:
            kwargs["report"] = lambda progress, message="": callback(
                TaskEvent(task_id="sync", kind=EventKind.PROGRESS, payload=(progress, message))
            )
        try:
            result = func(*self._spec.args, **kwargs)
        except Exception as exc:
            callback(
                TaskEvent(
                    task_id="sync",
                    kind=EventKind.ERROR,
                    payload={"exc_type": type(exc).__name__, "message": str(exc)},
                )
            )
        else:
            callback(TaskEvent(task_id="sync", kind=EventKind.RESULT, payload=result))


class _SyncExecutor:
    """同步执行器替身（与 ProcessExecutor 同接口，任务立即完成）."""

    def __init__(self) -> None:
        self.cancelled = False
        self.shutdown_called = False

    def submit(self, spec: TaskSpec) -> _SyncHandle:
        """返回同步句柄（执行推迟到 add_listener）."""
        return _SyncHandle(spec)

    def cancel(self, handle: _SyncHandle) -> None:
        """记录取消."""
        self.cancelled = True

    def shutdown(self, *, cancel_running: bool = True) -> None:
        """记录关闭."""
        self.shutdown_called = True


class _HangHandle:
    """永不完成的句柄."""

    task_id = "hang"

    def add_listener(self, callback) -> None:
        """注册监听但永不触发."""


class _HangExecutor(_SyncExecutor):
    """挂起执行器替身：submit 后不发任何事件（用于运行中状态与取消测试）."""

    def submit(self, spec: TaskSpec) -> _HangHandle:
        """返回永不完成的句柄."""
        return _HangHandle()


def _truss_graph(analysis_type: str = "analysis.static", analysis_params: dict | None = None) -> WorkflowGraph:
    """构造两节点桁架模板图."""
    solve_node: dict = {"id": "solve", "type": analysis_type, "inputs": {"model": "model.model"}}
    if analysis_params:
        solve_node["params"] = analysis_params
    template = Template.from_dict(
        {
            "id": "t.truss",
            "name": "桁架",
            "nodes": [{"id": "model", "type": "example.truss2_two_bar"}, solve_node],
        }
    )
    return WorkflowGraph(template)


def _combo_graph() -> WorkflowGraph:
    """构造组合模板图（model -> static / modal）."""
    template = Template.from_dict(
        {
            "id": "t.combo",
            "name": "组合",
            "nodes": [
                {"id": "model", "type": "example.cantilever_q4", "params": {"nx": 4, "ny": 2}},
                {"id": "static", "type": "analysis.static", "inputs": {"model": "model.model"}},
                {"id": "modal", "type": "analysis.modal", "inputs": {"model": "model.model"}},
            ],
        }
    )
    return WorkflowGraph(template)


def _kinds(events: list[NodeRunEvent]) -> list[tuple[str, EventKind]]:
    """提取事件的 (节点, 类型) 序列."""
    return [(event.node_id, event.kind) for event in events]


def _steps(events: list[NodeRunEvent]) -> list[tuple[str, EventKind]]:
    """提取 STARTED/RESULT/ERROR 骨架序列（滤掉 PROGRESS 心跳）."""
    return [(event.node_id, event.kind) for event in events if event.kind is not EventKind.PROGRESS]


class TestRunAll:
    """拓扑序执行与缓存."""

    def test_executes_in_topological_order(self) -> None:
        """两节点链按 model -> solve 顺序执行，终态 UP_TO_DATE."""
        graph = _truss_graph()
        runner = WorkflowRunner(graph, executor=_SyncExecutor())
        events: list[NodeRunEvent] = []
        runner.run_all(events.append)
        assert _steps(events) == [
            ("model", EventKind.STARTED),
            ("model", EventKind.RESULT),
            ("solve", EventKind.STARTED),
            ("solve", EventKind.RESULT),
        ]
        assert all(node.state is NodeState.UP_TO_DATE for node in graph.nodes())
        assert graph.node("solve").elapsed >= 0.0
        assert not runner.running

    def test_cache_hit_skips_execution(self) -> None:
        """二次 run_all 全部命中缓存，无任何事件."""
        graph = _truss_graph()
        runner = WorkflowRunner(graph, executor=_SyncExecutor())
        events: list[NodeRunEvent] = []
        runner.run_all(events.append)
        events.clear()
        runner.run_all(events.append)
        assert events == []

    def test_param_change_triggers_rerun(self) -> None:
        """参数变更级联失效后重跑."""
        graph = _truss_graph()
        runner = WorkflowRunner(graph, executor=_SyncExecutor())
        runner.run_all(lambda _event: None)
        graph.set_param("model", "rise", 0.6)
        events: list[NodeRunEvent] = []
        runner.run_all(events.append)
        assert ("model", EventKind.RESULT) in _kinds(events)
        assert ("solve", EventKind.RESULT) in _kinds(events)

    def test_progress_forwarded(self) -> None:
        """节点 report 进度经 PROGRESS 事件透传."""
        graph = _truss_graph("analysis.harmonic", {"f_max": 2.0, "n_freq": 10})
        runner = WorkflowRunner(graph, executor=_SyncExecutor())
        events: list[NodeRunEvent] = []
        runner.run_all(events.append)
        progress = [event for event in events if event.kind is EventKind.PROGRESS]
        assert progress  # 谐响应逐频点上报
        assert any(event.node_id == "solve" for event in progress)

    def test_event_callback_exception_swallowed(self) -> None:
        """事件回调异常仅记录，不中断编排."""

        def bad_listener(event: NodeRunEvent) -> None:
            """对 PROGRESS 抛异常的监听器."""
            if event.kind is EventKind.PROGRESS:
                raise RuntimeError("回调故障")

        graph = _truss_graph("analysis.harmonic", {"f_max": 2.0, "n_freq": 10})
        runner = WorkflowRunner(graph, executor=_SyncExecutor())
        runner.run_all(bad_listener)
        assert graph.node("solve").state is NodeState.UP_TO_DATE


class TestRunNode:
    """级联运行目标节点."""

    def test_cascades_only_stale_branch(self) -> None:
        """组合图中仅重跑失效分支（model/static 命中缓存）."""
        graph = _combo_graph()
        runner = WorkflowRunner(graph, executor=_SyncExecutor())
        runner.run_all(lambda _event: None)
        graph.set_param("modal", "n_modes", 8)
        events: list[NodeRunEvent] = []
        runner.run_node("modal", events.append)
        assert _steps(events) == [("modal", EventKind.STARTED), ("modal", EventKind.RESULT)]
        assert graph.node("static").state is NodeState.UP_TO_DATE


class TestFailure:
    """失败中止."""

    def test_error_stops_queue(self) -> None:
        """节点失败：ERROR 事件 + 节点 FAILED + 队列清空（下游不再运行）."""
        graph = _truss_graph("analysis.modal", {"n_modes": 50})  # 50 阶超过 6 自由度，求解必失败
        runner = WorkflowRunner(graph, executor=_SyncExecutor())
        events: list[NodeRunEvent] = []
        runner.run_all(events.append)
        assert _steps(events)[-1] == ("solve", EventKind.ERROR)
        assert graph.node("solve").state is NodeState.FAILED
        assert not runner.running
        # 修复参数后可重跑（桁架自由自由度为 2，阶数须取 1）
        graph.set_param("solve", "n_modes", 1)
        events.clear()
        runner.run_all(events.append)
        assert _steps(events) == [("solve", EventKind.STARTED), ("solve", EventKind.RESULT)]
        assert graph.node("solve").state is NodeState.UP_TO_DATE


class TestCancelAndLifecycle:
    """取消与生命周期."""

    def test_run_while_running_rejected(self) -> None:
        """运行中重复提交抛 StudioError."""
        graph = _truss_graph()
        runner = WorkflowRunner(graph, executor=_HangExecutor())
        runner.run_all(lambda _event: None)
        assert runner.running
        with pytest.raises(StudioError, match="运行中"):
            runner.run_all(lambda _event: None)
        runner.cancel()

    def test_cancel_resets_running_node(self) -> None:
        """取消：队列清空、executor 收到取消、运行中节点回到 READY."""
        graph = _truss_graph()
        executor = _HangExecutor()
        runner = WorkflowRunner(graph, executor=executor)
        runner.run_all(lambda _event: None)
        assert graph.node("model").state is NodeState.RUNNING
        runner.cancel()
        assert executor.cancelled
        assert not runner.running
        assert graph.node("model").state is NodeState.READY

    def test_cancel_when_idle_is_noop(self) -> None:
        """空闲时取消无副作用."""
        graph = _truss_graph()
        executor = _HangExecutor()
        runner = WorkflowRunner(graph, executor=executor)
        runner.cancel()
        assert not executor.cancelled
        assert not runner.running

    def test_shutdown_closes_only_owned_executor(self) -> None:
        """注入的执行器不被关闭；自建执行器被关闭."""
        graph = _truss_graph()
        executor = _SyncExecutor()
        WorkflowRunner(graph, executor=executor).shutdown()
        assert not executor.shutdown_called
        WorkflowRunner(_truss_graph()).shutdown()  # 自建 ProcessExecutor，无任务直接关闭

    def test_graph_property(self) -> None:
        """graph 属性返回编排的图."""
        graph = _truss_graph()
        assert WorkflowRunner(graph, executor=_SyncExecutor()).graph is graph

    def test_submit_after_executor_shutdown(self) -> None:
        """执行器已关闭时提交失败：节点 FAILED + ERROR 事件."""
        graph = _truss_graph()
        runner = WorkflowRunner(graph)  # 自建执行器
        runner.shutdown()
        events: list[NodeRunEvent] = []
        runner.run_all(events.append)
        assert _steps(events) == [("model", EventKind.STARTED), ("model", EventKind.ERROR)]
        assert graph.node("model").state is NodeState.FAILED


class TestRealExecutor:
    """真实进程执行器端到端（spawn 子进程）."""

    def test_end_to_end(self) -> None:
        """桁架静力两节点链在子进程执行完成."""
        graph = _truss_graph()
        runner = WorkflowRunner(graph)  # 自建 ProcessExecutor
        done = threading.Event()
        events: list[NodeRunEvent] = []

        def on_event(event: NodeRunEvent) -> None:
            events.append(event)
            if event.kind is EventKind.RESULT and event.node_id == "solve":
                done.set()

        try:
            runner.run_all(on_event)
            assert done.wait(timeout=60)
        finally:
            runner.shutdown()
        assert graph.node("solve").state is NodeState.UP_TO_DATE
