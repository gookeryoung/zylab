"""批处理执行：模板进程内拓扑序求解 + 参数覆盖/扫描 + 结果摘要.

与 :class:`~zylab.studio.runner.WorkflowRunner`（GUI 进程隔离编排）互补：
批处理场景（CLI、参数扫描）无交互，进程内直调节点函数省去子进程 pickle
往返，扫描多组参数时收益显著。失败策略 = 首个失败节点中止（下游依赖其
输出，不可继续），后续节点保持未执行。
"""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from zylab.fea import (
    BucklingSolution,
    ElectroThermalSolution,
    HarmonicResponse,
    ModalSolution,
    NonlinearSolution,
    StaticSolution,
    TransientSolution,
)

from .bundle import ConductionBundle, ModelBundle
from .errors import StudioError
from .graph import WorkflowGraph
from .template import Template

__all__ = ["NodeOutcome", "ReportFn", "RunOutcome", "resolve_target", "run_scan", "run_workflow", "summarize"]

logger = logging.getLogger(__name__)

#: 进度回调签名（与节点协议 / ProcessExecutor 注入约定一致）
ReportFn = Callable[[float, str], None]


def resolve_target(target: str) -> Callable[..., Any]:
    """解析 ``"模块:函数"`` 目标字符串为可调用对象."""
    module_name, _, attr = target.partition(":")
    return getattr(importlib.import_module(module_name), attr)


@dataclass(frozen=True)
class NodeOutcome:
    """单节点执行结果.

    :param node_id: 节点 id。
    :param name: 模块显示名。
    :param result: 节点输出对象（失败/未执行为 None）。
    :param error: 错误消息（空串表示成功）。
    :param elapsed: 执行耗时（秒）。
    """

    node_id: str
    name: str
    result: Any = None
    error: str = ""
    elapsed: float = 0.0

    @property
    def ok(self) -> bool:
        """是否执行成功."""
        return not self.error


@dataclass(frozen=True)
class RunOutcome:
    """单次工作流运行结果（拓扑序全部节点）.

    :param outcomes: 节点结果表（模板定义序）。
    """

    outcomes: tuple[NodeOutcome, ...]

    @property
    def succeeded(self) -> bool:
        """是否全部节点成功执行（失败中止后的未执行节点不计入）."""
        return all(o.ok and o.result is not None for o in self.outcomes)

    def outcome(self, node_id: str) -> NodeOutcome:
        """按节点 id 取执行结果；不存在抛 :class:`KeyError`."""
        for o in self.outcomes:
            if o.node_id == node_id:
                return o
        raise KeyError(f"运行结果中无节点 {node_id!r}")

    def first_error(self) -> str:
        """首个失败节点的错误消息（全部成功返回空串）."""
        for o in self.outcomes:
            if not o.ok:
                return f"{o.node_id} ({o.name}): {o.error}"
        return ""


def run_workflow(
    template: Template,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    report: ReportFn | None = None,
) -> RunOutcome:
    """进程内按拓扑序执行模板全部节点（失败即中止）.

    :param template: 分析模板。
    :param overrides: 节点参数覆盖表（节点 id -> 参数表，整体替换该节点 params）。
    :param report: 进度回调（透传给节点函数，``(progress, message)``）。
    """
    merged = template.with_params(dict(overrides)) if overrides else template
    graph = WorkflowGraph(merged)
    results: dict[str, Any] = {}
    outcomes: list[NodeOutcome] = []

    failed = False
    for node in graph.nodes():
        if failed:
            outcomes.append(NodeOutcome(node_id=node.id, name=node.name))
            continue
        inputs = {port: results.get(ref.partition(".")[0]) for port, ref in node.inputs.items()}
        fn = resolve_target(node.spec.target)
        started = time.perf_counter()
        try:
            results[node.id] = fn(inputs, dict(node.params), report)
            elapsed = time.perf_counter() - started
            outcomes.append(NodeOutcome(node_id=node.id, name=node.name, result=results[node.id], elapsed=elapsed))
            logger.debug("批处理节点完成: %s (%.3fs)", node.id, elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            message = f"{type(exc).__name__}: {exc}"
            outcomes.append(NodeOutcome(node_id=node.id, name=node.name, error=message, elapsed=elapsed))
            logger.warning("批处理节点失败: %s: %s", node.id, message)
            failed = True
    return RunOutcome(tuple(outcomes))


def run_scan(
    template: Template,
    param_ref: str,
    values: tuple[float, ...],
    report: ReportFn | None = None,
) -> tuple[RunOutcome, ...]:
    """参数化扫描：对 ``"node.param"`` 参数逐值运行整个工作流.

    :param param_ref: 参数引用（``"节点id.参数键"``）。
    :param values: 扫描取值序列（原样传入，不做插值）。
    :param report: 进度回调（透传给每次运行）。
    """
    node_id, _, key = param_ref.partition(".")
    if not node_id or not key:
        raise ValueError(f"参数引用 {param_ref!r} 应为 '节点id.参数键' 格式")
    runs = []
    for value in values:
        overrides = {node_id: {**_node_params(template, node_id), key: value}}
        runs.append(run_workflow(template, overrides, report))
    return tuple(runs)


def _node_params(template: Template, node_id: str) -> dict[str, Any]:
    """取模板节点原始参数表（不存在抛 :class:`ValueError`）."""
    try:
        return dict(template.node(node_id).params)
    except StudioError as exc:
        raise ValueError(f"模板 {template.id!r} 无节点 {node_id!r}: {exc}") from exc


def summarize(outcome: RunOutcome) -> str:
    """生成运行结果摘要（每结果节点一行关键指标，中文）."""
    lines: list[str] = []
    for o in outcome.outcomes:
        if not o.ok:
            lines.append(f"[{o.node_id}] {o.name} 失败: {o.error}")
        elif o.result is None:
            lines.append(f"[{o.node_id}] {o.name} 未执行（上游失败）")
        else:
            lines.append(f"[{o.node_id}] {o.name} {_describe(o.result)}（{o.elapsed:.3f}s）")
    return "\n".join(lines)


def _describe(result: Any) -> str:  # noqa: PLR0911  各类解各一行指标，分支语义不可合并
    """按解类型生成单行指标描述."""
    if isinstance(result, (ModelBundle, ConductionBundle)):
        mesh = result.mesh
        return f"模型: {mesh.n_nodes} 节点 / {mesh.n_elements} 单元"
    if isinstance(result, StaticSolution):
        u = result.displacements
        tip = float(np.linalg.norm(u, axis=1).max()) if u.size else 0.0
        return f"静力: 最大位移模长 {tip:.6g}，应变能 {result.strain_energy:.6g}"
    if isinstance(result, ModalSolution):
        hz = result.frequencies_hz[:3]
        shown = " / ".join(f"{f:.4g}" for f in hz)
        return f"模态: 前 {hz.size} 阶频率 {shown} Hz（共 {result.n_modes} 阶）"
    if isinstance(result, HarmonicResponse):
        amp = np.abs(result.displacements).max() if result.displacements.size else 0.0
        return f"谐响应: 峰值位移幅值 {float(amp):.6g}（{result.n_frequencies} 频率点）"
    if isinstance(result, BucklingSolution):
        factor = float(result.load_factors.min()) if result.load_factors.size else float("nan")
        return f"屈曲: 最小临界载荷因子 {factor:.6g}（共 {result.n_modes} 阶）"
    if isinstance(result, NonlinearSolution):
        u = result.displacements
        tip = float(np.linalg.norm(u, axis=1).max()) if u.size else 0.0
        state = "收敛" if result.converged else "未收敛"
        return f"非线性: {state}，最大位移模长 {tip:.6g}，Newton 迭代 {result.total_iterations} 次"
    if isinstance(result, TransientSolution):
        u = result.displacements
        peak = float(np.abs(u).max()) if u.size else 0.0
        return f"瞬态: 峰值位移 {peak:.6g}（{result.n_steps} 步，dt={result.dt:.4g}s）"
    if isinstance(result, ElectroThermalSolution):
        return f"电热: 峰值温度 {result.t_max:.6g}，总电功率 {result.total_power:.6g} W"
    return f"完成: {type(result).__name__}"
