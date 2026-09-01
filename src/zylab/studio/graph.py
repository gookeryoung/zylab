"""工作流图：模板实例化 + 节点状态机 + 级联脏传播 + 拓扑执行计划.

状态机（ANSYS 单元格状态简化版）::

    UNFULFILLED（输入未接齐）--接齐--> READY --运行--> RUNNING --成功--> UP_TO_DATE
                                        ^                            失败--> FAILED
    参数/连接变更：本节点与全部下游级联失效（UP_TO_DATE/FAILED -> READY/UNFULFILLED）

无环不变量：端口类型分层（SOURCE 产出 MODEL，ANALYSIS 消费 MODEL 产出解类型），
连接类型校验结构性阻断回边，拓扑排序必然完整。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Mapping

from .errors import LinkError, StudioError, TemplateError
from .module import ModuleSpec, module_spec
from .template import Template

__all__ = ["NodeInstance", "NodeState", "WorkflowGraph"]


@unique
class NodeState(Enum):
    """节点运行状态."""

    UNFULFILLED = "unfulfilled"  # 输入未接齐
    READY = "ready"  # 可运行（无有效结果）
    RUNNING = "running"  # 执行中
    UP_TO_DATE = "up_to_date"  # 结果有效（缓存命中，重跑自动跳过）
    FAILED = "failed"  # 上次执行失败


@dataclass
class NodeInstance:
    """运行期节点实例（状态由 :class:`WorkflowGraph` 方法迁移，外部只读）.

    :param id: 节点 id。
    :param spec: 模块类型规格。
    :param params: 已校验收敛的参数表。
    :param inputs: 入端口名 -> 上游引用（``"node_id.port"``）。
    :param result: 缓存的输出对象（失效时清空）。
    :param error: 上次执行错误消息（空串表示无错误）。
    :param running: 执行中标志（runner 管理）。
    :param elapsed: 上次执行耗时（秒）。
    """

    id: str
    spec: ModuleSpec
    params: dict[str, Any]
    inputs: dict[str, str]
    result: Any = None
    error: str = ""
    running: bool = False
    elapsed: float = 0.0

    @property
    def state(self) -> NodeState:
        """当前状态（由运行标志/输入完整性/错误/结果推导）."""
        if self.running:
            return NodeState.RUNNING
        if any(port.name not in self.inputs for port in self.spec.inputs if port.required):
            return NodeState.UNFULFILLED
        if self.error:
            return NodeState.FAILED
        if self.result is not None:
            return NodeState.UP_TO_DATE
        return NodeState.READY

    @property
    def needs_run(self) -> bool:
        """是否需要执行（READY/FAILED；UP_TO_DATE 命中缓存跳过，UNFULFILLED 不可运行）."""
        return self.state in (NodeState.READY, NodeState.FAILED)

    @property
    def name(self) -> str:
        """模块显示名."""
        return self.spec.name


class WorkflowGraph:
    """工作流图：节点状态机 + 参数/连接变更的级联脏传播 + 拓扑执行计划."""

    def __init__(self, template: Template) -> None:
        """由模板实例化图（参数按模块 schema 校验收敛）."""
        self._template = template
        self._nodes: dict[str, NodeInstance] = {}
        for tn in template.nodes:
            spec = module_spec(tn.type_id)
            self._nodes[tn.id] = NodeInstance(
                id=tn.id,
                spec=spec,
                params=spec.coerce_params(tn.params),
                inputs=dict(tn.inputs),
            )

    @property
    def template(self) -> Template:
        """来源模板."""
        return self._template

    # ------------------------------------------------------------------ 查询

    def nodes(self) -> tuple[NodeInstance, ...]:
        """全部节点（模板定义序）."""
        return tuple(self._nodes.values())

    def node(self, node_id: str) -> NodeInstance:
        """按 id 取节点；不存在抛 :class:`TemplateError`."""
        try:
            return self._nodes[node_id]
        except KeyError:
            raise TemplateError(f"图中无节点 {node_id!r}") from None

    def upstream_ids(self, node_id: str) -> tuple[str, ...]:
        """直接上游节点 id 表."""
        return tuple(ref.partition(".")[0] for ref in self.node(node_id).inputs.values())

    def downstream_ids(self, node_id: str) -> tuple[str, ...]:
        """直接下游节点 id 表."""
        self.node(node_id)
        return tuple(n.id for n in self._nodes.values() if node_id in self.upstream_ids(n.id))

    def ancestors(self, node_id: str) -> frozenset[str]:
        """全部递归上游（不含自身）."""
        result: set[str] = set()
        stack = list(self.upstream_ids(node_id))
        while stack:
            current = stack.pop()
            if current in result:
                continue
            result.add(current)
            stack.extend(self.upstream_ids(current))
        return frozenset(result)

    def descendants(self, node_id: str) -> frozenset[str]:
        """全部递归下游（不含自身）."""
        result: set[str] = set()
        stack = list(self.downstream_ids(node_id))
        while stack:
            current = stack.pop()
            if current in result:
                continue
            result.add(current)
            stack.extend(self.downstream_ids(current))
        return frozenset(result)

    def execution_order(self) -> tuple[str, ...]:
        """Kahn 拓扑序（同层按模板定义序，保证执行确定性）."""
        ids = [n.id for n in self._nodes.values()]
        indegree = {nid: len(self.upstream_ids(nid)) for nid in ids}
        ready = [nid for nid in ids if indegree[nid] == 0]
        order: list[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for nid in ids:
                if current in self.upstream_ids(nid):
                    indegree[nid] -= 1
                    if indegree[nid] == 0:
                        ready.append(nid)
        if len(order) != len(ids):  # pragma: no cover（端口类型分层结构性保证无环，防御未来扩展）
            raise LinkError("工作流图存在环")
        return tuple(order)

    # ------------------------------------------------------------------ 变更

    def set_param(self, node_id: str, key: str, value: object) -> None:
        """设置单个参数（按模块 schema 校验）；值实际变化时本节点与全部下游级联失效."""
        self.set_params(node_id, {key: value})

    def set_params(self, node_id: str, values: Mapping[str, Any]) -> None:
        """批量设置参数（先整体校验后应用，避免部分生效）；有变化时级联失效."""
        node = self.node(node_id)
        coerced = {key: node.spec.param(key).coerce(value) for key, value in values.items()}
        changed = {key: value for key, value in coerced.items() if node.params.get(key) != value}
        if not changed:
            return
        node.params.update(changed)
        self._invalidate(node_id)

    def add_link(self, dst_id: str, dst_port: str, src_ref: str) -> None:
        """连接上游输出到目标节点输入端口；本节点与下游级联失效.

        :param src_ref: 上游引用（``"node_id.port"`` 格式）。
        :raises LinkError: 端口不存在/引用格式错误/自连接/端口类型不匹配。
        """
        node = self.node(dst_id)
        try:
            in_port = node.spec.input_port(dst_port)
        except StudioError as exc:
            raise LinkError(str(exc)) from exc
        src_id, sep, src_port = src_ref.partition(".")
        if not sep or not src_id or not src_port:
            raise LinkError(f"连接 {src_ref!r} 应为 '节点id.端口名' 格式")
        if src_id == dst_id:
            raise LinkError(f"节点 {dst_id!r} 不允许自连接")
        try:
            out_port = self.node(src_id).spec.output_port(src_port)
        except StudioError as exc:
            raise LinkError(f"连接 {src_ref!r} 无效: {exc}") from exc
        if out_port.port_type is not in_port.port_type:
            raise LinkError(f"端口类型不匹配: {out_port.port_type.value} != {in_port.port_type.value}")
        node.inputs[dst_port] = src_ref
        self._invalidate(dst_id)

    def remove_link(self, dst_id: str, dst_port: str) -> None:
        """移除输入连接；本节点与下游级联失效."""
        node = self.node(dst_id)
        if dst_port in node.inputs:
            del node.inputs[dst_port]
            self._invalidate(dst_id)

    def invalidate(self, node_id: str) -> None:
        """手动失效本节点与全部下游（强制重跑用）；清空结果/错误/耗时."""
        self._invalidate(node_id)

    def _invalidate(self, node_id: str) -> None:
        """本节点与全部下游：清空结果/错误/耗时（回到 READY 或 UNFULFILLED）."""
        for nid in (node_id, *self.descendants(node_id)):
            node = self._nodes[nid]
            node.result = None
            node.error = ""
            node.elapsed = 0.0

    # ------------------------------------------------------------- 状态迁移（runner 调用）

    def mark_running(self, node_id: str) -> None:
        """标记执行中."""
        self.node(node_id).running = True

    def mark_result(self, node_id: str, result: Any, elapsed: float) -> None:
        """登记执行结果（-> UP_TO_DATE）并记录耗时."""
        node = self.node(node_id)
        node.result = result
        node.elapsed = elapsed
        node.running = False

    def mark_failed(self, node_id: str, error: str) -> None:
        """登记执行失败（-> FAILED）."""
        node = self.node(node_id)
        node.error = error
        node.running = False

    def mark_reset(self, node_id: str) -> None:
        """取消后清除运行标志（FAILED 重跑中被取消则回到 FAILED，否则 READY）."""
        self.node(node_id).running = False
