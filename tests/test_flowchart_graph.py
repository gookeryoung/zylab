"""flowchart.graph 工作流图测试：状态机、级联脏传播、拓扑序、连接编辑."""

from __future__ import annotations

import pytest

from zylab.flowchart import LinkError, NodeState, ParamError, Template, TemplateError, WorkflowGraph

__all__ = []

#: 组合模板：model -> static / modal（Share 连接）
_COMBO = {
    "id": "t.combo",
    "name": "组合",
    "nodes": [
        {"id": "model", "type": "example.cantilever_q4", "params": {"nx": 4, "ny": 2}},
        {"id": "static", "type": "analysis.static", "inputs": {"model": "model.model"}},
        {"id": "modal", "type": "analysis.modal", "inputs": {"model": "model.model"}},
    ],
}


def _graph() -> WorkflowGraph:
    """构造组合模板图."""
    return WorkflowGraph(Template.from_dict(_COMBO))


class TestStructure:
    """图结构与关系查询."""

    def test_initial_states_ready(self) -> None:
        """实例化后全部节点 READY（输入接齐、无结果）."""
        graph = _graph()
        for node in graph.nodes():
            assert node.state is NodeState.READY
            assert node.needs_run

    def test_relations(self) -> None:
        """上下游与祖先后代关系."""
        graph = _graph()
        assert graph.upstream_ids("static") == ("model",)
        assert graph.upstream_ids("model") == ()
        assert sorted(graph.downstream_ids("model")) == ["modal", "static"]
        assert graph.ancestors("modal") == frozenset({"model"})
        assert graph.descendants("model") == frozenset({"static", "modal"})

    def test_execution_order_deterministic(self) -> None:
        """拓扑序：model 在前，同层按定义序."""
        graph = _graph()
        assert graph.execution_order() == ("model", "static", "modal")

    def test_node_missing(self) -> None:
        """查询不存在节点抛 TemplateError."""
        graph = _graph()
        with pytest.raises(TemplateError, match="无节点"):
            graph.node("ghost")

    def test_template_property(self) -> None:
        """template 属性返回来源模板."""
        graph = _graph()
        assert graph.template.id == "t.combo"


class TestStateMachine:
    """节点状态迁移."""

    def test_full_cycle(self) -> None:
        """READY -> RUNNING -> UP_TO_DATE；失败路径 -> FAILED；复位 -> READY."""
        graph = _graph()
        graph.mark_running("static")
        assert graph.node("static").state is NodeState.RUNNING
        graph.mark_result("static", result=object(), elapsed=0.5)
        node = graph.node("static")
        assert node.state is NodeState.UP_TO_DATE
        assert node.elapsed == 0.5
        assert not node.needs_run

        graph.mark_running("modal")
        graph.mark_failed("modal", "求解失败")
        assert graph.node("modal").state is NodeState.FAILED
        assert graph.node("modal").error == "求解失败"
        assert graph.node("modal").needs_run

        graph.mark_running("modal")
        graph.mark_reset("modal")
        assert graph.node("modal").state is NodeState.FAILED  # 错误保留

    def test_node_name_proxy(self) -> None:
        """节点显示名代理自模块规格."""
        graph = _graph()
        assert graph.node("model").name == "悬臂梁（Q4 平面应力）"


class TestDirtyPropagation:
    """参数/连接变更的级联失效."""

    def _all_up_to_date(self, graph: WorkflowGraph) -> None:
        """将全部节点置为 UP_TO_DATE."""
        for node in graph.nodes():
            graph.mark_result(node.id, result=object(), elapsed=0.1)

    def test_param_change_invalidates_self_and_downstream(self) -> None:
        """改模型参数：自身与两路下游全部失效."""
        graph = _graph()
        self._all_up_to_date(graph)
        graph.set_param("model", "nx", 6)
        for node in graph.nodes():
            assert node.state is NodeState.READY
            assert node.result is None

    def test_param_change_invalidates_only_branch(self) -> None:
        """改 modal 参数：仅 modal 失效，static 保持 UP_TO_DATE."""
        graph = _graph()
        self._all_up_to_date(graph)
        graph.set_param("modal", "n_modes", 8)
        assert graph.node("modal").state is NodeState.READY
        assert graph.node("static").state is NodeState.UP_TO_DATE

    def test_param_unchanged_keeps_cache(self) -> None:
        """参数值未变化不失效（缓存保持）."""
        graph = _graph()
        self._all_up_to_date(graph)
        graph.set_param("model", "nx", 4)  # 与原值相同
        assert graph.node("model").state is NodeState.UP_TO_DATE

    def test_set_param_coerces_int(self) -> None:
        """整值浮点收敛为 int."""
        graph = _graph()
        graph.set_param("model", "nx", 6.0)
        assert graph.node("model").params["nx"] == 6

    def test_set_param_unknown_key(self) -> None:
        """未知参数键抛 ParamError."""
        graph = _graph()
        with pytest.raises(ParamError, match="无参数"):
            graph.set_param("model", "ghost", 1.0)

    def test_set_param_unknown_node(self) -> None:
        """未知节点抛 TemplateError."""
        graph = _graph()
        with pytest.raises(TemplateError, match="无节点"):
            graph.set_param("ghost", "nx", 1)

    def test_set_params_atomic(self) -> None:
        """批量设置先整体校验：含非法键时全部不生效."""
        graph = _graph()
        with pytest.raises(ParamError, match="无参数"):
            graph.set_params("model", {"nx": 6, "ghost": 1.0})
        assert graph.node("model").params["nx"] == 4


class TestOptionalPorts:
    """可选输入端口（屈曲 reference / 非线性 initial）."""

    def _linked_graph(self) -> WorkflowGraph:
        """model -> static -> buckling(reference) 三节点链接图."""
        return WorkflowGraph(
            Template.from_dict(
                {
                    "id": "t.linked",
                    "name": "链接",
                    "nodes": [
                        {"id": "model", "type": "example.column_beam2"},
                        {"id": "static", "type": "analysis.static", "inputs": {"model": "model.model"}},
                        {
                            "id": "buckling",
                            "type": "analysis.buckling",
                            "inputs": {"model": "model.model", "reference": "static.solution"},
                        },
                    ],
                }
            )
        )

    def test_optional_port_absent_stays_ready(self) -> None:
        """可选端口不连接仍为 READY（非 UNFULFILLED）."""
        graph = WorkflowGraph(
            Template.from_dict(
                {
                    "id": "t.b",
                    "name": "b",
                    "nodes": [
                        {"id": "model", "type": "example.column_beam2"},
                        {"id": "solve", "type": "analysis.buckling", "inputs": {"model": "model.model"}},
                    ],
                }
            )
        )
        assert graph.node("solve").state is NodeState.READY

    def test_linked_topology(self) -> None:
        """链接图拓扑：buckling 上游含 static，execution_order 含序."""
        graph = self._linked_graph()
        assert graph.upstream_ids("buckling") == ("model", "static")
        assert graph.ancestors("buckling") == frozenset({"model", "static"})
        order = graph.execution_order()
        assert order.index("model") < order.index("static") < order.index("buckling")

    def test_static_dirty_cascades_to_linked_buckling(self) -> None:
        """static 失效级联到 buckling（链接下游）."""
        graph = self._linked_graph()
        for node in graph.nodes():
            graph.mark_result(node.id, result=object(), elapsed=0.1)
        graph.invalidate("static")
        assert graph.node("static").state is NodeState.READY
        assert graph.node("buckling").state is NodeState.READY
        assert graph.node("model").state is NodeState.UP_TO_DATE


class TestLinkEditing:
    """连接编辑（画布交互的内核支撑）."""

    def test_remove_link_unfulfilled(self) -> None:
        """移除连接后节点 UNFULFILLED 且结果失效；兄弟分支不受影响."""
        graph = _graph()
        for node in graph.nodes():
            graph.mark_result(node.id, result=object(), elapsed=0.1)
        graph.remove_link("modal", "model")
        assert graph.node("modal").state is NodeState.UNFULFILLED
        assert graph.node("modal").result is None
        assert graph.node("static").state is NodeState.UP_TO_DATE

    def test_add_link_roundtrip(self) -> None:
        """重连后恢复 READY."""
        graph = _graph()
        graph.remove_link("modal", "model")
        graph.add_link("modal", "model", "model.model")
        assert graph.node("modal").state is NodeState.READY

    def test_add_link_bad_format(self) -> None:
        """引用缺端口名."""
        graph = _graph()
        with pytest.raises(LinkError, match="格式"):
            graph.add_link("modal", "model", "model")

    def test_add_link_unknown_port(self) -> None:
        """本节点端口不存在."""
        graph = _graph()
        with pytest.raises(LinkError, match="无输入端口"):
            graph.add_link("modal", "ghost", "model.model")

    def test_add_link_self_connection(self) -> None:
        """自连接非法."""
        graph = _graph()
        with pytest.raises(LinkError, match="自连接"):
            graph.add_link("modal", "model", "modal.model")

    def test_add_link_unknown_source(self) -> None:
        """上游节点不存在."""
        graph = _graph()
        with pytest.raises(LinkError, match="源节点不存在"):
            graph.add_link("modal", "model", "ghost.model")

    def test_add_link_type_mismatch(self) -> None:
        """端口类型不兼容（解端口接回模型输入）."""
        graph = _graph()
        with pytest.raises(LinkError, match="端口类型不兼容"):
            graph.add_link("modal", "model", "static.solution")


class TestContentHash:
    """NodeInstance.content_hash 与 WorkflowGraph.compute_node_hash."""

    def test_initial_hash_none(self) -> None:
        """实例化后 content_hash 为空."""
        graph = _graph()
        for node in graph.nodes():
            assert node.content_hash is None

    def test_invalidate_clears_hash(self) -> None:
        """级联失效清空全部下游哈希."""
        graph = _graph()
        for node in graph.nodes():
            graph.mark_result(node.id, result=object(), elapsed=0.1, content_hash="fake_hash")
        graph.invalidate("model")
        for node in graph.nodes():
            assert node.content_hash is None

    def test_mark_result_stores_hash(self) -> None:
        """mark_result 写入 content_hash."""
        graph = _graph()
        graph.mark_result("model", result=object(), elapsed=0.1, content_hash="abc123")
        assert graph.node("model").content_hash == "abc123"

    def test_mark_result_none_hash_preserves(self) -> None:
        """mark_result(content_hash=None) 不清空已有哈希."""
        graph = _graph()
        graph.mark_result("model", result=object(), elapsed=0.1, content_hash="abc")
        graph.mark_result("model", result=object(), elapsed=0.2)  # 默认 None
        assert graph.node("model").content_hash == "abc"

    def test_compute_node_hash_source_deterministic(self) -> None:
        """源节点（无输入）指纹仅由 params 决定."""
        graph = _graph()
        h1 = graph.compute_node_hash("model")
        h2 = graph.compute_node_hash("model")
        assert h1 == h2

    def test_compute_node_hash_changes_with_params(self) -> None:
        """参数变更改变指纹."""
        graph = _graph()
        h1 = graph.compute_node_hash("model")
        graph.set_param("model", "nx", 8)
        h2 = graph.compute_node_hash("model")
        assert h1 != h2

    def test_compute_node_hash_propagates_upstream(self) -> None:
        """上游哈希链纳入下游指纹."""
        graph = _graph()
        # 上游无内容哈希时计算下游
        h_no_up = graph.compute_node_hash("static")
        # 给上游填入内容哈希
        graph.node("model").content_hash = "upstream_hash_value"
        h_with_up = graph.compute_node_hash("static")
        assert h_no_up != h_with_up

    def test_compute_node_hash_full_cycle(self) -> None:
        """完整 cycle：执行 -> 哈希记录 -> 重算 -> 哈希相同."""
        graph = _graph()
        # 模拟执行
        for node_id in graph.execution_order():
            h = graph.compute_node_hash(node_id)
            graph.mark_result(node_id, result=object(), elapsed=0.1, content_hash=h)
        # 哈希应全部非空且一致
        for node in graph.nodes():
            assert node.content_hash is not None
            assert node.content_hash == graph.compute_node_hash(node.id)

    def test_compute_node_hash_change_invalidates_downstream(self) -> None:
        """参数变更后下游哈希自动不同（触发重算语义）."""
        graph = _graph()
        for node_id in graph.execution_order():
            h = graph.compute_node_hash(node_id)
            graph.mark_result(node_id, result=object(), elapsed=0.1, content_hash=h)
        # 改上游参数
        graph.set_param("model", "nx", 8)
        # model 和 downstream 的指纹都应变化
        assert graph.node("model").content_hash != graph.compute_node_hash("model")
        assert graph.node("static").content_hash != graph.compute_node_hash("static")
        assert graph.node("modal").content_hash != graph.compute_node_hash("modal")


class TestAddRemoveNode:
    """WorkflowGraph.add_node / remove_node 动态图编辑."""

    def test_add_node_auto_id(self) -> None:
        graph = _graph()
        nid = graph.add_node("example.cantilever_q4")
        assert nid.startswith("cantilever_q4_")
        assert len(graph.nodes()) == 4
        assert graph.node(nid).spec.type_id == "example.cantilever_q4"
        # 源节点（无输入）新增后 READY
        assert graph.node(nid).state is NodeState.READY

    def test_add_node_explicit_id(self) -> None:
        graph = _graph()
        nid = graph.add_node("analysis.harmonic", node_id="extra_harmonic")
        assert nid == "extra_harmonic"
        assert graph.node(nid).params  # harmonic 有 f_max/n_freq/alpha/beta 四参数

    def test_add_node_id_conflict_raises(self) -> None:
        graph = _graph()
        with pytest.raises(TemplateError, match="冲突"):
            graph.add_node("fea.modal", node_id="model")

    def test_add_node_with_position(self) -> None:
        graph = _graph()
        nid = graph.add_node("fea.modal", position=(300.0, 100.0))
        assert graph.node(nid).position == (300.0, 100.0)

    def test_remove_node_cleans_downstream_refs(self) -> None:
        graph = _graph()
        # modal 和 static 都连到 model，删 model 后两者 inputs 应清空
        graph.remove_node("model")
        assert len(graph.nodes()) == 2
        assert graph.node("static").inputs == {}
        assert graph.node("modal").inputs == {}

    def test_remove_nonexistent_raises(self) -> None:
        graph = _graph()
        with pytest.raises(TemplateError, match="无节点"):
            graph.remove_node("ghost")


class TestDescendantsDiamond:
    """descendants 在 diamond 依赖（同一节点多次 push）时应正确跳过."""

    def test_diamond_deps_dedup(self) -> None:
        """手动构造 diamond 拓扑后 descendants 不应重复包含 D."""
        from unittest.mock import MagicMock

        from zylab.flowchart.graph import WorkflowGraph

        g = MagicMock(spec=WorkflowGraph)
        # downstream_ids: A→[B,C], B→[D], C→[D], D→[]
        downstream_map = {
            "a": ["b", "c"],
            "b": ["d"],
            "c": ["d"],
            "d": [],
        }
        g.downstream_ids = lambda nid: frozenset(downstream_map[nid])
        # 直接调原始实现：WorkflowGraph.descendants
        result = WorkflowGraph.descendants(g, "a")
        assert result == frozenset({"b", "c", "d"})  # D 只出现一次


class TestGraphCanConnect:
    """WorkflowGraph.can_connect 委托给 ports.can_connect."""

    def test_can_connect_delegates(self) -> None:
        """graph.can_connect 应返回 (True, '') 对合法 MODEL→MODEL."""
        g = _graph()
        ok, reason = g.can_connect("model", "model", "static", "model")
        assert ok
        assert reason == ""
