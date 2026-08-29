"""studio.graph 工作流图测试：状态机、级联脏传播、拓扑序、连接编辑."""

from __future__ import annotations

import pytest

from zylab.studio import LinkError, NodeState, ParamError, Template, TemplateError, WorkflowGraph

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
        with pytest.raises(LinkError, match="无效"):
            graph.add_link("modal", "model", "ghost.model")

    def test_add_link_type_mismatch(self) -> None:
        """端口类型不匹配（解端口接回模型输入）."""
        graph = _graph()
        with pytest.raises(LinkError, match="端口类型不匹配"):
            graph.add_link("modal", "model", "static.solution")
