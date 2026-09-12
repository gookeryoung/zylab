"""GUI Widget 测试（不依赖 PySide 运行时）."""

from __future__ import annotations

__all__ = []


# ------------------------------------------------------------------ Graph Position 传递


def test_position_from_template_to_node_instance() -> None:
    """TemplateNode.position -> NodeInstance.position 链路."""
    from zylab.flowchart import Template, WorkflowGraph

    combo = {
        "id": "t.combo",
        "name": "组合",
        "nodes": [
            {"id": "model", "type": "example.cantilever_q4", "params": {"nx": 4, "ny": 2}, "position": [120.0, 240.0]},
            {"id": "static", "type": "analysis.static", "inputs": {"model": "model.model"}, "position": None},
        ],
    }
    t = Template.from_dict(combo)
    g = WorkflowGraph(t)
    assert g.node("model").position == (120.0, 240.0)
    assert g.node("static").position is None


def test_template_position_serialization() -> None:
    """Template.to_dict 正确序列化/反序列化 position 字段."""
    from zylab.flowchart import Template

    data = {
        "id": "T001",
        "name": "test_pos",
        "nodes": [
            {"id": "model", "type": "example.cantilever_q4", "position": [100.0, 200.0]},
            {
                "id": "static",
                "type": "analysis.static",
                "inputs": {"model": "model.model"},
            },  # 无 position
        ],
    }
    t = Template.from_dict(data)
    d = t.to_dict()

    # 有 position 的节点保留
    m_node = next(n for n in d["nodes"] if n["id"] == "model")
    assert m_node["position"] == [100.0, 200.0]

    # 无 position 的节点字段不存在
    s_node = next(n for n in d["nodes"] if n["id"] == "static")
    assert "position" not in s_node

    # 反序列化回来
    t2 = Template.from_dict(d)
    nodes = list(t2.nodes)
    assert nodes[0].position == (100.0, 200.0)
    assert nodes[1].position is None
