"""GUI Widget 测试（不依赖 PySide 运行时）."""

from __future__ import annotations

__all__ = []


# ------------------------------------------------------------------ Toolbox 纯逻辑


def test_toolbox_group_modules() -> None:
    """_group_modules 按 ModuleCategory 正确分组."""
    from zylab.flowchart.module import ModuleCategory, ModuleSpec
    from zylab.gui.widgets.toolbox import _group_modules

    def _mk(type_id, category):
        return ModuleSpec(type_id=type_id, name=type_id, category=category, target="x")

    specs = [
        _mk("a", ModuleCategory.SOURCE),
        _mk("b", ModuleCategory.ANALYSIS),
        _mk("c", ModuleCategory.SOURCE),
        _mk("d", ModuleCategory.POST),
    ]
    buckets = _group_modules(specs)
    assert list(buckets.keys()) == [ModuleCategory.SOURCE, ModuleCategory.ANALYSIS, ModuleCategory.POST]
    assert [s.type_id for s in buckets[ModuleCategory.SOURCE]] == ["a", "c"]
    assert [s.type_id for s in buckets[ModuleCategory.ANALYSIS]] == ["b"]


def test_toolbox_describe_static() -> None:
    """ModuleToolbox._describe 端口/参数摘要拼接."""
    from zylab.flowchart.module import (
        ModuleCategory,
        ModuleSpec,
        ParamSpec,
        ParamType,
        PortSpec,
        PortType,
    )
    from zylab.gui.widgets.toolbox import ModuleToolbox

    spec = ModuleSpec(
        type_id="t",
        name="T",
        category=ModuleCategory.ANALYSIS,
        target="x",
        inputs=(PortSpec("in1", PortType.MODEL),),
        outputs=(PortSpec("out1", PortType.ANY, required=False),),
        params=(
            ParamSpec("p1", "P1", ParamType.STR, default=""),
            ParamSpec("p2", "P2", ParamType.INT, default=0),
        ),
    )
    desc = ModuleToolbox._describe(spec)
    assert "输入: in1" in desc
    assert "输出: out1" in desc
    assert "参数: p1, p2" in desc

    # 空 spec
    spec0 = ModuleSpec(type_id="x", name="X", category=ModuleCategory.SOURCE, target="x")
    assert ModuleToolbox._describe(spec0) == ""


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
