"""gui.widgets.param_form 参数表单测试：schema 驱动生成、编辑信号、值同步."""

from __future__ import annotations

import pytest

from zylab.gui.widgets.param_form import ParamForm
from zylab.studio import Template, WorkflowGraph

_MODAL_TEMPLATE = {
    "id": "t.modal",
    "name": "模态",
    "nodes": [
        {"id": "model", "type": "example.cantilever_q4", "params": {"nx": 4, "ny": 2}},
        {"id": "solve", "type": "analysis.modal", "inputs": {"model": "model.model"}},
    ],
    "ui": {
        "param_groups": [
            {"title": "几何与网格", "params": ["model.length", "model.height", "model.nx", "model.ny"]},
            {"title": "分析", "params": ["solve.n_modes"]},
        ],
        "results": ["solve"],
    },
}


def _form_with_template(qtbot) -> tuple[ParamForm, WorkflowGraph]:
    """构建已装配模板的表单."""
    graph = WorkflowGraph(Template.from_dict(_MODAL_TEMPLATE))
    form = ParamForm()
    qtbot.addWidget(form)
    form.set_graph(graph, graph.template.param_groups)
    return form, graph


@pytest.mark.gui
def test_form_builds_grouped_fields(qtbot) -> None:
    """按分组生成 QGroupBox 与输入框（标签含单位）."""
    form, _graph = _form_with_template(qtbot)
    assert ("model", "nx") in form._fields
    assert ("solve", "n_modes") in form._fields
    spin = form._fields[("model", "nx")]
    assert spin.value() == 4.0
    assert spin.decimals() == 0  # INT 参数整数显示
    assert "范围" in spin.toolTip()
    # 未暴露的参数不生成
    assert ("model", "e_modulus") not in form._fields


@pytest.mark.gui
def test_edit_emits_param_edited(qtbot) -> None:
    """编辑输入框发出 (node_id, key, value) 信号."""
    form, _graph = _form_with_template(qtbot)
    received: list[tuple[str, str, object]] = []
    form.param_edited.connect(lambda nid, key, value: received.append((nid, key, value)))
    form._fields[("solve", "n_modes")].setValue(10.0)
    assert received == [("solve", "n_modes", 10.0)]


@pytest.mark.gui
def test_refresh_values_syncs_without_signal(qtbot) -> None:
    """外部图变更后同步控件值且不触发信号（防回环）."""
    form, graph = _form_with_template(qtbot)
    received: list[tuple[str, str, object]] = []
    form.param_edited.connect(lambda nid, key, value: received.append((nid, key, value)))
    graph.set_param("model", "nx", 12)
    form.refresh_values()
    assert form._fields[("model", "nx")].value() == 12.0
    assert received == []


@pytest.mark.gui
def test_set_fields_enabled(qtbot) -> None:
    """运行中禁用全部输入."""
    form, _graph = _form_with_template(qtbot)
    form.set_fields_enabled(False)
    assert all(not spin.isEnabled() for spin in form._fields.values())
    form.set_fields_enabled(True)
    assert all(spin.isEnabled() for spin in form._fields.values())


@pytest.mark.gui
def test_flat_form_when_no_groups(qtbot) -> None:
    """模板未声明分组时按节点平铺全部参数."""
    template = Template.from_dict(
        {
            "id": "t.flat",
            "name": "平铺",
            "nodes": [
                {"id": "model", "type": "example.truss2_two_bar"},
                {"id": "solve", "type": "analysis.static", "inputs": {"model": "model.model"}},
            ],
        }
    )
    graph = WorkflowGraph(template)
    form = ParamForm()
    qtbot.addWidget(form)
    form.set_graph(graph, template.param_groups)  # 空分组 -> 平铺
    # 桁架源节点全部参数 + 静力无参数
    assert ("model", "half_span") in form._fields
    assert ("model", "area") in form._fields
    assert not any(nid == "solve" for nid, _key in form._fields)


@pytest.mark.gui
def test_refresh_without_graph_is_noop(qtbot) -> None:
    """未装配图时刷新为空操作."""
    form = ParamForm()
    qtbot.addWidget(form)
    form.refresh_values()  # 不抛异常


@pytest.mark.gui
def test_set_graph_rebuilds_fields(qtbot) -> None:
    """重复装配：旧控件清除，新分组生效."""
    form, _graph = _form_with_template(qtbot)
    first_spin = form._fields[("model", "nx")]
    form.set_graph(_graph, _graph.template.param_groups)
    assert form._fields[("model", "nx")] is not first_spin  # 重建为新控件
