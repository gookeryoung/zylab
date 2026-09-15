"""最终冲刺：补 _table_utils + dsl_param_form + param_form + command_palette 等小遗漏."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# --- _table_utils.py: format_cell_with_format 异常分支 + col_alignment 分支 ---


def test_format_cell_with_format_bad_fmt() -> None:
    """format_cell_with_format 中 format(value, fmt) 抛 ValueError 分支."""
    from zylab.gui.widgets._table_utils import format_cell_with_format

    result = format_cell_with_format(1.5, "bad_fmt_xyz")
    assert isinstance(result, str)


def test_col_alignment_variants() -> None:
    """col_alignment 各分支（right/left/center/default）."""
    from zylab.gui.widgets._table_utils import col_alignment

    Qt = pytest.importorskip("PySide2.QtCore").Qt
    right = col_alignment(SimpleNamespace(align="right"))
    left = col_alignment(SimpleNamespace(align="left"))
    center = col_alignment(SimpleNamespace(align="center"))
    default = col_alignment(SimpleNamespace(align=""))

    assert right == (Qt.AlignRight | Qt.AlignVCenter)
    assert left == (Qt.AlignLeft | Qt.AlignVCenter)
    assert center == Qt.AlignCenter
    assert default == (Qt.AlignRight | Qt.AlignVCenter)


# --- dsl_param_form.py: 空模板 ---


def test_dsl_param_form_empty_template(qapp) -> None:
    """DslParamForm 设置空模板后 values 返回空字典."""
    from zylab.flowchart.dsl import DslTemplate
    from zylab.gui.widgets.dsl_param_form import DslParamForm

    form = DslParamForm()
    form.set_template(DslTemplate(id="t", name="t", nodes=(), dsl_params=()))
    assert form.values() == {}
    form.set_fields_enabled(False)
    form.set_fields_enabled(True)


# --- param_form.py line 72: 非 numeric param_type 跳过 ---


def test_param_form_non_numeric_skip(qapp) -> None:
    """ParamForm.set_graph 遇到 STR/MAP 等非 numeric 参数类型时跳过."""
    from types import SimpleNamespace

    from zylab.gui.widgets.param_form import ParamForm

    param_spec = SimpleNamespace(param_type="str")
    node_spec = SimpleNamespace(params=[param_spec], param=lambda k: param_spec)
    node = SimpleNamespace(spec=node_spec, id="n1", name="Node1", params={})
    graph = SimpleNamespace(node=lambda nid: node, nodes=lambda: [node])
    groups = (SimpleNamespace(title="Test", params=("n1.myparam",)),)

    form = ParamForm()
    form.set_graph(graph, groups)
    # STR 被跳过 → _fields 应空
    assert form._fields == {}


# --- simple_heatmap.py: 140 + 157 ---


def test_simple_heatmap_no_data(qapp) -> None:
    """simple_heatmap 无数据分支."""
    from zylab.gui.widgets.simple_heatmap import SimpleHeatmap

    w = SimpleHeatmap()
    w.set_data(None)
    w.resize(200, 200)
    w.show()
    w.repaint()


# --- trial_record_edit.py: 空记录编辑 ---


def test_trial_record_edit_empty(qapp) -> None:
    """TrialRecordEdit 空记录 → setText + clear → 保持空."""
    from zylab.gui.widgets.trial_record_edit import TrialRecordEdit

    w = TrialRecordEdit()
    w.setText("")
    assert w.text() == ""
    w._clear()  # 已空 → 不应 crash
    assert w.text() == ""
    w.set_suggest_step(0.1)
    w._append(1)
    w._append(0)
    w._remove_last()
    w._clear()
    assert w.text() == ""
