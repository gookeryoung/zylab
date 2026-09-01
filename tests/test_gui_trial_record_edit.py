"""gui.widgets.trial_record_edit 试验记录输入控件测试：升降规则、编辑修正、表单集成."""

from __future__ import annotations

import pytest

from zylab.gui.widgets.dsl_param_form import DslParamForm
from zylab.gui.widgets.trial_record_edit import TrialRecordEdit
from zylab.studio import BUILTIN_TEMPLATES


@pytest.fixture
def edit(qtbot) -> TrialRecordEdit:
    """空记录控件（初始刺激量 3.2、步长 0.05）."""
    widget = TrialRecordEdit()
    qtbot.addWidget(widget)
    return widget


@pytest.mark.gui
def test_append_follows_updown_rule(qtbot, edit: TrialRecordEdit) -> None:
    """追加按升降规则推水平：响应降一步、不响应升一步."""
    edit._add_hit_btn.click()
    assert edit.text() == "3.2 O"
    edit._add_miss_btn.click()
    assert edit.text() == "3.2 O, 3.15 X"  # O → 降 0.05
    edit._add_miss_btn.click()
    assert edit.text() == "3.2 O, 3.15 X, 3.2 X"  # X → 升 0.05
    assert edit._table.rowCount() == 3


@pytest.mark.gui
def test_text_changed_signal_emitted(qtbot, edit: TrialRecordEdit) -> None:
    """记录变化发出 textChanged（QLineEdit 兼容信号）."""
    with qtbot.waitSignal(edit.textChanged, timeout=1000) as blocker:
        edit._add_hit_btn.click()
    assert blocker.args == ["3.2 O"]


@pytest.mark.gui
def test_set_text_roundtrip_and_invalid(qtbot, edit: TrialRecordEdit) -> None:
    """setText 解析记录重建表格；格式非法时清空."""
    edit.setText("3.20 O, 3.15 X, 3.20 X")
    assert edit._table.rowCount() == 3
    assert edit.text() == "3.2 O, 3.15 X, 3.2 X"
    edit.setText("not a record")
    assert edit._table.rowCount() == 0
    assert edit.text() == ""


@pytest.mark.gui
def test_cell_edit_updates_and_reverts(qtbot, edit: TrialRecordEdit) -> None:
    """表格修正：改刺激量生效；非法输入回退原值."""
    edit.setText("3.20 O, 3.15 X")
    edit._table.item(0, 1).setText("3.25")  # 修正第一发刺激量
    assert edit.text().startswith("3.25 O")
    edit._table.item(1, 2).setText("O")  # 第二发响应改为响应
    assert "3.15 O" in edit.text()
    edit._table.item(0, 1).setText("abc")  # 非法刺激量回退
    assert edit.text().startswith("3.25 O")


@pytest.mark.gui
def test_undo_and_clear(qtbot, edit: TrialRecordEdit) -> None:
    """撤销删除末发、清空删除全部（空表操作无副作用）."""
    edit._add_hit_btn.click()
    edit._add_miss_btn.click()
    edit._undo_btn.click()
    assert edit.text() == "3.2 O"
    edit._clear_btn.click()
    assert edit.text() == ""
    edit._undo_btn.click()  # 空表撤销不报错
    assert edit.text() == ""


@pytest.mark.gui
def test_dsl_param_form_dispatch_and_step_sync(qtbot) -> None:
    """DslParamForm 按声明分发定制控件并联动 step 参数建议步长."""
    template = next(t for t in BUILTIN_TEMPLATES if t.id == "dsl.sensitivity_updown_records")
    form = DslParamForm()
    qtbot.addWidget(form)
    form.set_template(template)
    record_edit = form._record_edits["records"]
    assert isinstance(record_edit, TrialRecordEdit)
    # 模板默认值（Excel 24 发记录）载入并可收集
    assert record_edit._table.rowCount() == 24
    assert form.values()["records"].startswith("3.2 O")
    assert form.values()["step"] == 0.05
    # step 参数变化联动建议步长
    form._spins["step"].setValue(0.1)
    assert record_edit._step_spin.value() == 0.1
    # 收集值直接可被后端分析（还原分析步长 0.05）
    from zylab.reliability import analyze_updown_records

    form._spins["step"].setValue(0.05)
    result = analyze_updown_records(form.values()["records"], form.values()["step"])
    assert result.mu_hat == pytest.approx(3.225, abs=1.0e-6)
