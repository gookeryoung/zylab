"""gui.pages.template_page 模板应用页测试：加载/参数表单/运行/结果渲染/报告导出."""

from __future__ import annotations

from pathlib import Path

import pytest

from zylab.gui.pages.template_page import TemplatePage
from zylab.gui.widgets.dsl_param_form import DslParamForm
from zylab.gui.widgets.dsl_result_view import DslResultView
from zylab.studio.dsl import dsl_from_yaml

_YAML = """
meta: {id: t.page, name: 页面模板}
theme: ""
params:
  几何:
    items:
      lo: {value: 0.0, unit: m, min: -1.0, max: 10.0}
      hi: {value: 2.0, unit: m, min: 0.0, max: 10.0}
      span: {expr: "hi - lo", unit: m}
pipeline:
  - id: sweep
    type: compute.sweep
    params:
      var: x
      from: "$lo"
      to: "$hi"
      count: 3
      body:
        nodes:
          - id: y
            type: compute.expr
            params: {expr: "x ** 2", vars: {x: "$x"}}
        collect: ["y"]
results:
  - id: curve_y
    kind: curve
    title: 平方曲线
    x: sweep.values
    y: sweep.series.y
  - id: summary
    kind: text
    title: 摘要
    text: "最大值 {max}"
    values: {max: "sweep.series.y.2"}
report:
  sections:
    - title: 曲线
      figure: curve_y
docs:
  intro: {text: 输入扫描范围后运行}
"""


@pytest.fixture
def template():
    """扫参 DSL 模板."""
    return dsl_from_yaml(_YAML)


@pytest.mark.gui
def test_param_form_numeric_and_derived(qtbot, template) -> None:
    """数值参数生成输入框；派生参数只读行随输入实时重算."""
    form = DslParamForm()
    qtbot.addWidget(form)
    form.set_template(template)
    assert form._spins["lo"].value() == 0.0
    assert form._spins["lo"].suffix() == " m"
    assert form._spins["lo"].minimum() == -1.0
    assert form.values() == {"lo": 0.0, "hi": 2.0}
    assert form._derived_labels["span"].text() == "2"
    form._spins["lo"].setValue(1.5)
    assert form.values()["lo"] == 1.5
    assert form._derived_labels["span"].text() == "0.5"


@pytest.mark.gui
def test_page_load_and_run(qtbot, template) -> None:
    """加载模板建参数表单与结果页签；后台运行后渲染曲线与文本页."""
    page = TemplatePage()
    qtbot.addWidget(page)
    page.load_template(template)
    assert page._run_btn.isEnabled()
    assert page._tabs.count() == 2  # curve + text 两结果页
    assert page._tabs.tabText(0) == "平方曲线"
    assert page._docs_label.isVisibleTo(page) and "扫描范围" in page._docs_label.text()

    with qtbot.waitSignal(page.run_finished, timeout=10000) as blocker:
        page.run()
    outputs, error = blocker.args
    assert error == ""
    assert outputs["sweep"]["series"]["y"] == [0.0, 1.0, 4.0]
    # 曲线页 + 文本页渲染完成
    assert page._tabs.count() == 2
    curve_page = page._tabs.widget(0)
    assert isinstance(curve_page, DslResultView)
    assert curve_page._title.text() == "平方曲线"
    assert "尚未运行" not in curve_page._body.text() if hasattr(curve_page._body, "text") else True
    assert page._status_label.text() == "运行完成"
    assert page._export_btn.isEnabled()


@pytest.mark.gui
def test_page_run_param_error_kept_idle(qtbot) -> None:
    """运行期派生表达式除零：运行前置校验直接提示，不进入运行态."""
    yaml_text = _YAML.replace('span: {expr: "hi - lo", unit: m}', 'span: {expr: "hi / lo", unit: m}')
    yaml_text = yaml_text.replace("lo: {value: 0.0", "lo: {value: 1.0")
    page = TemplatePage()
    qtbot.addWidget(page)
    page.load_template(dsl_from_yaml(yaml_text))
    messages = []
    page.status_message.connect(messages.append)
    page._param_form._spins["lo"].setValue(0.0)  # 运行期才非法（构造期默认 1.0 合法）
    page.run()
    assert any("参数错误" in m for m in messages)
    assert not page._running
    assert page._run_btn.isEnabled()


@pytest.mark.gui
def test_page_run_failure_shows_error(qtbot) -> None:
    """节点运行失败：运行失败态 + 状态栏首个错误."""
    yaml_text = _YAML.replace('expr: "x ** 2"', 'expr: "x ** 2 +"')
    page = TemplatePage()
    qtbot.addWidget(page)
    page.load_template(dsl_from_yaml(yaml_text))
    messages = []
    page.status_message.connect(messages.append)
    with qtbot.waitSignal(page.run_finished, timeout=10000):
        page.run()
    assert page._status_label.text() == "运行失败"
    assert any("运行失败" in m for m in messages)
    assert page._run_btn.isEnabled()  # 失败后可重试


@pytest.mark.gui
def test_page_export_report(qtbot, template, tmp_path: Path, monkeypatch) -> None:
    """导出报告：按保存对话框路径写出 Markdown（含 data URI 曲线）."""
    from zylab.gui.qt_compat import QFileDialog

    page = TemplatePage()
    qtbot.addWidget(page)
    page.load_template(template)
    with qtbot.waitSignal(page.run_finished, timeout=10000):
        page.run()

    target = tmp_path / "报告.md"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *_a, **_k: (str(target), "Markdown 报告 (*.md)"))
    )
    page.export_report()
    text = target.read_text(encoding="utf-8")
    assert "# 页面模板" in text
    assert "## 参数" in text  # 参数概览表
    assert "![平方曲线](data:image/svg+xml;base64," in text
    assert "最大值" not in text  # 声明 report 时未引用的 text 结果不渲染


@pytest.mark.gui
def test_page_theme_requested(qtbot) -> None:
    """模板声明主题：加载时发 theme_requested 信号."""
    yaml_text = _YAML.replace('theme: ""', "theme: light")
    page = TemplatePage()
    qtbot.addWidget(page)
    with qtbot.waitSignal(page.theme_requested) as blocker:
        page.load_template(dsl_from_yaml(yaml_text))
    assert blocker.args == ["light"]


# ------------------------------------------------ 边界与兜底


_MINIMAL_YAML = """
meta: {id: t.min, name: 极简模板}
pipeline:
  - id: calc
    type: compute.expr
    params: {expr: "1 + 1"}
"""

_FORM_YAML = """
meta: {id: t.form, name: 表单模板}
params:
  输入:
    items:
      who: {value: demo}
      n: {value: 2, min: 0, max: 10}
pipeline:
  - id: calc
    type: compute.expr
    params: {expr: "n + 1"}
"""

_CLOUD_YAML = """
meta: {id: t.cloud, name: 云图模板}
pipeline:
  - id: calc
    type: compute.expr
    params: {expr: "1 + 1"}
results:
  - id: bad
    kind: text
    title: 坏引用
    text: "{v}"
    values: {v: "ghost.field"}
  - id: view
    kind: cloud
    title: 云图
    ref: calc
"""


@pytest.mark.gui
def test_param_form_text_and_rebuild(qtbot) -> None:
    """文本参数生成行编辑；重复 set_template 重建控件；禁用态覆盖全部输入."""
    form = DslParamForm()
    qtbot.addWidget(form)
    form.set_template(dsl_from_yaml(_FORM_YAML))
    assert form._edits["who"].text() == "demo"
    assert form.values() == {"who": "demo", "n": 2.0}
    old_edit = form._edits["who"]
    form.set_template(dsl_from_yaml(_MINIMAL_YAML))  # 二次加载重建
    assert form._edits == {} and form._spins == {}
    assert form._edits.get("who") is not old_edit
    form.set_template(dsl_from_yaml(_FORM_YAML))
    form.set_fields_enabled(False)
    assert not form._edits["who"].isEnabled()
    assert not form._spins["n"].isEnabled()


@pytest.mark.gui
def test_page_minimal_placeholder_and_html_export(qtbot, tmp_path: Path, monkeypatch) -> None:
    """无 results 声明：结果区占位；exports 缺省 html；无模板时运行/导出早退."""
    from zylab.gui.qt_compat import QFileDialog

    page = TemplatePage()
    qtbot.addWidget(page)
    page.run()  # 无模板：直接返回不进入运行态
    page.export_report()  # 无模板：直接返回
    assert not page._running

    page.load_template(dsl_from_yaml(_MINIMAL_YAML))
    assert page._tabs.count() == 1  # 无 results 声明 -> 单占位页
    with qtbot.waitSignal(page.run_finished, timeout=10000):
        page.run()
    assert page._tabs.count() == 1  # 运行后仍为占位
    target = tmp_path / "报告.html"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *_a, **_k: (str(target), "HTML 报告 (*.html)"))
    )
    page.export_report()
    html = target.read_text(encoding="utf-8")
    assert "<h1>极简模板</h1>" in html


@pytest.mark.gui
def test_page_load_template_file(qtbot, tmp_path: Path, monkeypatch) -> None:
    """文件对话框选择 YAML 模板加载；取消选择直接返回."""
    from zylab.gui.qt_compat import QFileDialog

    path = tmp_path / "t.yaml"
    path.write_text(_MINIMAL_YAML, encoding="utf-8")
    page = TemplatePage()
    qtbot.addWidget(page)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *_a, **_k: (str(path), "")))
    page.load_template_file()
    assert page._template is not None and page._template.id == "t.min"
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *_a, **_k: ("", "")))
    page.load_template_file()  # 取消：保持原模板不变
    assert page._template.id == "t.min"


@pytest.mark.gui
def test_page_run_worker_exception(qtbot, monkeypatch) -> None:
    """执行器外异常：线程边界兜底回传错误类型与消息."""
    page = TemplatePage()
    qtbot.addWidget(page)
    page.load_template(dsl_from_yaml(_MINIMAL_YAML))

    def boom(_executable):
        raise RuntimeError("执行环境异常")

    monkeypatch.setattr("zylab.gui.pages.template_page.run_workflow", boom)
    with qtbot.waitSignal(page.run_finished, timeout=10000) as blocker:
        page.run()
    _outputs, error = blocker.args
    assert "RuntimeError" in error and "执行环境异常" in error
    assert page._status_label.text() == "运行失败"


@pytest.mark.gui
def test_page_market_dialog(qtbot, monkeypatch, tmp_path: Path) -> None:
    """模板市场对话框列出全部 DSL 模板，确认选择后加载定制化界面."""
    from zylab.gui.widgets.template_dialog import TemplateDialog

    monkeypatch.setattr("zylab.core.config.default_data_dir", lambda: tmp_path)
    page = TemplatePage()
    qtbot.addWidget(page)
    shown: list[TemplateDialog] = []

    def fake_exec(dialog: TemplateDialog) -> int:
        shown.append(dialog)
        # 选中「函数逼近对比」并确认
        for i in range(dialog._tree.topLevelItemCount()):
            group = dialog._tree.topLevelItem(i)
            for j in range(group.childCount()):
                item = group.child(j)
                if item.text(0) == "函数逼近对比":
                    item.setSelected(True)
                    dialog._tree.setCurrentItem(item)
        dialog._on_accept()
        return 1

    monkeypatch.setattr("zylab.gui.pages.template_page.exec_dialog", fake_exec)
    page.open_market()
    assert len(shown) == 1
    assert page._template is not None and page._template.id == "dsl.math_compare"
    assert page._run_btn.isEnabled()
    assert page._tabs.count() == 2  # 对比曲线 + 摘要两结果页


@pytest.mark.gui
def test_page_market_empty(qtbot, monkeypatch, tmp_path: Path) -> None:
    """模板市场为空（无 DSL 模板）时提示且不弹对话框."""
    monkeypatch.setattr("zylab.core.config.default_data_dir", lambda: tmp_path)
    monkeypatch.setattr("zylab.gui.pages.template_page._builtin_dsl_templates", lambda: [])
    page = TemplatePage()
    qtbot.addWidget(page)
    messages: list[str] = []
    page.status_message.connect(messages.append)
    page.open_market()
    assert any("模板市场为空" in m for m in messages)
    assert page._template is None


@pytest.mark.gui
def test_page_grouped_results_single_tab(qtbot) -> None:
    """同组结果合并一页按块渲染；单页隐藏页签条；块级失败显示错误块."""
    from zylab.gui.qt_compat import QGroupBox
    from zylab.gui.widgets.dsl_result_view import DslGroupedResultView

    yaml_text = _YAML.replace("  - id: curve_y\n", "  - id: curve_y\n    group: 分析结果\n")
    yaml_text = yaml_text.replace("  - id: summary\n", "  - id: summary\n    group: 分析结果\n")
    page = TemplatePage()
    qtbot.addWidget(page)
    page.load_template(dsl_from_yaml(yaml_text))
    # 加载期即聚合为一页（组名页签），单页隐藏页签条
    assert page._tabs.count() == 1
    assert page._tabs.tabText(0) == "分析结果"
    assert page._tabs.tabBar().isHidden()

    with qtbot.waitSignal(page.run_finished, timeout=10000):
        page.run()
    assert page._tabs.count() == 1
    grouped = page._tabs.widget(0)
    assert isinstance(grouped, DslGroupedResultView)
    widgets = [grouped._layout.itemAt(i).widget() for i in range(grouped._layout.count())]
    boxes = [w for w in widgets if isinstance(w, QGroupBox)]
    assert [b.title() for b in boxes] == ["平方曲线", "摘要"]
    # 曲线块定高（同页多块不挤占）、摘要块文本渲染
    assert boxes[0].height() > 0


@pytest.mark.gui
def test_page_grouped_block_error_isolated(qtbot) -> None:
    """组内单块引用失败仅该块显示错误文本，其余块正常渲染."""
    from zylab.gui.qt_compat import QGroupBox
    from zylab.gui.widgets.dsl_result_view import DslGroupedResultView

    yaml_text = _YAML.replace("  - id: curve_y\n", "  - id: curve_y\n    group: 分析结果\n")
    yaml_text = yaml_text.replace("  - id: summary\n", "  - id: summary\n    group: 分析结果\n")
    yaml_text = yaml_text.replace("y: sweep.series.y", "y: sweep.series.ghost")  # 曲线引用失败
    page = TemplatePage()
    qtbot.addWidget(page)
    page.load_template(dsl_from_yaml(yaml_text))
    with qtbot.waitSignal(page.run_finished, timeout=10000):
        page.run()
    grouped = page._tabs.widget(0)
    assert isinstance(grouped, DslGroupedResultView)
    widgets = [grouped._layout.itemAt(i).widget() for i in range(grouped._layout.count())]
    boxes = [w for w in widgets if isinstance(w, QGroupBox)]
    assert [b.title() for b in boxes] == ["平方曲线", "摘要"]
    from zylab.gui.qt_compat import QLabel

    error_texts = [lab for lab in boxes[0].findChildren(QLabel) if lab.objectName() == "errorText"]
    assert error_texts and "ghost" in error_texts[0].text()
    # 摘要块正常渲染文本
    result_texts = [lab for lab in boxes[1].findChildren(QLabel) if lab.objectName() == "resultText"]
    assert result_texts and "最大值" in result_texts[0].text()


@pytest.mark.gui
def test_page_result_error_and_cloud_fallback(qtbot) -> None:
    """结果引用失败显示错误页；cloud 非解对象载荷回落解算视图错误提示."""
    from zylab.gui.widgets.result_view import ResultView

    page = TemplatePage()
    qtbot.addWidget(page)
    page.load_template(dsl_from_yaml(_CLOUD_YAML))
    with qtbot.waitSignal(page.run_finished, timeout=10000):
        page.run()
    assert page._tabs.count() == 2
    error_page = page._tabs.widget(0)
    assert isinstance(error_page, DslResultView)
    assert "ghost" in error_page._body.text()
    assert isinstance(page._tabs.widget(1), ResultView)  # 非解载荷 -> show_error 分支
