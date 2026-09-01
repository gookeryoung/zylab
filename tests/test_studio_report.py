"""studio.report DSL 报告生成器测试：Markdown/HTML 双载体 + SVG 曲线内嵌."""

from __future__ import annotations

import base64

import pytest

from zylab.studio.dsl import dsl_from_yaml
from zylab.studio.errors import TemplateError
from zylab.studio.report import build_html, build_markdown

_YAML = """
meta: {id: t.report, name: 悬臂梁扫参}
params:
  sweep:
    items:
      lo: {value: 0.0, unit: m}
      hi: {value: 2.0, unit: m}
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
    x_label: x
    y_label: x²
  - id: table_y
    kind: table
    title: 数值表
    columns: ["sweep.values", "sweep.series.y"]
  - id: summary
    kind: text
    title: 摘要
    text: "最大值 {max}"
    values: {max: "sweep.series.y.2"}
report:
  sections:
    - title: 曲线章节
      text: 按参数扫描绘制
      figure: curve_y
    - title: 数值章节
      table: table_y
    - title: 结论
      text: 扫描范围 {lo} ~ {hi} m
      figure: ""
"""


def _template():
    """加载扫参 DSL 模板."""
    return dsl_from_yaml(_YAML)


def _outputs(template):
    """运行扫参节点产出输出载荷."""
    from zylab.studio.nodes import compute_sweep

    return {"sweep": compute_sweep({}, template.node("sweep").params)}


def test_markdown_structure() -> None:
    """Markdown：标题/参数表/章节正文/曲线 data URI/管道表."""
    template = _template()
    md = build_markdown(template, _outputs(template))
    assert md.startswith("# 悬臂梁扫参")
    assert "## 参数" in md
    assert "| lo | 0 | m |" in md  # 参数概览含单位
    assert "## 曲线章节" in md
    assert "按参数扫描绘制" in md
    assert "![平方曲线](data:image/svg+xml;base64," in md
    assert "| 2 | 4 |" in md  # 数值表行
    assert "## 结论" in md
    assert "扫描范围 0.0 ~ 2.0 m" in md
    assert "最大值" not in md  # 声明 report 时未引用的 summary 不渲染


def test_markdown_svg_curve_payload() -> None:
    """data URI 可解码还原 SVG（折线 + 轴标签 + 图例）."""
    template = _template()
    md = build_markdown(template, _outputs(template))
    uri = next(line for line in md.splitlines() if line.startswith("!["))
    payload = uri.split("(", 1)[1].rstrip(")")
    svg = base64.b64decode(payload.split(",", 1)[1]).decode("utf-8")
    assert svg.startswith("<svg")
    assert svg.count("<polyline") == 1
    assert "x²" in svg  # 轴标签
    assert "y" in svg  # 图例序列名


def test_html_structure() -> None:
    """HTML：完整自包含文档（声明 + 内联样式 + 表格 + 内嵌图）."""
    template = _template()
    html = build_html(template, _outputs(template))
    assert html.startswith('<!DOCTYPE html><html lang="zh">')
    assert '<meta charset="utf-8">' in html
    assert "<title>悬臂梁扫参</title>" in html
    assert '<img alt="平方曲线" src="data:image/svg+xml;base64,' in html
    assert "<th>title</th>" not in html  # 表头用声明列名
    assert "<td>2</td><td>4</td>" in html
    assert "<style>" in html


def test_report_defaults_without_report_section() -> None:
    """无 report 声明：按 results 兜底（每结果一章节，文本并入正文）."""
    yaml_text = _YAML.replace(
        'report:\n  sections:\n    - title: 曲线章节\n      text: 按参数扫描绘制\n      figure: curve_y\n    - title: 数值章节\n      table: table_y\n    - title: 结论\n      text: 扫描范围 {lo} ~ {hi} m\n      figure: ""\n',
        "",
    )
    template = dsl_from_yaml(yaml_text)
    md = build_markdown(template, _outputs(template))
    assert "## 平方曲线" in md
    assert "## 数值表" in md
    assert "## 摘要" in md
    assert "最大值 4" in md  # 文本结果正文进入章节


def test_section_figure_unknown_result_rejected() -> None:
    """章节引用未声明的结果报错."""
    yaml_text = _YAML.replace("figure: curve_y", "figure: nope")
    template = dsl_from_yaml(yaml_text)
    with pytest.raises(TemplateError, match="引用未声明的结果"):
        build_markdown(template, _outputs(template))


def test_section_figure_kind_mismatch_rejected() -> None:
    """figure 引用非曲线结果报错."""
    yaml_text = _YAML.replace("figure: curve_y", "figure: table_y")
    template = dsl_from_yaml(yaml_text)
    with pytest.raises(TemplateError, match="应为曲线结果"):
        build_markdown(template, _outputs(template))


def test_empty_curve_rejected() -> None:
    """曲线无数据点报错."""
    yaml_text = _YAML.replace(
        '\n  - id: summary\n    kind: text\n    title: 摘要\n    text: "最大值 {max}"\n    values: {max: "sweep.series.y.2"}',
        "",
    )
    template = dsl_from_yaml(yaml_text)
    outputs = {"sweep": {"var": "x", "values": [], "series": {"y": []}}}
    with pytest.raises(TemplateError, match="无数据点"):
        build_markdown(template, outputs)


def test_section_text_missing_binding_rejected() -> None:
    """章节正文占位符缺参数绑定报错."""
    yaml_text = _YAML.replace("text: 扫描范围 {lo} ~ {hi} m", "text: 未知 {ghost}")
    template = dsl_from_yaml(yaml_text)
    with pytest.raises(TemplateError, match="正文格式化失败"):
        build_markdown(template, _outputs(template))


def test_html_escapes_markup() -> None:
    """章节正文含 HTML 标记时转义."""
    template = dsl_from_yaml(_YAML.replace("text: 按参数扫描绘制", "text: <b>加粗</b> & 注入"))
    html = build_html(template, _outputs(template))
    assert "&lt;b&gt;加粗&lt;/b&gt; &amp; 注入" in html
    assert "<b>加粗</b>" not in html


def test_section_table_kind_mismatch_rejected() -> None:
    """table 引用非表格结果报错."""
    yaml_text = _YAML.replace("table: table_y", "table: curve_y")
    template = dsl_from_yaml(yaml_text)
    with pytest.raises(TemplateError, match="应为表格结果"):
        build_markdown(template, _outputs(template))


def test_minimal_template_without_params_and_description() -> None:
    """无参数/无 description 模板：省略参数段，标题后直接结果章节."""
    yaml_text = """
meta: {id: t.min, name: 极简模板}
pipeline:
  - id: calc
    type: compute.expr
    params: {expr: "1 + 1"}
results:
  - id: out
    kind: text
    title: 输出
    text: "结果 {v}"
    values: {v: "calc"}
"""
    template = dsl_from_yaml(yaml_text)
    outputs = {"calc": 2}
    md = build_markdown(template, outputs)
    html = build_html(template, outputs)
    assert "## 参数" not in md
    assert "结果 2" in md
    assert "<h2>参数</h2>" not in html
    assert "<p>结果 2</p>" in html


def test_cloud_and_degenerate_curve_views() -> None:
    """cloud 结果占位文本 + 单点退化区间曲线（HTML/Markdown 双载体）."""
    yaml_text = (
        _YAML
        + """
  - id: cloud1
    kind: cloud
    title: 云图
    ref: sweep
  - id: flat
    kind: curve
    title: 常值曲线
    x: sweep.values
    y: sweep.series.y
"""
    )
    template = dsl_from_yaml(
        yaml_text.replace(
            'report:\n  sections:\n    - title: 曲线章节\n      text: 按参数扫描绘制\n      figure: curve_y\n    - title: 数值章节\n      table: table_y\n    - title: 结论\n      text: 扫描范围 {lo} ~ {hi} m\n      figure: ""\n',
            "",
        )
    )
    outputs = {"sweep": {"var": "x", "values": [1.0, 2.0, 3.0], "series": {"y": [4.0, 4.0, 4.0]}}}
    md = build_markdown(template, outputs)
    html = build_html(template, outputs)
    assert "云图结果 'sweep' 由应用界面渲染" in md
    assert "（云图结果 sweep 由应用界面渲染）" in html
    # 常值曲线 y 区间退化为 ±1 保护，仍可拼出 SVG
    assert "![常值曲线](data:image/svg+xml;base64," in md
    assert '<img alt="常值曲线"' in html
