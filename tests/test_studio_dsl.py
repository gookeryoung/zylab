"""studio.dsl DSL 模板测试：YAML 解析、校验、$ 参数代入与注册表接入."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from zylab.studio.dsl import DslParam, DslTemplate, dsl_from_yaml, load_dsl
from zylab.studio.errors import TemplateError
from zylab.studio.registry import TemplateRegistry

#: 最小合法 DSL 模板（悬臂梁静力，$ 引用绑定参数默认值）
_MINIMAL_YAML = """
meta:
  id: structural.cantilever_dsl
  name: 悬臂梁静力（DSL）
  discipline: structural
  description: YAML DSL 声明的悬臂梁静力分析
theme: light
params:
  geometry:
    label: 几何
    items:
      length: {label: 梁长, value: 40.0, unit: mm, min: 0.1, max: 1.0e4, step: 1.0}
      height: 8.0
  material:
    label: 材料
    items:
      e_modulus: {label: 弹性模量, value: 2.1e5, unit: MPa}
      poisson: {value: 0.3, min: 0.0, max: 0.49}
  derived:
    label: 派生
    items:
      inertia: {label: 惯性矩, expr: "height ** 3 / 12"}
pipeline:
  - id: model
    type: example.cantilever_q4
    params: {length: "$length", height: "$height", e_modulus: "$e_modulus", poisson: "$poisson"}
  - id: solve
    type: analysis.static
    inputs: {model: model.model}
results:
  - id: stress
    kind: cloud
    title: 应力云图
    ref: solve
    field: von_mises
  - id: summary
    kind: text
    title: 摘要
    text: 计算完成
report:
  sections:
    - title: 概述
      text: 梁长 $length mm 的静力分析。
      figure: stress
  exports: [html, md]
docs:
  intro: {text: 调整参数后点击运行, image: assets/docs/setup.png}
"""


def _minimal_dict() -> dict[str, Any]:
    """等价 JSON 结构（YAML 是 JSON 超集，二者解析结果应一致）."""
    return {
        "meta": {"id": "structural.cantilever_dsl", "name": "悬臂梁静力（DSL）", "discipline": "structural"},
        "params": {"geometry": {"label": "几何", "items": {"length": {"value": 40.0}, "height": 8.0}}},
        "pipeline": [
            {"id": "model", "type": "example.cantilever_q4", "params": {"length": "$length"}},
            {"id": "solve", "type": "analysis.static", "inputs": {"model": "model.model"}},
        ],
        "results": [{"id": "stress", "kind": "cloud", "ref": "solve"}],
    }


# ------------------------------------------------ 解析与校验


def test_parse_full_yaml() -> None:
    """全量 YAML 声明各节均正确解析为对应数据结构."""
    t = dsl_from_yaml(_MINIMAL_YAML)
    assert isinstance(t, DslTemplate)
    assert t.id == "structural.cantilever_dsl"
    assert t.theme == "light"
    # 参数分组：label 缺省取组键、items 简写展开
    labels = {g.label for g in t.dsl_params}
    assert labels == {"几何", "材料", "派生"}
    geo = t.dsl_params[0]
    assert geo.param("length") == DslParam(label="梁长", value=40.0, unit="mm", min=0.1, max=1.0e4, step=1.0)
    assert geo.param("height") == DslParam(label="height", value=8.0)
    assert t.dsl_param("inertia") is not None and t.dsl_param("inertia").derived
    # 结果声明：种类/标题/spec
    kinds = {r.kind for r in t.dsl_results}
    assert kinds == {"cloud", "text"}
    cloud = next(r for r in t.dsl_results if r.kind == "cloud")
    assert cloud.title == "应力云图" and cloud.spec["field"] == "von_mises"
    # 报告与文档
    assert t.report is not None
    assert t.report.exports == ("html", "md")
    assert t.report.sections[0].figure == "stress"
    assert t.docs is not None and t.docs.image == "assets/docs/setup.png"
    # 继承 Template：$ 引用已代入默认值，节点参数可被模块 schema 校验
    assert t.node("model").params["length"] == 40.0
    assert t.node("model").params["poisson"] == 0.3
    # cloud 结果映射为 Template.results（结果节点引用）
    assert t.results == ("solve",)


def test_yaml_json_equivalent() -> None:
    """YAML 与 JSON 载体共用同一解析管线."""
    t_yaml = dsl_from_yaml(_MINIMAL_YAML)
    t_json = DslTemplate.from_mapping(_minimal_dict())
    assert t_yaml.id == t_json.id
    assert t_yaml.node("model").params["length"] == t_json.node("model").params["length"] == 40.0


def test_param_defaults_and_binding() -> None:
    """param_defaults 提供非派生默认值；bind_params 以用户值重代入."""
    t = dsl_from_yaml(_MINIMAL_YAML)
    assert t.param_defaults() == {"length": 40.0, "height": 8.0, "e_modulus": 2.1e5, "poisson": 0.3}
    bound = t.bind_params({"height": 12.0})
    assert bound.node("model").params["height"] == 12.0
    assert bound.node("model").params["length"] == 40.0  # 未覆盖参数用默认值
    assert not isinstance(bound, DslTemplate)  # 绑定产物是纯执行模板


def test_undeclared_param_ref_rejected() -> None:
    """pipeline 引用未声明参数应报错."""
    text = _MINIMAL_YAML.replace('"$length"', '"$undeclared"')
    with pytest.raises(TemplateError, match="undeclared"):
        dsl_from_yaml(text)


def test_invalid_param_definitions() -> None:
    """参数声明非法（范围倒置/非数值边界/空分组）应报错."""
    base = _minimal_dict()
    base["params"]["geometry"]["items"]["length"] = {"value": 1.0, "min": 10.0, "max": 2.0}
    with pytest.raises(TemplateError, match="范围非法"):
        DslTemplate.from_mapping(base)
    base["params"]["geometry"]["items"]["length"] = {"value": 1.0, "min": "low"}
    with pytest.raises(TemplateError, match="min 应为数值"):
        DslTemplate.from_mapping(base)
    base["params"]["geometry"]["items"] = {}
    with pytest.raises(TemplateError, match="不含任何参数"):
        DslTemplate.from_mapping(base)


def test_invalid_results_and_report() -> None:
    """结果 kind/结构非法与报告导出格式非法均应报错."""
    base = _minimal_dict()
    base["results"] = [{"id": "r", "kind": "pie"}]
    with pytest.raises(TemplateError, match="非法"):
        DslTemplate.from_mapping(base)
    base["results"] = [{"id": "r", "kind": "curve"}]  # 曲线缺 x/y
    with pytest.raises(TemplateError, match="x/y"):
        DslTemplate.from_mapping(base)
    base["results"] = [{"id": "r", "kind": "table"}]  # 表格缺 columns
    with pytest.raises(TemplateError, match="columns"):
        DslTemplate.from_mapping(base)
    base["results"] = [{"id": "a", "kind": "cloud", "ref": "solve"}, {"id": "a", "kind": "text"}]
    with pytest.raises(TemplateError, match="重复"):
        DslTemplate.from_mapping(base)
    base["results"] = [{"id": "a", "kind": "cloud", "ref": "solve"}]
    base["report"] = {"sections": [], "exports": ["pdf"]}
    with pytest.raises(TemplateError, match="不受支持"):
        DslTemplate.from_mapping(base)


def test_missing_meta_and_pipeline() -> None:
    """缺 meta 必填字段或缺 pipeline 应报错."""
    with pytest.raises(TemplateError):
        DslTemplate.from_mapping({"meta": {"name": "x"}, "pipeline": []})
    with pytest.raises(TemplateError, match="pipeline"):
        DslTemplate.from_mapping({"meta": {"id": "a", "name": "b"}})


def test_pipeline_validated_against_modules() -> None:
    """pipeline 走既有模板校验：未知模块类型/缺输入连接/坏连接均报错."""
    from zylab.studio.errors import ModuleNotFoundError_

    base = _minimal_dict()
    base["pipeline"][0]["type"] = "example.nonexistent"
    with pytest.raises((TemplateError, ModuleNotFoundError_)):
        DslTemplate.from_mapping(base)
    base = _minimal_dict()
    base["pipeline"].append({"id": "solve2", "type": "analysis.static", "inputs": {}})
    with pytest.raises(TemplateError, match="缺输入连接"):
        DslTemplate.from_mapping(base)


def test_yaml_syntax_error() -> None:
    """YAML 语法错误应转 TemplateError."""
    with pytest.raises(TemplateError, match="YAML 解析失败"):
        dsl_from_yaml("meta: [unclosed")


def test_result_group_parsed() -> None:
    """结果声明的 group 字段解析为 DslResult.group 并从 spec 排除；缺省空串."""
    text = _MINIMAL_YAML.replace(
        "  - id: summary\n    kind: text", "  - id: summary\n    group: 分析结果\n    kind: text"
    )
    t = dsl_from_yaml(text)
    summary = next(r for r in t.dsl_results if r.id == "summary")
    assert summary.group == "分析结果"
    assert "group" not in summary.spec  # 非渲染配置，不进入 spec
    cloud = next(r for r in t.dsl_results if r.kind == "cloud")
    assert cloud.group == ""  # 未声明组保持独立页


# ------------------------------------------------ 文件与注册表接入


def test_load_dsl_dispatch_by_suffix(tmp_path: Path) -> None:
    """按扩展名分派：.yaml 走 YAML、.json 走 JSON；.yml 同 .yaml."""
    yml = tmp_path / "t.yaml"
    yml.write_text(_MINIMAL_YAML, encoding="utf-8")
    t = load_dsl(yml)
    assert t.id == "structural.cantilever_dsl"
    jsn = tmp_path / "t.json"
    import json

    jsn.write_text(json.dumps(_minimal_dict()), encoding="utf-8")
    assert load_dsl(jsn).id == t.id


def test_registry_loads_yaml_dir(tmp_path: Path) -> None:
    """注册表 load_dir 识别 YAML DSL 模板并与 JSON 模板同池注册."""
    (tmp_path / "structural").mkdir()
    (tmp_path / "structural" / "beam.yaml").write_text(_MINIMAL_YAML, encoding="utf-8")
    # 非法 YAML 文件跳过不阻断
    (tmp_path / "structural" / "bad.yaml").write_text("meta: [unclosed", encoding="utf-8")
    registry = TemplateRegistry()
    count = registry.load_dir(tmp_path)
    assert count == 1
    assert isinstance(registry.get("structural.cantilever_dsl"), DslTemplate)


# ------------------------------------------------ 声明边界（容错路径）


def test_dsl_param_query_miss_returns_none() -> None:
    """dsl_param 查询未声明参数返回 None（跨分组扁平命名空间）."""
    t = dsl_from_yaml(_MINIMAL_YAML)
    assert t.dsl_param("ghost") is None


def test_from_mapping_type_error_wrapped() -> None:
    """构造期 TypeError 包装为 TemplateError（不可哈希/不可迭代等坏声明）."""
    base = _minimal_dict()
    base["params"] = {"geometry": {"label": "几何", "items": None}}  # items 非 Mapping
    with pytest.raises(TemplateError):
        DslTemplate.from_mapping(base)


def test_params_non_mapping_rejected() -> None:
    """params 顶层非对象报错."""
    base = _minimal_dict()
    base["params"] = ["not", "a", "mapping"]
    with pytest.raises(TemplateError, match="params"):
        DslTemplate.from_mapping(base)


def test_bool_bound_rejected() -> None:
    """min/max/step 为布尔值报错（bool 是 int 子类，须显式拒绝）."""
    base = _minimal_dict()
    base["params"]["geometry"]["items"]["length"] = {"value": 1.0, "min": True}
    with pytest.raises(TemplateError, match="应为数值"):
        DslTemplate.from_mapping(base)
    base["params"]["geometry"]["items"]["length"] = {"value": 1.0, "step": False}
    with pytest.raises(TemplateError, match="应为数值"):
        DslTemplate.from_mapping(base)


def test_non_numeric_step_type_rejected() -> None:
    """step 为列表等非数值类型报错."""
    base = _minimal_dict()
    base["params"]["geometry"]["items"]["length"] = {"value": 1.0, "step": [1.0]}
    with pytest.raises(TemplateError, match="应为数值"):
        DslTemplate.from_mapping(base)


def test_empty_result_id_rejected() -> None:
    """结果声明缺 id 报错."""
    base = _minimal_dict()
    base["results"] = [{"kind": "cloud", "ref": "solve"}]
    with pytest.raises(TemplateError, match="非空 id"):
        DslTemplate.from_mapping(base)


def test_cloud_missing_ref_rejected() -> None:
    """cloud 结果缺 ref 报错."""
    base = _minimal_dict()
    base["results"] = [{"id": "r", "kind": "cloud"}]
    with pytest.raises(TemplateError, match="ref"):
        DslTemplate.from_mapping(base)


def test_sweep_without_body_params_kept() -> None:
    """compute.sweep 无 body 声明：头参数正常代入（body 分支跳过）."""
    text = _MINIMAL_YAML.replace(
        'params: {length: "$length", height: "$height", e_modulus: "$e_modulus", poisson: "$poisson"}',
        'params: {var: L, from: "$lo", to: "$hi", count: 3}',
    ).replace(
        "  - id: solve\n    type: analysis.static\n    inputs: {model: model.model}\n",
        "",
    )
    text = text.replace(
        "      e_modulus: {label: 弹性模量, value: 2.1e5, unit: MPa}\n",
        "      e_modulus: {label: 弹性模量, value: 2.1e5, unit: MPa}\n      lo: {value: 1.0}\n      hi: {value: 2.0}\n",
    )
    # 替换 model 节点类型为 compute.sweep，结果引用改为 model 节点
    text = text.replace("    type: example.cantilever_q4", "    type: compute.sweep")
    text = text.replace("    ref: solve", "    ref: model")
    t = dsl_from_yaml(text)
    assert t.node("model").params["from"] == 1.0  # 头参数代入默认值
    assert "body" not in t.node("model").params


def test_substitute_value_list_recursion() -> None:
    """$ 引用在列表值内逐项替换."""
    from zylab.studio.dsl import substitute_refs

    result = substitute_refs({"items": ["$a", "$b", 3.0], "nest": {"k": "$a"}}, {"a": 1, "b": 2}, "t.x")
    assert result == {"items": [1, 2, 3.0], "nest": {"k": 1}}


def test_derived_chain_cached_resolution() -> None:
    """派生链 b<-a 多次求值命中缓存（同值不再重复求值）."""
    text = _MINIMAL_YAML.replace(
        '      inertia: {label: 惯性矩, expr: "height ** 3 / 12"}',
        '      half: {expr: "height / 2"}\n      inertia: {label: 惯性矩, expr: "half ** 3 / 12"}',
    )
    t = dsl_from_yaml(text)
    namespace = t.evaluate({})
    assert namespace["half"] == 4.0
    assert namespace["inertia"] == pytest.approx(64.0 / 12.0)


def test_derived_name_error_wrapped() -> None:
    """派生表达式引用未声明变量包装为 TemplateError."""
    base = _minimal_dict()
    base["params"]["derived"] = {"label": "派生", "items": {"bad": {"expr": "no_such_var * 2"}}}

    with pytest.raises(TemplateError, match="引用未定义名称"):
        DslTemplate.from_mapping(base)


def test_yaml_top_level_non_mapping_rejected() -> None:
    """YAML 顶层非对象报错."""
    with pytest.raises(TemplateError, match="顶层应为对象"):
        dsl_from_yaml("- just\n- a\n- list\n")


def test_load_dsl_read_error_wrapped(tmp_path: Path) -> None:
    """文件读取失败（权限/不存在）包装 TemplateError."""
    with pytest.raises(TemplateError, match="读取失败"):
        load_dsl(tmp_path / "no_such_file.yaml")


def test_load_dsl_json_top_level_rejected(tmp_path: Path) -> None:
    """JSON 载体顶层非对象报错."""
    import json

    path = tmp_path / "t.json"
    path.write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(TemplateError, match="顶层应为对象"):
        load_dsl(path)


def test_load_dsl_json_syntax_error_wrapped(tmp_path: Path) -> None:
    """JSON 载体语法错误包装 TemplateError."""
    path = tmp_path / "t.json"
    path.write_text("{bad json", encoding="utf-8")
    with pytest.raises(TemplateError, match="JSON 解析失败"):
        load_dsl(path)


def test_load_dsl_yaml_error_wrapped_with_filename(tmp_path: Path) -> None:
    """YAML 文件模板非法时错误消息携带文件名."""
    path = tmp_path / "bad.yaml"
    path.write_text("meta: {id: a, name: b}\npipeline: []\n", encoding="utf-8")  # 空节点列表
    with pytest.raises(TemplateError, match=r"bad\.yaml 非法"):
        load_dsl(path)
