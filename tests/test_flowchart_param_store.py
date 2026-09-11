"""Phase 3 参数中心化测试：OutputParam / ParameterStore / Template.output_params / DSL outputs.

覆盖范围：

1. OutputParam.validate 边界
2. ParameterStore.from_template 输入参数扁平收集
3. ParameterStore.resolve_all source 路径解析
4. ParameterStore.resolve_all + expr 二次变换 + 输入参数下划线别名
5. RunOutcome.resolve_outputs 集成链路
6. Template from_dict/to_dict 往返 + output_params 解析
7. DSL YAML outputs 声明解析
8. resolve_path 公开 API 基础测试
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from zylab.flowchart import (
    OutputParam,
    ParameterStore,
    Template,
    TemplateNode,
    load_template,
    save_template,
)
from zylab.flowchart.batch import run_workflow
from zylab.flowchart.dsl import DslTemplate, dsl_from_yaml
from zylab.flowchart.errors import FlowchartError, TemplateError
from zylab.flowchart.results import resolve_path

# ------------------------------------------------------------------ 夹具


@pytest.fixture()
def cantilever_template() -> Template:
    """最小悬臂梁静力参数化计算（两个节点 + OutputParam 声明）."""
    return Template(
        id="structural.test_cantilever",
        name="悬臂梁测试",
        nodes=(
            TemplateNode(
                id="cantilever",
                type_id="example.cantilever_q4",
                params={"length": 10.0, "height": 2.0, "tip_load": -10.0},
                inputs={},
            ),
            TemplateNode(
                id="solve",
                type_id="analysis.static",
                params={},
                inputs={"model": "cantilever.model"},
            ),
        ),
        discipline="structural",
        output_params=(
            OutputParam(
                name="max_displacement",
                source="solve.displacements",
                expr="float(amax(abs(value)))",
                unit="mm",
                label="最大位移",
            ),
            OutputParam(
                name="strain_energy",
                source="solve.strain_energy",
                unit="N·mm",
                label="应变能",
            ),
        ),
    )


# ================================================================== 1. OutputParam.validate 边界


class TestOutputParamValidate:
    def test_valid_source(self, cantilever_template: Template) -> None:
        """合法 source + 合法表达式通过校验."""
        known = {n.id for n in cantilever_template.nodes}
        for op in cantilever_template.output_params:
            op.validate(cantilever_template.id, known)

    def test_empty_name(self) -> None:
        """空 name 报错."""
        op = OutputParam(name="", source="solve.displacements")
        with pytest.raises(TemplateError, match="空 name"):
            op.validate("tpl", {"solve"})

    def test_bad_source_format(self) -> None:
        """缺路径段报错."""
        op = OutputParam(name="x", source="solve")
        with pytest.raises(TemplateError, match=r"应为.*节点id.字段路径"):
            op.validate("tpl", {"solve"})

    def test_unknown_source_node(self) -> None:
        """引用不存在的节点报错."""
        op = OutputParam(name="x", source="ghost.field")
        with pytest.raises(TemplateError, match="未知节点"):
            op.validate("tpl", set())

    def test_bad_expr_syntax(self) -> None:
        """非法表达式报错（lambda）."""
        op = OutputParam(name="x", source="solve.field", expr="lambda x: x")
        with pytest.raises(TemplateError, match="表达式非法"):
            op.validate("tpl", {"solve"})

    def test_missing_var_in_expr(self) -> None:
        """表达式引用未声明变量报错."""
        op = OutputParam(name="x", source="solve.field", expr="undefined_name + 1")
        with pytest.raises(TemplateError, match="表达式非法"):
            op.validate("tpl", {"solve"})


# ================================================================== 2. ParameterStore.from_template 输入参数扁平收集


class TestParameterStoreFromTemplate:
    def test_input_collection(self, cantilever_template: Template) -> None:
        """inputs 按 node_id.param_key 扁平化."""
        store = ParameterStore.from_template(cantilever_template)
        assert store.get_input("cantilever.length") == pytest.approx(10.0)
        assert store.get_input("cantilever.height") == pytest.approx(2.0)
        assert store.get_input("cantilever.tip_load") == pytest.approx(-10.0)

    def test_missing_input(self, cantilever_template: Template) -> None:
        """取不存在的输入参数抛 FlowchartError."""
        store = ParameterStore.from_template(cantilever_template)
        with pytest.raises(FlowchartError):
            store.get_input("ghost.param")

    def test_output_decl_collection(self, cantilever_template: Template) -> None:
        """outputs 从 template.output_params 收集."""
        store = ParameterStore.from_template(cantilever_template)
        assert set(store.outputs.keys()) == {"max_displacement", "strain_energy"}
        assert store.get_output_decl("max_displacement").source == "solve.displacements"
        assert store.get_output_decl("max_displacement").expr == "float(amax(abs(value)))"
        assert store.get_output_decl("max_displacement").unit == "mm"

    def test_explicit_override(self, cantilever_template: Template) -> None:
        """显式传入 output_params 覆盖 template 声明."""
        extra = {"custom": OutputParam(name="custom", source="solve.strain_energy")}
        store = ParameterStore.from_template(cantilever_template, output_params=extra)
        assert set(store.outputs.keys()) == {"custom"}
        assert store.get_input("cantilever.length") == pytest.approx(10.0)

    def test_no_output_params_template(self) -> None:
        """空 output_params 的参数化计算不抛错."""
        tpl = Template(
            id="test.empty_out",
            name="空输出",
            nodes=(
                TemplateNode(
                    id="src",
                    type_id="example.cantilever_q4",
                    params={"length": 5.0},
                    inputs={},
                ),
            ),
        )
        store = ParameterStore.from_template(tpl)
        assert store.inputs.get("src.length") == pytest.approx(5.0)
        assert store.outputs == {}


# ================================================================== 3. ParameterStore.resolve_all 基本 source 路径解析


class TestResolveAll:
    def test_dict_key(self) -> None:
        """source 解析 dict 键."""
        outputs = {"n1": {"k": 42}}
        store = ParameterStore()
        store.outputs["out"] = OutputParam(name="out", source="n1.k")
        store.resolve_all(outputs)
        assert store.resolved["out"] == 42

    def test_sequence_index(self) -> None:
        """source 解析列表索引 + 负索引."""
        outputs = {"n1": {"arr": [10, 20, 30, 40]}}
        store = ParameterStore()
        store.outputs["first"] = OutputParam(name="first", source="n1.arr.0")
        store.outputs["last"] = OutputParam(name="last", source="n1.arr.-1")
        store.resolve_all(outputs)
        assert store.resolved["first"] == 10
        assert store.resolved["last"] == 40

    def test_object_attribute(self) -> None:
        """source 解析公开属性."""

        class Obj:
            def __init__(self) -> None:
                self.t_max = 123.4

        outputs = {"n1": Obj()}
        store = ParameterStore()
        store.outputs["temp"] = OutputParam(name="temp", source="n1.t_max")
        store.resolve_all(outputs)
        assert store.resolved["temp"] == pytest.approx(123.4)

    def test_multi_segment_path(self) -> None:
        """多级路径下行."""
        outputs = {"n1": {"a": {"b": {"c": 99}}}}
        store = ParameterStore()
        store.outputs["deep"] = OutputParam(name="deep", source="n1.a.b.c")
        store.resolve_all(outputs)
        assert store.resolved["deep"] == 99

    def test_missing_source_node(self) -> None:
        """source 节点无输出时抛 FlowchartError."""
        store = ParameterStore()
        store.outputs["out"] = OutputParam(name="out", source="ghost.field")
        with pytest.raises(FlowchartError, match="尚未运行"):
            store.resolve_all({})

    def test_missing_path_segment(self) -> None:
        """路径段不存在时抛 FlowchartError."""
        outputs = {"n1": {"k": 1}}
        store = ParameterStore()
        store.outputs["out"] = OutputParam(name="out", source="n1.missing")
        with pytest.raises(FlowchartError, match="解析失败"):
            store.resolve_all(outputs)

    def test_get_output_decl_missing(self) -> None:
        """取不存在的输出参数声明抛 FlowchartError."""
        store = ParameterStore()
        with pytest.raises(FlowchartError):
            store.get_output_decl("ghost")

    def test_get_resolved_missing(self) -> None:
        """取未解析/不存在的 resolved 抛 FlowchartError."""
        store = ParameterStore()
        store.outputs["a"] = OutputParam(name="a", source="n1.field")
        with pytest.raises(FlowchartError):
            store.get_resolved("a")


# ================================================================== 4. ParameterStore + expr 二次变换 + 输入参数下划线别名


class TestOutputParamExpr:
    def test_expr_value_binding(self) -> None:
        """expr 中 value 绑定 source 原值."""
        outputs = {"n1": {"val": 5}}
        store = ParameterStore()
        store.outputs["doubled"] = OutputParam(name="doubled", source="n1.val", expr="value * 2")
        store.resolve_all(outputs)
        assert store.resolved["doubled"] == 10

    def test_expr_input_param_underscore_alias(self) -> None:
        """expr 通过下划线别名引用带点输入参数名."""
        store = ParameterStore()
        store.inputs["n1.length"] = 10.0
        store.outputs["scaled"] = OutputParam(name="scaled", source="n1.val", expr="value * n1_length")
        outputs = {"n1": {"val": 3}}
        store.resolve_all(outputs)
        assert store.resolved["scaled"] == pytest.approx(30.0)

    def test_expr_uses_array_math(self) -> None:
        """expr 可用数组数学函数（ARRAY_MATH_NAMESPACE 的 np.sqrt 支持 list）."""
        outputs = {"n1": {"vals": np.array([1.0, 4.0, 9.0])}}
        store = ParameterStore()
        store.outputs["roots"] = OutputParam(name="roots", source="n1.vals", expr="sqrt(value)")
        store.resolve_all(outputs)
        # resolve_path._plain 把 ndarray 转成 list，np.sqrt 对 list 会递归处理
        assert store.resolved["roots"] == pytest.approx([1.0, 2.0, 3.0])

    def test_expr_eval_failure(self) -> None:
        """expr 求值失败时抛 FlowchartError."""
        outputs = {"n1": {"val": 0}}
        store = ParameterStore()
        store.outputs["bad"] = OutputParam(name="bad", source="n1.val", expr="1 / value")
        with pytest.raises(FlowchartError, match="表达式求值失败"):
            store.resolve_all(outputs)

    def test_expr_float_conversion(self) -> None:
        """expr 里 float() 类型转换正常工作."""
        outputs = {"n1": {"val": 3.14}}
        store = ParameterStore()
        store.outputs["f"] = OutputParam(name="f", source="n1.val", expr="float(value)")
        store.resolve_all(outputs)
        assert store.resolved["f"] == pytest.approx(3.14)

    def test_expr_amax_abs_pattern(self) -> None:
        """amax(abs(value)) 模式（DSL 里常见的最大绝对值提取）."""
        outputs = {"n1": {"arr": np.array([-3.0, 1.5, -2.0, 0.5])}}
        store = ParameterStore()
        store.outputs["max_abs"] = OutputParam(name="max_abs", source="n1.arr", expr="float(amax(abs(value)))")
        store.resolve_all(outputs)
        assert store.resolved["max_abs"] == pytest.approx(3.0)


# ================================================================== 5. RunOutcome.resolve_outputs 集成链路


class TestRunOutcomeResolveOutputs:
    def test_full_integration(self, cantilever_template: Template) -> None:
        """完整链路：run_workflow → resolve_outputs 自动解析."""
        outcome = run_workflow(cantilever_template)
        assert outcome.succeeded
        resolved = outcome.resolve_outputs(cantilever_template)
        assert "max_displacement" in resolved
        assert "strain_energy" in resolved
        # 最大位移 > 0（悬臂梁受力必有位移）
        assert resolved["max_displacement"] > 0
        assert resolved["strain_energy"] > 0

    def test_no_output_params_returns_empty(self) -> None:
        """参数化计算未声明 output_params 时返回空 dict."""
        tpl = Template(
            id="test.empty",
            name="无输出参数",
            nodes=(TemplateNode(id="src", type_id="example.cantilever_q4", params={}, inputs={}),),
        )
        outcome = run_workflow(tpl)
        assert outcome.succeeded
        assert outcome.resolve_outputs(tpl) == {}

    def test_failed_workflow_rejects_resolve(self) -> None:
        """失败的工作流 resolve_outputs 抛 FlowchartError."""
        # cantilever_q4 length 合理值都能运行成功，构造一个节点运行失败的场景
        # 用 compute.sweep 不合法 body 让运行失败过于复杂；直接构造 outcome
        from zylab.flowchart.batch import NodeOutcome, RunOutcome

        outcome = RunOutcome(outcomes=(NodeOutcome(node_id="bad", name="坏节点", error="boom"),))
        tpl = Template(
            id="tpl.fail",
            name="失败",
            nodes=(TemplateNode(id="n1", type_id="example.cantilever_q4", params={}, inputs={}),),
            output_params=(OutputParam(name="x", source="n1.model.mesh.n_nodes"),),
        )
        with pytest.raises(FlowchartError, match="失败节点"):
            outcome.resolve_outputs(tpl)


# ================================================================== 6. Template output_params from_dict/to_dict 往返


class TestTemplateOutputParamsSerialization:
    def test_from_dict_list_form(self) -> None:
        """list 形式 output_params 解析."""
        data: dict[str, Any] = {
            "id": "tpl.with_outputs",
            "name": "带输出参数",
            "nodes": [
                {"id": "n1", "type": "example.cantilever_q4", "params": {"length": 5.0}, "inputs": {}},
            ],
            "ui": {
                "output_params": [
                    {"name": "n_nodes", "source": "n1.model.mesh.n_nodes", "unit": "nodes"},
                ]
            },
        }
        tpl = Template.from_dict(data)
        assert len(tpl.output_params) == 1
        assert tpl.output_params[0].name == "n_nodes"
        assert tpl.output_params[0].source == "n1.model.mesh.n_nodes"
        assert tpl.output_params[0].unit == "nodes"

    def test_from_dict_mapping_form(self) -> None:
        """简写 mapping 形式解析."""
        data: dict[str, Any] = {
            "id": "tpl.map_form",
            "name": "简写",
            "nodes": [
                {"id": "n1", "type": "example.cantilever_q4", "params": {"length": 5.0}, "inputs": {}},
            ],
            "ui": {
                "output_params": {
                    "size": "n1.model.mesh.n_nodes",
                }
            },
        }
        tpl = Template.from_dict(data)
        assert len(tpl.output_params) == 1
        assert tpl.output_params[0].name == "size"
        assert tpl.output_params[0].source == "n1.model.mesh.n_nodes"

    def test_to_dict_includes_output_params(self) -> None:
        """to_dict 序列化包含 output_params."""
        tpl = Template(
            id="tpl.serialize",
            name="往返",
            nodes=(TemplateNode(id="n1", type_id="example.cantilever_q4", params={}, inputs={}),),
            output_params=(OutputParam(name="out", source="n1.model.mesh.n_nodes", unit="u", label="L", doc="D"),),
        )
        d = tpl.to_dict()
        assert "output_params" in d["ui"]
        op_d = d["ui"]["output_params"][0]
        assert op_d["name"] == "out"
        assert op_d["source"] == "n1.model.mesh.n_nodes"
        assert op_d["unit"] == "u"
        assert op_d["label"] == "L"
        assert op_d["doc"] == "D"
        assert op_d["expr"] == ""

    def test_roundtrip_via_json(self, tmp_path: Path) -> None:
        """JSON 往返一致性."""
        tpl = Template(
            id="tpl.roundtrip",
            name="往返测试",
            nodes=(
                TemplateNode(
                    id="c",
                    type_id="example.cantilever_q4",
                    params={"length": 20.0, "tip_load": -5.0},
                    inputs={},
                ),
            ),
            output_params=(
                OutputParam(name="a", source="c.model.mesh.n_nodes"),
                OutputParam(
                    name="b",
                    source="c.model.mesh.n_elements",
                    unit="个",
                    expr="value + 1",
                ),
            ),
        )
        path = save_template(tpl, tmp_path / "test.json")
        loaded = load_template(path)
        assert loaded.output_params == tpl.output_params
        assert loaded.nodes[0].params.get("length") == pytest.approx(20.0)

    def test_to_dict_skips_when_empty(self) -> None:
        """空 output_params 时 to_dict 不写该字段."""
        tpl = Template(
            id="tpl.no_op",
            name="无",
            nodes=(TemplateNode(id="n1", type_id="example.cantilever_q4", params={}, inputs={}),),
        )
        d = tpl.to_dict()
        assert "output_params" not in d.get("ui", {})

    def test_validate_duplicate_names(self) -> None:
        """重复输出参数名报错."""
        data: dict[str, Any] = {
            "id": "tpl.dup",
            "name": "重复",
            "nodes": [
                {"id": "n1", "type": "example.cantilever_q4", "params": {"length": 5.0}, "inputs": {}},
            ],
            "ui": {
                "output_params": [
                    {"name": "x", "source": "n1.model.mesh.n_nodes"},
                    {"name": "x", "source": "n1.model.mesh.n_elements"},
                ]
            },
        }
        with pytest.raises(TemplateError, match="重复"):
            Template.from_dict(data)

    def test_direct_dup_in_validate(self) -> None:
        """直接构造重复 output_params 触发 validate 防御性检查."""
        tpl = Template(
            id="tpl.direct_dup",
            name="dup",
            nodes=(TemplateNode(id="n1", type_id="example.cantilever_q4", params={}, inputs={}),),
            output_params=(
                OutputParam(name="x", source="n1.model.mesh.n_nodes"),
                OutputParam(name="x", source="n1.model.mesh.n_elements"),
            ),
        )
        with pytest.raises(TemplateError, match="重复"):
            tpl.validate()

    def test_output_params_garbage_type(self) -> None:
        """ui.output_params 非列表/对象时抛错."""
        data: dict[str, Any] = {
            "id": "tpl.bad",
            "name": "坏",
            "nodes": [{"id": "n1", "type": "example.cantilever_q4", "params": {}, "inputs": {}}],
            "ui": {"output_params": "not_a_list"},
        }
        with pytest.raises(TemplateError, match="应为列表或对象"):
            Template.from_dict(data)

    def test_output_params_list_item_not_mapping(self) -> None:
        """list 项不是 mapping 时报错."""
        data: dict[str, Any] = {
            "id": "tpl.bad2",
            "name": "坏2",
            "nodes": [{"id": "n1", "type": "example.cantilever_q4", "params": {}, "inputs": {}}],
            "ui": {"output_params": ["just_a_string"]},
        }
        with pytest.raises(TemplateError, match="项应为对象"):
            Template.from_dict(data)

    def test_output_params_list_item_missing_fields(self) -> None:
        """list 项缺 name 或 source 时报错."""
        data: dict[str, Any] = {
            "id": "tpl.bad3",
            "name": "坏3",
            "nodes": [{"id": "n1", "type": "example.cantilever_q4", "params": {}, "inputs": {}}],
            "ui": {"output_params": [{"name": "only_name"}]},
        }
        with pytest.raises(TemplateError, match="须含 name"):
            Template.from_dict(data)


# ================================================================== 7. DSL YAML outputs 声明


class TestDslOutputs:
    DSL_WITH_OUTPUTS = """
meta:
  id: dsl.test_outputs
  name: DSL 输出参数
  discipline: structural

params:
  geom:
    label: 几何
    items:
      L: 10.0
      H: 2.0
      tip_load: -10.0

pipeline:
  - id: cantilever
    type: example.cantilever_q4
    params:
      length: $L
      height: $H
      tip_load: $tip_load
  - id: solve
    type: analysis.static
    inputs:
      model: cantilever.model

outputs:
  max_displacement:
    source: solve.displacements
    expr: float(amax(abs(value)))
    unit: mm
    label: 最大位移
  strain_energy: solve.strain_energy
"""

    def test_dsl_outputs_parsed(self) -> None:
        """DSL YAML outputs 两种形式都正确解析."""
        tpl = dsl_from_yaml(self.DSL_WITH_OUTPUTS)
        assert isinstance(tpl, DslTemplate)
        assert len(tpl.output_params) == 2
        names = {op.name for op in tpl.output_params}
        assert names == {"max_displacement", "strain_energy"}
        # 完整对象形式
        md = next(op for op in tpl.output_params if op.name == "max_displacement")
        assert md.source == "solve.displacements"
        assert md.expr == "float(amax(abs(value)))"
        assert md.unit == "mm"
        assert md.label == "最大位移"
        # 简写字符串形式
        se = next(op for op in tpl.output_params if op.name == "strain_energy")
        assert se.source == "solve.strain_energy"
        assert se.expr == ""

    def test_dsl_outputs_validated(self) -> None:
        """DSL outputs 中 source 非法会在 validate 阶段报错."""
        bad_yaml = self.DSL_WITH_OUTPUTS.replace("solve.strain_energy", "ghost.path")
        with pytest.raises(TemplateError, match="未知节点"):
            dsl_from_yaml(bad_yaml)

    def test_dsl_no_outputs(self) -> None:
        """DSL 未声明 outputs 时 output_params 为空."""
        minimal = """
meta:
  id: dsl.minimal
  name: 最小
pipeline:
  - id: src
    type: example.cantilever_q4
params: {}
"""
        tpl = dsl_from_yaml(minimal)
        assert tpl.output_params == ()


# ================================================================== 8. resolve_path 公开 API 基础测试


class TestResolvePath:
    def test_basic(self) -> None:
        """公开 resolve_path 正常解析."""
        outputs = {"n1": {"a": {"b": [1, 2, 3]}}}
        assert resolve_path("n1.a.b.0", outputs) == 1
        assert resolve_path("n1.a.b.-1", outputs) == 3

    def test_node_missing(self) -> None:
        """节点无输出报 TemplateError."""
        with pytest.raises(TemplateError):
            resolve_path("ghost.field", {})
