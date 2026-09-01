"""studio compute/post 节点测试：公式计算、参数扫参、静力结果提取 + DSL 集成."""

from __future__ import annotations

import numpy as np
import pytest

from zylab.studio.dsl import dsl_from_yaml
from zylab.studio.errors import ParamError, StudioError, TemplateError
from zylab.studio.module import ParamSpec, ParamType, module_spec
from zylab.studio.nodes import build_cantilever, compute_expr, compute_sweep, post_static, run_static

__all__ = []


# ------------------------------------------------ 参数规格 STR/MAP 扩展


def test_param_spec_str_coerce() -> None:
    """STR 参数收敛为字符串，非字符串拒绝."""
    spec = ParamSpec("expr", "表达式", ParamType.STR, "0")
    assert spec.coerce("a + b") == "a + b"
    with pytest.raises(ParamError, match="应为文本"):
        spec.coerce(123)


def test_param_spec_map_coerce() -> None:
    """MAP 参数校验为映射并拷贝为字典，非映射拒绝."""
    spec = ParamSpec("vars", "变量绑定", ParamType.MAP, {})
    coerced = spec.coerce({"a": 1})
    assert coerced == {"a": 1}
    with pytest.raises(ParamError, match="应为对象"):
        spec.coerce([1, 2])


def test_compute_modules_registered() -> None:
    """compute.expr/sweep 与 post.static 已注册且端口/参数类型正确."""
    for type_id, doc_key, doc_type in (
        ("compute.expr", "expr", ParamType.STR),
        ("compute.sweep", "body", ParamType.MAP),
        ("post.static", "expr", ParamType.STR),
    ):
        spec = module_spec(type_id)
        assert spec.param(doc_key).param_type is doc_type
    assert module_spec("compute.expr").output_port("data").port_type.value == "data"


# ------------------------------------------------ compute.expr


def test_compute_expr_scalar() -> None:
    """标量公式按 vars 绑定求值."""
    assert compute_expr({}, {"expr": "a + 2 * b", "vars": {"a": 1, "b": 2.5}}) == 6.0


def test_compute_expr_array_elementwise() -> None:
    """数组变量逐元素运算（列表收敛为 numpy 数组）."""
    result = compute_expr({}, {"expr": "x ** 2", "vars": {"x": [1, 2, 3]}})
    assert np.allclose(result, [1.0, 4.0, 9.0])


def test_compute_expr_array_functions() -> None:
    """数组命名空间的 linspace/sqrt 等构造与逐元素函数可用."""
    result = compute_expr({}, {"expr": "sqrt(linspace(0, 1, 3)) * 2"})
    assert np.allclose(result, [0.0, np.sqrt(0.5) * 2, 2.0])


def test_compute_expr_data_input_mapping() -> None:
    """data 输入为映射时逐项并入命名空间."""
    result = compute_expr({"data": {"x": np.array([1.0, 4.0])}}, {"expr": "sqrt(x)"})
    assert np.allclose(result, [1.0, 2.0])


def test_compute_expr_data_input_scalar_binding() -> None:
    """data 输入非映射时以 'data' 为名绑定."""
    assert compute_expr({"data": 5.0}, {"expr": "data * 2"}) == 10.0


def test_compute_expr_invalid_expr() -> None:
    """非法表达式报 ParamError."""
    with pytest.raises(ParamError):
        compute_expr({}, {"expr": "a + ", "vars": {}})


def test_compute_expr_param_type_checked() -> None:
    """expr 非文本 / vars 非对象在参数收敛期拒绝."""
    with pytest.raises(ParamError, match="应为文本"):
        compute_expr({}, {"expr": 123})
    with pytest.raises(ParamError, match="应为对象"):
        compute_expr({}, {"expr": "1", "vars": [1]})


# ------------------------------------------------ post.static


def _cantilever_static() -> object:
    """小网格悬臂梁静力解（8x2 Q4）."""
    model = build_cantilever({}, {"nx": 8, "ny": 2})
    return run_static({"model": model}, {})


def test_post_static_extract_tip_displacement() -> None:
    """表达式下标提取末端竖向位移，量级与梁理论解一致."""
    solution = _cantilever_static()
    value = post_static({"solution": solution}, {"expr": "displacements[-1, 1]"})
    # 理论端部挠度 P L^3 / (3 E I) ≈ -0.238 mm（P=-100N, L=40, E=2.1e5, I=1*8^3/12）
    assert value == pytest.approx(-0.238, rel=0.25)


def test_post_static_default_strain_energy() -> None:
    """默认提取应变能（正值）."""
    solution = _cantilever_static()
    assert post_static({"solution": solution}, {}) > 0.0


def test_post_static_rejects_wrong_payload() -> None:
    """输入非 StaticSolution 报 TypeError."""
    with pytest.raises(TypeError, match="StaticSolution"):
        post_static({"solution": {"fake": True}}, {})


# ------------------------------------------------ compute.sweep


def test_compute_sweep_math_curve() -> None:
    """纯数学扫参：body 内经 vars 绑定 $x 引用，逐点收集序列."""
    body = {
        "nodes": [{"id": "y", "type": "compute.expr", "params": {"expr": "x ** 2", "vars": {"x": "$x"}}}],
        "collect": ["y"],
    }
    result = compute_sweep({}, {"var": "x", "from": 0.0, "to": 1.0, "count": 5, "body": body})
    assert result["var"] == "x"
    assert result["values"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert result["series"]["y"] == [0.0, 0.0625, 0.25, 0.5625, 1.0]


def test_compute_sweep_cae_length() -> None:
    """CAE 扫参：悬臂梁长度扫描，端部挠度随 L 增大（近似 L^3）."""
    body = {
        "nodes": [
            {"id": "model", "type": "example.cantilever_q4", "params": {"length": "$L", "nx": 16, "ny": 4}},
            {"id": "static", "type": "analysis.static", "inputs": {"model": "model.model"}},
            {
                "id": "tip",
                "type": "post.static",
                "inputs": {"solution": "static.solution"},
                "params": {"expr": "displacements[-1, 1]"},
            },
        ],
        "collect": ["tip"],
    }
    result = compute_sweep({}, {"var": "L", "from": 40.0, "to": 80.0, "count": 3, "body": body})
    tips = [abs(v) for v in result["series"]["tip"]]
    assert len(tips) == 3
    assert tips[1] > tips[0]
    # 挠度 ∝ L^3：L 翻倍约 8 倍（粗网格允许宽裕系数）
    assert tips[2] > 4.0 * tips[0]


def test_compute_sweep_body_validation() -> None:
    """body 缺 nodes/collect 或含未声明 $ 引用均报错."""
    with pytest.raises(ParamError, match="nodes"):
        compute_sweep({}, {"var": "x", "from": 0.0, "to": 1.0, "count": 2, "body": {}})
    body = {"nodes": [{"id": "y", "type": "compute.expr", "params": {"expr": "1"}}]}
    with pytest.raises(ParamError, match="collect"):
        compute_sweep({}, {"var": "x", "from": 0.0, "to": 1.0, "count": 2, "body": body})
    bad_ref = {
        "nodes": [{"id": "y", "type": "compute.expr", "params": {"expr": "x", "vars": {"x": "$missing"}}}],
        "collect": ["y"],
    }
    with pytest.raises(StudioError, match="missing"):
        compute_sweep({}, {"var": "x", "from": 0.0, "to": 1.0, "count": 2, "body": bad_ref})


# ------------------------------------------------ DSL 集成（sweep body 保留 + 重绑定）


_SWEEP_YAML = """
meta: {id: t.sweep, name: 扫参模板}
params:
  sweep:
    items:
      lo: {value: 0.0}
      hi: {value: 1.0}
      n: {value: 5}
pipeline:
  - id: sweep
    type: compute.sweep
    params:
      var: x
      from: "$lo"
      to: "$hi"
      count: "$n"
      body:
        nodes:
          - id: y
            type: compute.expr
            params: {expr: "x ** 2", vars: {x: "$x"}}
        collect: ["y"]
"""


def test_dsl_sweep_body_kept_raw() -> None:
    """DSL 构造期 sweep 自身参数代入默认值，body 内 $x 引用保留原始文本."""
    template = dsl_from_yaml(_SWEEP_YAML)
    node = template.node("sweep")
    assert node.params["from"] == 0.0
    assert node.params["count"] == 5
    body_node = node.params["body"]["nodes"][0]
    assert body_node["params"]["vars"]["x"] == "$x"


def test_dsl_sweep_bind_and_run() -> None:
    """bind_params 重绑定扫描范围后可直接执行扫参节点."""
    template = dsl_from_yaml(_SWEEP_YAML)
    bound = template.bind_params({"hi": 2.0})
    result = compute_sweep({}, bound.node("sweep").params)
    assert result["values"] == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert result["series"]["y"] == [0.0, 0.25, 1.0, 2.25, 4.0]
    # 重绑定不破坏 body 原始引用
    assert bound.node("sweep").params["body"]["nodes"][0]["params"]["vars"]["x"] == "$x"


def test_dsl_sweep_body_shares_dsl_params() -> None:
    """body 内非扫描变量的 DSL 参数引用代入，$var 保留给运行期."""
    yaml_text = """
meta: {id: t.body, name: body 参数}
params:
  g:
    items:
      k: {value: 2.0}
      lo: {value: 0.0}
      hi: {value: 4.0}
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
            params: {expr: "k * x", vars: {x: "$x", k: "$k"}}
        collect: ["y"]
"""
    template = dsl_from_yaml(yaml_text)
    body_node = template.node("sweep").params["body"]["nodes"][0]
    assert body_node["params"]["vars"]["x"] == "$x"  # 扫描变量保留原样
    assert body_node["params"]["vars"]["k"] == 2.0  # DSL 参数代入默认值
    assert compute_sweep({}, template.node("sweep").params)["series"]["y"] == [0.0, 4.0, 8.0]
    bound = template.bind_params({"k": 3.0, "hi": 6.0})
    bound_node = bound.node("sweep").params["body"]["nodes"][0]
    assert bound_node["params"]["vars"]["k"] == 3.0  # 绑定期随用户输入更新
    assert bound_node["params"]["vars"]["x"] == "$x"
    result = compute_sweep({}, bound.node("sweep").params)
    assert result["values"] == [0.0, 3.0, 6.0]
    assert result["series"]["y"] == [0.0, 9.0, 18.0]


def test_dsl_sweep_body_unknown_ref_rejected() -> None:
    """body 内引用未声明的 DSL 参数报 TemplateError."""
    yaml_text = """
meta: {id: t.bad, name: 非法引用}
params:
  g:
    items:
      lo: {value: 0.0}
      hi: {value: 1.0}
pipeline:
  - id: sweep
    type: compute.sweep
    params:
      var: x
      from: "$lo"
      to: "$hi"
      count: 2
      body:
        nodes:
          - id: y
            type: compute.expr
            params: {expr: "x", vars: {x: "$ghost"}}
        collect: ["y"]
"""
    with pytest.raises(TemplateError, match="ghost"):
        dsl_from_yaml(yaml_text)


def test_dsl_deep_substitution_in_vars() -> None:
    """compute.expr 的 vars 嵌套 $ 引用在 DSL 层深度代入."""
    yaml_text = """
meta: {id: t.vars, name: 变量绑定}
params:
  base:
    items:
      a: {value: 3.0}
pipeline:
  - id: calc
    type: compute.expr
    params: {expr: "a * 10", vars: {a: "$a"}}
results: []
"""
    template = dsl_from_yaml(yaml_text)
    bound = template.bind_params({"a": 4.0})
    assert compute_expr({}, bound.node("calc").params) == 40.0
