"""studio.expressions 表达式安全求值测试 + DSL 派生参数求值集成测试."""

from __future__ import annotations

import numpy as np
import pytest

from zylab.studio.dsl import dsl_from_yaml
from zylab.studio.errors import ParamError, TemplateError
from zylab.studio.expressions import SAFE_MATH_NAMESPACE, expr_names, safe_eval

# ------------------------------------------------ safe_eval 基础


def test_safe_eval_arithmetic() -> None:
    """四则/幂/取模运算与括号优先级正确."""
    assert safe_eval("1 + 2 * 3") == 7
    assert safe_eval("(1 + 2) * 3") == 9
    assert safe_eval("2 ** 10") == 1024
    assert safe_eval("7 % 3") == 1
    assert safe_eval("-x + 5", {"x": 2}) == 3
    assert safe_eval("7 // 2") == 3


def test_safe_eval_math_functions() -> None:
    """白名单数学函数/常量可用."""
    assert safe_eval("sqrt(16)") == 4.0
    assert safe_eval("hypot(3, 4)") == 5.0
    assert safe_eval("sin(pi / 2)") == 1.0
    assert safe_eval("log(e)") == 1.0
    assert safe_eval("min(3, 1, 2)") == 1


def test_safe_eval_comparison_and_logic() -> None:
    """比较/布尔运算返回正确布尔值."""
    assert safe_eval("1 < 2") is True
    assert safe_eval("2 <= 1") is False
    assert safe_eval("1 < 2 and 3 > 2") is True
    assert safe_eval("not 1 == 1") is False


def test_safe_eval_rejects_dangerous_constructs() -> None:
    """属性访问/导入/lambda/推导式等危险构造一律拒绝（只读下标已解禁）."""
    for expr in (
        "__import__('os')",
        "().__class__",
        "[x for x in range(3)]",
        "lambda: 1",
        "(1).bit_length()",
        "open('x')",
        "x = 1",
        "x := 1",
    ):
        with pytest.raises(ParamError):
            safe_eval(expr, {"x": 1})


def test_safe_eval_subscript() -> None:
    """只读下标/切片/多维索引可用（列表/字典/numpy 数组）."""
    assert safe_eval("a[0] + a[-1]", {"a": [1, 2, 3]}) == 4
    assert safe_eval("m['k']", {"m": {"k": 5}}) == 5
    assert safe_eval("a[1:3]", {"a": [1, 2, 3, 4]}) == [2, 3]
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert safe_eval("m[1, 0]", {"m": matrix}) == 3.0
    assert safe_eval("m[-1, 1] * 2", {"m": matrix}) == 8.0


def test_safe_eval_array_aggregates() -> None:
    """数组聚合 amax/amin 可用（ARRAY_MATH 命名空间叠加后）."""
    from zylab.studio.expressions import ARRAY_MATH_NAMESPACE

    ns = {**ARRAY_MATH_NAMESPACE, "a": np.array([3.0, -1.0, 2.0])}
    assert safe_eval("amax(a)", ns) == 3.0
    assert safe_eval("amin(a)", ns) == -1.0
    assert safe_eval("amax(abs(a))", ns) == 3.0


def test_safe_eval_unknown_name_rejected() -> None:
    """未绑定名称求值报 ParamError（eval 阶段 NameError 转换）."""
    with pytest.raises(ParamError, match="未定义名称"):
        safe_eval("unknown_var + 1", {})


def test_safe_eval_syntax_error() -> None:
    """语法错误表达式报 ParamError."""
    with pytest.raises(ParamError, match="语法错误"):
        safe_eval("1 +")


def test_safe_eval_zero_division() -> None:
    """除零报 ParamError 而非裸异常."""
    with pytest.raises(ParamError, match="除零"):
        safe_eval("1 / 0")


def test_expr_names_extraction() -> None:
    """标识符提取覆盖函数名与变量名."""
    assert expr_names("a + b * c") == {"a", "b", "c"}
    assert expr_names("sqrt(x) + pi") == {"sqrt", "x", "pi"}
    assert expr_names("1 + 2") == set()


def test_safe_math_namespace_contents() -> None:
    """安全命名空间不含危险入口（open/exec/eval/import 等）."""
    for dangerous in ("open", "exec", "eval", "compile", "__import__", "input", "vars", "globals"):
        assert dangerous not in SAFE_MATH_NAMESPACE


# ------------------------------------------------ DSL 派生参数集成


def test_evaluate_derived_chain() -> None:
    """派生参数支持链式依赖（派生量引用派生量）与用户值覆盖."""
    yaml_text = """
meta: {id: t.chain, name: 链式派生}
params:
  base: {items: {b: 2.0}}
  derived:
    items:
      area: {expr: "b ** 2"}
      volume: {expr: "area * b"}
      scaled: {expr: "volume * 10"}
pipeline:
  - {id: n1, type: example.cantilever_q4, params: {}}
"""
    t = dsl_from_yaml(yaml_text)
    ns = t.evaluate()
    assert ns["area"] == 4.0
    assert ns["volume"] == 8.0
    assert ns["scaled"] == 80.0
    # 用户值覆盖后全链重算
    ns2 = t.evaluate({"b": 3.0})
    assert ns2["volume"] == 27.0
    assert ns2["scaled"] == 270.0


def test_evaluate_derived_math_functions() -> None:
    """派生表达式可调用白名单数学函数."""
    yaml_text = """
meta: {id: t.math, name: 数学派生}
params:
  geometry:
    items:
      b: {value: 6.0}
      h: {value: 8.0}
  derived:
    items:
      diagonal: {expr: "hypot(b, h)"}
pipeline:
  - {id: n1, type: example.cantilever_q4, params: {}}
"""
    t = dsl_from_yaml(yaml_text)
    assert t.evaluate()["diagonal"] == 10.0


def test_derived_cycle_rejected() -> None:
    """派生参数循环依赖在构造期即报错."""
    yaml_text = """
meta: {id: t.cycle, name: 循环}
params:
  derived:
    items:
      a: {expr: "b + 1"}
      b: {expr: "a + 1"}
pipeline:
  - {id: n1, type: example.cantilever_q4, params: {}}
"""
    with pytest.raises(TemplateError, match="循环依赖"):
        dsl_from_yaml(yaml_text)


def test_derived_unknown_var_rejected() -> None:
    """派生表达式引用未声明变量在构造期报错."""
    yaml_text = """
meta: {id: t.unknown, name: 未声明}
params:
  derived:
    items:
      a: {expr: "missing + 1"}
pipeline:
  - {id: n1, type: example.cantilever_q4, params: {}}
"""
    with pytest.raises(TemplateError, match=r"未声明变量|求值失败"):
        dsl_from_yaml(yaml_text)


def test_bind_params_with_derived_refs() -> None:
    """pipeline 的 $ 引用可绑定派生参数（用户改输入后派生量重算）."""
    yaml_text = """
meta: {id: t.bind, name: 派生绑定}
params:
  geometry:
    items:
      length: {value: 40.0}
      width: {value: 5.0}
  derived:
    items:
      area: {expr: "length * width"}
pipeline:
  - id: n1
    type: example.cantilever_q4
    params: {length: "$length"}
  - id: n2
    type: example.column_beam2
    params: {height: "$area", tip_load: -1.0}
"""
    t = dsl_from_yaml(yaml_text)
    bound = t.bind_params({})
    assert bound.node("n2").params["height"] == 200.0
    bound2 = t.bind_params({"width": 10.0})
    assert bound2.node("n2").params["height"] == 400.0  # 派生量随输入重算
