"""低挂果补齐测试：expressions.py + template.py + cache.py 未覆盖分支.

这些测试不依赖 Qt/PySide，全部纯 Python；目标把 expressions 8 miss → 0、
cache 6 miss → 0、template 37 miss → 个位数。
"""

from __future__ import annotations

import ast
import pickle
from unittest.mock import patch

import pytest

from zylab.flowchart.cache import content_hash
from zylab.flowchart.errors import ParamError, TemplateError
from zylab.flowchart.expressions import expr_names, safe_eval
from zylab.flowchart.template import (
    OutputParam,
    Template,
    _parse_output_params,
    _parse_position,
)

__all__ = []


# cache pickle 分支用的模块级可序列化/不可序列化类
class _HashableObj:
    """模块级类实例可 pickle（局部类不行）."""

    def __init__(self, v):
        self.v = v


class _Unpicklable:
    def __reduce__(self):
        raise pickle.PicklingError("no way")


# ================================================================== expressions


class TestExprNames:
    """expr_names 错误/边界路径（miss at 33-34: SyntaxError → empty set）."""

    def test_invalid_syntax_returns_empty(self) -> None:
        assert expr_names("a +* b") == set()
        assert expr_names("") == set()
        assert expr_names("def f():") == set()

    def test_valid_expr_collects_names(self) -> None:
        assert expr_names("a + b * c") == {"a", "b", "c"}
        assert expr_names("sin(x) + y") == {"sin", "x", "y"}
        assert expr_names("1 + 2") == set()


class TestSafeEvalErrorPaths:
    """safe_eval 错误路径覆盖（miss at 160-161 TypeError/ValueError,
    以及 _validate_tree 四个防御分支通过 patch 触发）."""

    # ---- eval 运行时错误 ----

    def test_type_error_wrapped(self) -> None:
        """不兼容类型算术 → ParamError '表达式求值失败'."""
        with pytest.raises(ParamError, match="表达式求值失败"):
            safe_eval("a + b", {"a": 1, "b": "hello"})

    def test_value_error_wrapped(self) -> None:
        """数学域错误 (sqrt 负数) → ParamError '表达式求值失败'."""
        with pytest.raises(ParamError, match="表达式求值失败"):
            safe_eval("sqrt(-1)", {})

    # ---- AST 验证：BinOp 不允许的运算符 ----

    def test_binop_bitxor_disallowed(self) -> None:
        """ast.BitXor (^) 不在白名单，触发 BinOp 错误分支."""
        with pytest.raises(ParamError, match="表达式含不支持的运算符"):
            safe_eval("a ^ b", {"a": 1, "b": 1})

    def test_binop_matmul_disallowed(self) -> None:
        """ast.MatMult (@) 不在白名单."""
        with pytest.raises(ParamError, match="表达式含不支持的运算符"):
            safe_eval("a @ b", {"a": [[1]], "b": [[1]]})

    # ---- AST 验证：防御分支通过 patch 触发 ----

    def test_unaryop_disallowed_via_patch(self) -> None:
        with patch("zylab.flowchart.expressions._ALLOWED_UNARY", ()), pytest.raises(ParamError, match="不支持的运算符"):
            safe_eval("+x", {"x": 1})

    def test_compare_disallowed_via_patch(self) -> None:
        with patch(
            "zylab.flowchart.expressions._ALLOWED_CMPOPS",
            (ast.Lt,),
        ), pytest.raises(ParamError, match="不支持的比较运算"):
            safe_eval("x == 1", {"x": 1})

    def test_boolop_disallowed_via_patch(self) -> None:
        with patch("zylab.flowchart.expressions._ALLOWED_BOOLOPS", ()), pytest.raises(
            ParamError, match="不支持的布尔运算"
        ):
            safe_eval("x and y", {"x": 1, "y": 2})


# ================================================================== cache


class TestContentHashPickleFallback:
    """content_hash 对未知类型走 pickle 分支（miss at 126-132）."""

    def test_custom_class_uses_pickle_fallback(self) -> None:
        h1 = content_hash(_HashableObj(1))
        h2 = content_hash(_HashableObj(2))
        h3 = content_hash(_HashableObj(1))
        assert h1 == h3
        assert h1 != h2
        assert isinstance(h1, str) and len(h1) == 64  # SHA-256 hex

    def test_pickling_error_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="不支持类型"):
            content_hash(_Unpicklable())


# ================================================================== template


class TestParsePosition:
    """_parse_position 边界（miss at 319-320）."""

    def test_none_and_empty(self) -> None:
        assert _parse_position(None) is None
        assert _parse_position([]) is None

    def test_wrong_length(self) -> None:
        assert _parse_position([100]) is None
        assert _parse_position([100, 200, 300]) is None

    def test_non_numeric(self) -> None:
        assert _parse_position(["abc", 200]) is None
        assert _parse_position([100, None]) is None

    def test_valid(self) -> None:
        assert _parse_position([100, 200]) == (100.0, 200.0)
        assert _parse_position([100.5, 200.0]) == (100.5, 200.0)


class TestParseOutputParams:
    """_parse_output_params 各种输入格式与错误分支（miss at 339-366）."""

    def test_dict_shorthand(self) -> None:
        ops = _parse_output_params({"mass": "m1.mass", "disp": "m2.disp"}, "T001")
        assert len(ops) == 2
        assert ops[0].name == "mass" and ops[0].source == "m1.mass"

    def test_empty_dict_or_list(self) -> None:
        assert _parse_output_params({}, "T001") == []
        assert _parse_output_params([], "T001") == []

    def test_bad_top_level_type(self) -> None:
        with pytest.raises(TemplateError, match="应为列表或对象"):
            _parse_output_params(42, "T001")

    def test_bad_item_type(self) -> None:
        with pytest.raises(TemplateError, match="项应为对象"):
            _parse_output_params([123], "T001")

    def test_missing_name_or_source(self) -> None:
        with pytest.raises(TemplateError, match="须含 name \\+ source"):
            _parse_output_params([{}], "T001")
        with pytest.raises(TemplateError, match="须含 name \\+ source"):
            _parse_output_params([{"name": "x"}], "T001")

    def test_duplicate_name_in_list(self) -> None:
        with pytest.raises(TemplateError, match="输出参数名重复"):
            _parse_output_params(
                [
                    {"name": "x", "source": "a"},
                    {"name": "x", "source": "b"},
                ],
                "T001",
            )


class TestOutputParamValidate:
    """OutputParam.validate 错误分支（miss at 90-103）."""

    def test_empty_name(self) -> None:
        op = OutputParam(name="", source="m1.x")
        with pytest.raises(TemplateError, match="空 name"):
            op.validate("T001", {"m1"})

    def test_bad_source_format_no_dot(self) -> None:
        op = OutputParam(name="x", source="nodot")
        with pytest.raises(TemplateError, match="source 应为"):
            op.validate("T001", {"m1"})

    def test_bad_source_format_empty_parts(self) -> None:
        op = OutputParam(name="x", source=".field")
        with pytest.raises(TemplateError, match="source 应为"):
            op.validate("T001", {"m1"})
        op2 = OutputParam(name="x", source="node.")
        with pytest.raises(TemplateError, match="source 应为"):
            op2.validate("T001", {"m1"})

    def test_unknown_node_ref(self) -> None:
        op = OutputParam(name="x", source="unknown.field")
        with pytest.raises(TemplateError, match="未知节点"):
            op.validate("T001", {"m1"})

    def test_bad_expr_triggers_template_error(self) -> None:
        op = OutputParam(name="x", source="m1.x", expr="1/0")
        with pytest.raises(TemplateError, match="表达式非法"):
            op.validate("T001", {"m1"})

    def test_valid_param(self) -> None:
        op = OutputParam(name="x", source="m1.field")
        op.validate("T001", {"m1"})  # 不抛


class TestTemplateOutputParamsDuplicate:
    """Template.from_dict 校验 output_params 重复名（miss at 206-209）."""

    def _base(self) -> dict:
        return {
            "id": "T001",
            "name": "t",
            "nodes": [{"id": "m", "type": "example.cantilever_q4"}],
        }

    def test_duplicate_in_ui_output_params(self) -> None:
        data = self._base()
        data["ui"] = {
            "output_params": [
                {"name": "mass", "source": "m.mass"},
                {"name": "mass", "source": "m.mass2"},
            ]
        }
        with pytest.raises(TemplateError, match="输出参数名重复"):
            Template.from_dict(data)

    def test_to_dict_includes_output_params(self) -> None:
        data = self._base()
        data["ui"] = {
            "output_params": [
                {"name": "mass", "source": "m.mass", "unit": "kg", "label": "总质量"},
            ]
        }
        t = Template.from_dict(data)
        d = t.to_dict()
        ui = d["ui"]
        assert "output_params" in ui
        assert ui["output_params"][0]["name"] == "mass"
        assert ui["output_params"][0]["unit"] == "kg"

    def test_to_dict_skips_output_params_when_empty(self) -> None:
        t = Template.from_dict(self._base())
        d = t.to_dict()
        assert "output_params" not in d.get("ui", {})
