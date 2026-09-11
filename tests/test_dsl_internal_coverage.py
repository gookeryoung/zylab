"""dsl.py + lsc.py 内部函数补测，集中覆盖纯 Python 逻辑分支."""

from __future__ import annotations

import pytest

from zylab.flowchart import dsl
from zylab.flowchart.errors import TemplateError

__all__ = []


# ================================================================== dsl._parse_outputs


class TestParseOutputs:
    """dsl._parse_outputs 私有函数分支覆盖 (miss at 473-500)."""

    def test_empty_returns_empty(self) -> None:
        assert dsl._parse_outputs({}, "t.test") == ()
        assert dsl._parse_outputs(None, "t.test") == ()

    def test_string_shorthand(self) -> None:
        result = dsl._parse_outputs({"mass": "model.mass"}, "t.test")
        assert len(result) == 1
        assert result[0].name == "mass"
        assert result[0].source == "model.mass"

    def test_mapping_full(self) -> None:
        result = dsl._parse_outputs(
            {"sigma": {"source": "fea.sigma", "unit": "MPa", "label": "应力"}},
            "t.test",
        )
        assert result[0].name == "sigma"
        assert result[0].source == "fea.sigma"
        assert result[0].unit == "MPa"
        assert result[0].label == "应力"

    def test_duplicate_name_raises(self) -> None:
        # Python dict key 重复则后者覆盖前者；这里用 int+str 混合键 → str() 后重名
        with pytest.raises(TemplateError, match="重复"):
            dsl._parse_outputs({"1": "a.b", 1: "c.d"}, "t.test")

    def test_mapping_missing_source_raises(self) -> None:
        with pytest.raises(TemplateError, match="须含 source"):
            dsl._parse_outputs({"bad": {"unit": "kg"}}, "t.test")

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(TemplateError, match="应为字符串或对象"):
            dsl._parse_outputs({"bad": 42}, "t.test")

    def test_not_mapping_raises(self) -> None:
        with pytest.raises(TemplateError):
            dsl._parse_outputs(["bad", "list"], "t.test")


# ================================================================== dsl 其它 miss lines


class TestDslOtherMisses:
    """dsl.py 其它零散 miss 覆盖 (239, 592, 606-607, 622)."""

    def test_substitute_node_params_unknown_value_raises(self) -> None:
        """值中有 $引用但 value 表无匹配 → TemplateError."""
        with pytest.raises(TemplateError, match="引用未声明"):
            dsl._substitute_node_params("fea.modal", {"freq": "$f_target"}, {}, "t.test")

    def test_substitute_node_params_normal_value(self) -> None:
        """值中无 $引用 — 原样保留."""
        result = dsl._substitute_node_params("fea.static", {"nx": 4}, {}, "t.test")
        assert result["nx"] == 4
