"""ParameterStore 参数中心化存储测试."""

from __future__ import annotations

import pytest

from zylab.flowchart.errors import FlowchartError
from zylab.flowchart.param_store import ParameterStore
from zylab.flowchart.template import OutputParam

__all__ = []


class _FakeNode:
    def __init__(self, nid: str, params: dict) -> None:
        self.id = nid
        self.params = params


class _FakeTemplate:
    def __init__(self, output_params=()) -> None:
        self.nodes = [
            _FakeNode("m1", {"nx": 4, "ny": 2}),
            _FakeNode("m2", {"p": 1.5}),
        ]
        self.output_params = output_params


def test_from_template_collects_inputs() -> None:
    """from_template 正确扁平化收集节点输入参数."""
    store = ParameterStore.from_template(_FakeTemplate())
    assert store.inputs == {
        "m1.nx": 4,
        "m1.ny": 2,
        "m2.p": 1.5,
    }
    assert store.outputs == {}


def test_from_template_collects_output_declarations() -> None:
    """from_template 从 Template.output_params 收集输出声明."""
    ops = (
        OutputParam(name="mass", source="m1.mass"),
        OutputParam(name="disp", source="m2.disp"),
    )
    store = ParameterStore.from_template(_FakeTemplate(output_params=ops))
    assert set(store.outputs.keys()) == {"mass", "disp"}


def test_from_template_explicit_outputs_override() -> None:
    """显式传入 output_params 时完全覆盖 Template 内建声明."""
    store = ParameterStore.from_template(
        _FakeTemplate(output_params=(OutputParam(name="x", source="m1.x"),)),
        output_params={"y": OutputParam(name="y", source="m2.y")},
    )
    # 显式传入时 Template 内建被跳过
    assert "x" not in store.outputs
    assert "y" in store.outputs


def test_get_input_not_found_raises() -> None:
    """get_input 不存在时抛 FlowchartError."""
    store = ParameterStore.from_template(_FakeTemplate())
    with pytest.raises(FlowchartError, match="无输入参数"):
        store.get_input("nonexistent")


def test_get_output_decl_not_found_raises() -> None:
    """get_output_decl 不存在时抛 FlowchartError."""
    store = ParameterStore.from_template(_FakeTemplate())
    with pytest.raises(FlowchartError, match="无输出参数"):
        store.get_output_decl("missing")


def test_get_resolved_not_found_raises() -> None:
    """get_resolved 未解析时抛 FlowchartError."""
    ops = (OutputParam(name="x", source="m1.x"),)
    store = ParameterStore.from_template(_FakeTemplate(output_params=ops))
    with pytest.raises(FlowchartError, match="尚未解析"):
        store.get_resolved("x")


def test_resolve_all_basic_paths() -> None:
    """resolve_all 正确解析 source 路径到节点结果."""
    ops = (
        OutputParam(name="mass", source="m1.mass"),
        OutputParam(name="max_disp", source="m2.disp"),
    )
    store = ParameterStore.from_template(_FakeTemplate(output_params=ops))
    outputs = {
        "m1": {"mass": 12.3},
        "m2": {"disp": [0.1, 0.15, 0.05]},
    }
    resolved = store.resolve_all(outputs)
    assert resolved["mass"] == 12.3
    assert list(resolved["max_disp"]) == [0.1, 0.15, 0.05]
    # resolved 是同引用
    assert store.resolved is resolved


def test_resolve_all_source_not_found_raises() -> None:
    """resolve_all source 路径不存在时抛 FlowchartError."""
    ops = (OutputParam(name="x", source="m1.nonexistent"),)
    store = ParameterStore.from_template(_FakeTemplate(output_params=ops))
    with pytest.raises(FlowchartError, match="source 解析失败"):
        store.resolve_all({"m1": {"mass": 1.0}})


def test_resolve_all_with_expr_transform() -> None:
    """resolve_all 支持 expr 表达式变换（用 alias 输入参数）."""
    ops = (
        OutputParam(
            name="scaled_mass",
            source="m1.mass",
            expr="value * m1_nx",  # m1.nx → m1_nx alias
        ),
    )
    store = ParameterStore.from_template(_FakeTemplate(output_params=ops))
    resolved = store.resolve_all({"m1": {"mass": 10.0}})
    assert resolved["scaled_mass"] == 10.0 * 4  # m1_nx = 4


def test_resolve_all_expr_error_raises() -> None:
    """resolve_all expr 求值失败时抛 FlowchartError."""
    ops = (OutputParam(name="bad", source="m1.x", expr="1/0"),)
    store = ParameterStore.from_template(_FakeTemplate(output_params=ops))
    with pytest.raises(FlowchartError, match="表达式求值失败"):
        store.resolve_all({"m1": {"x": 1.0}})


def test_resolve_all_empty_outputs_ok() -> None:
    """无 output_params 时 resolve_all 返回空 dict."""
    store = ParameterStore.from_template(_FakeTemplate())
    resolved = store.resolve_all({})
    assert resolved == {}


class TestParameterStoreGetErrorPaths:
    """get_input / get_output_decl / get_resolved 失败分支."""

    def test_get_input_missing_raises(self) -> None:
        store = ParameterStore()
        with pytest.raises(FlowchartError, match="无输入参数"):
            store.get_input("nope.x")

    def test_get_output_decl_missing_raises(self) -> None:
        store = ParameterStore()
        with pytest.raises(FlowchartError, match="无输出参数"):
            store.get_output_decl("missing")

    def test_get_resolved_not_parsed_raises(self) -> None:
        store = ParameterStore()
        with pytest.raises(FlowchartError, match="尚未解析"):
            store.get_resolved("unparsed")
