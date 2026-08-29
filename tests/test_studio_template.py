"""studio.template 模板定义测试：from_dict 校验、JSON 加载、错误路径."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zylab.studio import (
    BUILTIN_TEMPLATES,
    LinkError,
    ModuleNotFoundError_,
    Template,
    TemplateError,
    load_template,
    template_from_json,
)

__all__ = []

#: 最小合法模板（源 -> 静力）
_MINIMAL = {
    "id": "test.minimal",
    "name": "最小模板",
    "nodes": [
        {"id": "model", "type": "example.truss2_two_bar"},
        {"id": "solve", "type": "analysis.static", "inputs": {"model": "model.model"}},
    ],
}


class TestFromDict:
    """字典构造与默认值."""

    def test_minimal_template(self) -> None:
        """最小模板取默认值（学科/描述/分组/结果）."""
        template = Template.from_dict(_MINIMAL)
        assert template.id == "test.minimal"
        assert template.discipline == "structural"
        assert template.param_groups == ()
        assert template.results == ()
        assert template.node("solve").inputs == {"model": "model.model"}

    def test_full_template(self) -> None:
        """完整字段（含 ui 分组与结果引用）."""
        raw = {
            **_MINIMAL,
            "description": "说明",
            "tags": ["入门"],
            "ui": {
                "param_groups": [{"title": "几何", "params": ["model.half_span", "model.rise"]}],
                "results": ["solve"],
            },
        }
        template = Template.from_dict(raw)
        assert template.description == "说明"
        assert template.tags == ("入门",)
        assert template.param_groups[0].params == ("model.half_span", "model.rise")
        assert template.results == ("solve",)

    def test_node_lookup_missing(self) -> None:
        """查询不存在的节点抛 TemplateError."""
        template = Template.from_dict(_MINIMAL)
        with pytest.raises(TemplateError, match="无节点"):
            template.node("ghost")


class TestValidateErrors:
    """定义非法的各错误路径."""

    def test_empty_nodes(self) -> None:
        """无节点模板非法."""
        with pytest.raises(TemplateError, match="不含任何节点"):
            Template.from_dict({"id": "t", "name": "t", "nodes": []})

    def test_duplicate_node_id(self) -> None:
        """节点 id 重复非法."""
        raw = {
            "id": "t",
            "name": "t",
            "nodes": [
                {"id": "a", "type": "example.truss2_two_bar"},
                {"id": "a", "type": "example.truss2_two_bar"},
            ],
        }
        with pytest.raises(TemplateError, match="节点 id 重复"):
            Template.from_dict(raw)

    def test_empty_node_id(self) -> None:
        """空节点 id 非法."""
        raw = {"id": "t", "name": "t", "nodes": [{"id": "", "type": "example.truss2_two_bar"}]}
        with pytest.raises(TemplateError, match="空 id"):
            Template.from_dict(raw)

    def test_unknown_module_type(self) -> None:
        """未知模块类型抛 ModuleNotFoundError_."""
        raw = {"id": "t", "name": "t", "nodes": [{"id": "a", "type": "no.such.type"}]}
        with pytest.raises(ModuleNotFoundError_, match="未知模块类型"):
            Template.from_dict(raw)

    def test_unknown_param_key(self) -> None:
        """节点参数键不在模块 schema 中."""
        raw = {
            "id": "t",
            "name": "t",
            "nodes": [{"id": "a", "type": "example.truss2_two_bar", "params": {"ghost": 1.0}}],
        }
        with pytest.raises(TemplateError, match="参数非法"):
            Template.from_dict(raw)

    def test_param_out_of_range(self) -> None:
        """节点参数取值越界."""
        raw = {
            "id": "t",
            "name": "t",
            "nodes": [{"id": "a", "type": "example.truss2_two_bar", "params": {"half_span": -1.0}}],
        }
        with pytest.raises(TemplateError, match="参数非法"):
            Template.from_dict(raw)

    def test_missing_required_field(self) -> None:
        """缺 name 字段抛 TemplateError."""
        with pytest.raises(TemplateError, match="缺字段"):
            Template.from_dict({"id": "t", "nodes": _MINIMAL["nodes"]})

    def test_nodes_not_list(self) -> None:
        """nodes 非列表抛 TemplateError."""
        with pytest.raises(TemplateError, match="应为列表"):
            Template.from_dict({"id": "t", "name": "t", "nodes": {}})

    def test_params_not_mapping(self) -> None:
        """节点 params 非键值结构抛 TemplateError（TypeError 包装）."""
        raw = {
            "id": "t",
            "name": "t",
            "nodes": [{"id": "a", "type": "example.truss2_two_bar", "params": [1, 2]}],
        }
        with pytest.raises(TemplateError, match="类型错误"):
            Template.from_dict(raw)

    def test_ui_not_mapping(self) -> None:
        """ui 非对象抛 TemplateError."""
        with pytest.raises(TemplateError, match="'ui' 应为对象"):
            Template.from_dict({**_MINIMAL, "ui": []})


class TestLinkValidation:
    """连接校验."""

    def _linked(self, inputs: dict, node_type: str = "analysis.static") -> dict:
        """构造 model->solve 两节点模板，solve 的 inputs 由参数给出."""
        return {
            "id": "t",
            "name": "t",
            "nodes": [
                {"id": "model", "type": "example.truss2_two_bar"},
                {"id": "solve", "type": node_type, "inputs": inputs},
            ],
        }

    def test_bad_ref_format(self) -> None:
        """连接引用缺端口名."""
        with pytest.raises(LinkError, match="格式"):
            Template.from_dict(self._linked({"model": "model"}))

    def test_self_connection(self) -> None:
        """自连接非法."""
        with pytest.raises(LinkError, match="自连接"):
            Template.from_dict(self._linked({"model": "solve.model"}))

    def test_unknown_source_node(self) -> None:
        """上游节点不存在."""
        with pytest.raises(LinkError, match="无效"):
            Template.from_dict(self._linked({"model": "ghost.model"}))

    def test_unknown_output_port(self) -> None:
        """上游端口不存在."""
        with pytest.raises(LinkError, match="无效"):
            Template.from_dict(self._linked({"model": "model.ghost"}))

    def test_unknown_input_port(self) -> None:
        """本节点输入端口不存在."""
        with pytest.raises(TemplateError, match="无输入端口"):
            Template.from_dict(self._linked({"ghost": "model.model"}))

    def test_port_type_mismatch(self) -> None:
        """端口类型不匹配（解端口接回模型输入）."""
        raw = {
            "id": "t",
            "name": "t",
            "nodes": [
                {"id": "model", "type": "example.truss2_two_bar"},
                {"id": "a", "type": "analysis.static", "inputs": {"model": "model.model"}},
                {"id": "b", "type": "analysis.static", "inputs": {"model": "a.solution"}},
            ],
        }
        with pytest.raises(LinkError, match="端口类型不匹配"):
            Template.from_dict(raw)


class TestUiReferences:
    """UI 呈现配置引用校验."""

    def test_result_ref_missing(self) -> None:
        """结果引用不存在的节点."""
        raw = {**_MINIMAL, "ui": {"results": ["ghost"]}}
        with pytest.raises(TemplateError, match="无节点"):
            Template.from_dict(raw)

    def test_param_ref_bad_format(self) -> None:
        """参数引用缺参数键."""
        raw = {**_MINIMAL, "ui": {"param_groups": [{"title": "g", "params": ["model"]}]}}
        with pytest.raises(TemplateError, match="格式"):
            Template.from_dict(raw)

    def test_param_ref_unknown_node(self) -> None:
        """参数引用不存在的节点."""
        raw = {**_MINIMAL, "ui": {"param_groups": [{"title": "g", "params": ["ghost.x"]}]}}
        with pytest.raises(TemplateError, match="无效"):
            Template.from_dict(raw)

    def test_param_ref_unknown_key(self) -> None:
        """参数引用模块 schema 外的键."""
        raw = {**_MINIMAL, "ui": {"param_groups": [{"title": "g", "params": ["model.ghost"]}]}}
        with pytest.raises(TemplateError, match="无效"):
            Template.from_dict(raw)


class TestJsonLoading:
    """JSON 文本与文件加载."""

    def test_from_json_roundtrip(self) -> None:
        """JSON 文本解析."""
        template = template_from_json(json.dumps(_MINIMAL))
        assert template.id == "test.minimal"

    def test_from_json_bad_syntax(self) -> None:
        """JSON 语法错误."""
        with pytest.raises(TemplateError, match="JSON 解析失败"):
            template_from_json("{oops")

    def test_from_json_top_level_not_object(self) -> None:
        """顶层非对象."""
        with pytest.raises(TemplateError, match="顶层应为对象"):
            template_from_json("[]")

    def test_load_template_file(self, tmp_path: Path) -> None:
        """文件加载成功."""
        path = tmp_path / "t.json"
        path.write_text(json.dumps(_MINIMAL), encoding="utf-8")
        assert load_template(path).id == "test.minimal"

    def test_load_template_missing_file(self, tmp_path: Path) -> None:
        """文件缺失."""
        with pytest.raises(TemplateError, match="读取失败"):
            load_template(tmp_path / "nope.json")

    def test_load_template_invalid_content(self, tmp_path: Path) -> None:
        """文件内容非法（错误消息含文件名）."""
        path = tmp_path / "bad.json"
        path.write_text("{oops", encoding="utf-8")
        with pytest.raises(TemplateError, match=r"bad\.json"):
            load_template(path)


class TestBuiltinTemplates:
    """内置模板表完整性."""

    def test_ids_unique(self) -> None:
        """内置模板 id 全局唯一."""
        ids = [t.id for t in BUILTIN_TEMPLATES]
        assert len(ids) == len(set(ids))

    def test_all_have_results(self) -> None:
        """内置模板均声明结果节点."""
        for template in BUILTIN_TEMPLATES:
            assert template.results, template.id
