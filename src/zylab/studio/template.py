"""分析模板：预连节点图 + 参数默认值 + UI 呈现配置（声明式 JSON 可加载/校验）.

模板是「定制化计算工具」的配置载体：引用内置模块类型（:data:`BUILTIN_MODULES`）
实例化节点、声明连接与参数默认值、描述哪些参数暴露给用户（param_groups）、
哪些节点输出作为结果显示（results）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import LinkError, ParamError, StudioError, TemplateError
from .module import ModuleSpec, module_spec

__all__ = ["ParamGroup", "Template", "TemplateNode", "load_template", "save_template", "template_from_json"]


@dataclass(frozen=True)
class TemplateNode:
    """模板中的节点实例.

    :param id: 节点 id（模板内唯一）。
    :param type_id: 模块类型 id（引用内置模块表）。
    :param params: 参数覆盖表（缺省取模块默认值）。
    :param inputs: 入端口名 -> 上游引用（``"node_id.port_name"``）。
    """

    id: str
    type_id: str
    params: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ParamGroup:
    """参数分组（GUI 属性面板的组标题与参数引用列表）.

    :param title: 组标题（如 ``"几何与网格"``）。
    :param params: 参数引用（``"node_id.param_key"``）。
    """

    title: str
    params: tuple[str, ...]


@dataclass(frozen=True)
class Template:
    """分析模板（定制化计算工具的完整配置）.

    :param id: 模板 id（全局唯一，点分命名空间，如 ``"structural.cantilever_static"``）。
    :param name: 中文显示名。
    :param nodes: 节点实例表。
    :param discipline: 学科标识（如 ``"structural"``，模板库按此分组）。
    :param description: 用途说明。
    :param tags: 检索标签。
    :param param_groups: 暴露给用户的参数分组（缺省时 GUI 展示全部参数）。
    :param results: 结果节点 id 表（求解完成后默认展示这些节点的输出）。
    """

    id: str
    name: str
    nodes: tuple[TemplateNode, ...]
    discipline: str = "structural"
    description: str = ""
    tags: tuple[str, ...] = ()
    param_groups: tuple[ParamGroup, ...] = ()
    results: tuple[str, ...] = ()

    def node(self, node_id: str) -> TemplateNode:
        """按 id 取节点；不存在抛 :class:`TemplateError`."""
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise TemplateError(f"模板 {self.id!r} 无节点 {node_id!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Template:
        """由字典构造并校验模板；定义非法抛 :class:`TemplateError`/:class:`LinkError`."""
        try:
            nodes = tuple(
                TemplateNode(
                    id=str(raw["id"]),
                    type_id=str(raw["type"]),
                    params=dict(raw.get("params", {})),
                    inputs=dict(raw.get("inputs", {})),
                )
                for raw in _expect_list(data, "nodes", "<root>")
            )
            ui = data.get("ui", {})
            if not isinstance(ui, Mapping):
                raise TemplateError(f"模板 {data.get('id')!r} 的 'ui' 应为对象")
            groups = tuple(
                ParamGroup(title=str(g["title"]), params=tuple(str(p) for p in _expect_list(g, "params", "ui")))
                for g in ui.get("param_groups", [])
            )
            template = cls(
                id=str(data["id"]),
                name=str(data["name"]),
                nodes=nodes,
                discipline=str(data.get("discipline", "structural")),
                description=str(data.get("description", "")),
                tags=tuple(str(t) for t in data.get("tags", ())),
                param_groups=groups,
                results=tuple(str(r) for r in ui.get("results", ())),
            )
        except KeyError as exc:
            raise TemplateError(f"模板定义缺字段: {exc}") from exc
        except TypeError as exc:
            raise TemplateError(f"模板定义类型错误: {exc}") from exc
        template.validate()
        return template

    def validate(self) -> None:
        """校验模板：节点唯一性、模块类型、参数键与取值、连接类型匹配、无环、结果引用."""
        if not self.nodes:
            raise TemplateError(f"模板 {self.id!r} 不含任何节点")
        seen: set[str] = set()
        for n in self.nodes:
            if not n.id:
                raise TemplateError(f"模板 {self.id!r} 含空 id 节点")
            if n.id in seen:
                raise TemplateError(f"模板 {self.id!r} 节点 id 重复: {n.id!r}")
            seen.add(n.id)
            spec = module_spec(n.type_id)  # 未知模块类型在此抛出
            try:
                spec.coerce_params(n.params)
            except ParamError as exc:
                raise TemplateError(f"模板 {self.id!r} 节点 {n.id!r} 参数非法: {exc}") from exc
            self._validate_node_links(n, spec)
            for port in spec.inputs:
                if port.required and port.name not in n.inputs:
                    raise TemplateError(f"模板 {self.id!r} 节点 {n.id!r} 缺输入连接: {port.name!r}")
        for ref in self.results:
            self.node(ref)  # 结果引用必须存在
        for group in self.param_groups:
            for ref in group.params:
                self._validate_param_ref(ref)

    def _validate_node_links(self, node: TemplateNode, spec: ModuleSpec) -> None:
        """校验单个节点的全部输入连接（端口存在 + 引用格式 + 类型匹配）."""
        for port_name, ref in node.inputs.items():
            try:
                in_port = spec.input_port(port_name)
            except StudioError as exc:
                raise TemplateError(f"模板 {self.id!r} 节点 {node.id!r}: {exc}") from exc
            src_id, sep, src_port = ref.partition(".")
            if not sep or not src_id or not src_port:
                raise LinkError(f"模板 {self.id!r} 节点 {node.id!r} 连接 {ref!r} 应为 '节点id.端口名' 格式")
            if src_id == node.id:
                raise LinkError(f"模板 {self.id!r} 节点 {node.id!r} 不允许自连接 {ref!r}")
            try:
                src_spec = module_spec(self.node(src_id).type_id)
                out_port = src_spec.output_port(src_port)
            except StudioError as exc:
                raise LinkError(f"模板 {self.id!r} 节点 {node.id!r} 连接 {ref!r} 无效: {exc}") from exc
            if out_port.port_type is not in_port.port_type:
                raise LinkError(
                    f"模板 {self.id!r} 连接 {ref!r} -> {node.id}.{port_name} 端口类型不匹配: "
                    f"{out_port.port_type.value} != {in_port.port_type.value}"
                )

    def _validate_param_ref(self, ref: str) -> None:
        """校验 ``"node_id.param_key"`` 参数引用."""
        node_id, sep, key = ref.partition(".")
        if not sep or not node_id or not key:
            raise TemplateError(f"模板 {self.id!r} 参数引用 {ref!r} 应为 '节点id.参数键' 格式")
        try:
            module_spec(self.node(node_id).type_id).param(key)
        except StudioError as exc:
            raise TemplateError(f"模板 {self.id!r} 参数引用 {ref!r} 无效: {exc}") from exc

    def with_params(self, overrides: Mapping[str, Mapping[str, Any]]) -> Template:
        """以节点参数覆盖表生成新模板（如保存工作流图当前参数）.

        :param overrides: 节点 id -> 参数表（整体替换该节点 params）。
        """
        nodes = tuple(
            TemplateNode(
                id=n.id,
                type_id=n.type_id,
                params=dict(overrides.get(n.id, n.params)),
                inputs=dict(n.inputs),
            )
            for n in self.nodes
        )
        return Template(
            id=self.id,
            name=self.name,
            nodes=nodes,
            discipline=self.discipline,
            description=self.description,
            tags=self.tags,
            param_groups=self.param_groups,
            results=self.results,
        )

    def to_dict(self) -> dict[str, Any]:
        """序列化为可 :meth:`from_dict` 回读的字典."""
        return {
            "id": self.id,
            "name": self.name,
            "discipline": self.discipline,
            "description": self.description,
            "tags": list(self.tags),
            "nodes": [
                {"id": n.id, "type": n.type_id, "params": dict(n.params), "inputs": dict(n.inputs)} for n in self.nodes
            ],
            "ui": {
                "param_groups": [{"title": g.title, "params": list(g.params)} for g in self.param_groups],
                "results": list(self.results),
            },
        }


def _expect_list(data: Mapping[str, Any], key: str, where: str) -> list[Any]:
    """取字段并要求为列表."""
    value = data[key]
    if not isinstance(value, list):
        raise TemplateError(f"{where} 的 {key!r} 应为列表")
    return value


def template_from_json(text: str) -> Template:
    """由 JSON 文本解析模板；解析失败抛 :class:`TemplateError`."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TemplateError(f"模板 JSON 解析失败: {exc}") from exc
    if not isinstance(data, Mapping):
        raise TemplateError("模板 JSON 顶层应为对象")
    return Template.from_dict(data)


def load_template(path: Path) -> Template:
    """由 JSON 文件加载模板；文件缺失/解析失败抛 :class:`TemplateError`."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TemplateError(f"模板文件读取失败 {path}: {exc}") from exc
    try:
        return template_from_json(text)
    except TemplateError as exc:
        raise TemplateError(f"模板文件 {path.name} 非法: {exc}") from exc


def save_template(template: Template, path: Path) -> Path:
    """将模板写为 JSON 文件（UTF-8 无 BOM、保留中文），返回路径."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(template.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
