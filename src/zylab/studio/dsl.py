"""分析模板 DSL：YAML 声明式模板的解析、校验与既有模板体系归一化.

DSL 模板是「定制化计算工具」的完整声明载体，在既有节点图模板
（:class:`~zylab.studio.template.Template`）之上扩展六类信息：

1. ``params``：参数化变量（声明式 schema，含单位/范围/表达式派生）；
2. ``pipeline``：计算过程（节点图，参数值可经 ``$name`` 引用 DSL 参数）；
3. ``results``：输出结果形式（curve/table/text/cloud 四种视图声明）；
4. ``report``：报告形式（章节 + 图表引用，导出 HTML/Markdown）；
5. ``docs``：图文说明（界面引导面板）；
6. ``theme``：界面主题绑定（主题名，指向 assets/themes 或用户主题目录）。

YAML 是 JSON 超集，同一解析管线兼容两类载体；``$name`` 引用在加载时以
参数声明默认值代入并走既有 :meth:`Template.validate` 全链路校验，
运行期按用户输入重新代入执行。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ParamError, TemplateError
from .expressions import expr_names, safe_eval
from .template import Template, TemplateNode

__all__ = [
    "DslDocs",
    "DslParam",
    "DslParamGroup",
    "DslReport",
    "DslReportSection",
    "DslResult",
    "DslTemplate",
    "dsl_from_yaml",
    "load_dsl",
    "substitute_refs",
]

logger = logging.getLogger(__name__)

#: 结果视图种类（P4 按种类分发渲染器）
_RESULT_KINDS = ("curve", "table", "text", "cloud")


@dataclass(frozen=True)
class DslParam:
    """DSL 参数化变量项.

    :param label: 中文显示名（缺省取参数名）。
    :param value: 默认值（数值或字符串）。
    :param unit: 单位标识（UI 后缀展示，如 ``m``/``Pa``）。
    :param min: 取值下限（None 不限制）。
    :param max: 取值上限（None 不限制）。
    :param step: UI 步进（None 用控件默认）。
    :param expr: 派生表达式（依赖其它参数，非空时为只读派生量）。
    """

    label: str = ""
    value: float | int | str | None = None
    unit: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    expr: str = ""

    @property
    def derived(self) -> bool:
        """是否表达式派生量（无独立默认值，按表达式求值）."""
        return bool(self.expr)


@dataclass(frozen=True)
class DslParamGroup:
    """DSL 参数分组（界面分组标题 + 有序参数项表）."""

    label: str
    items: tuple[tuple[str, DslParam], ...]

    def param(self, name: str) -> DslParam | None:
        """按名取参数项；不存在返回 None."""
        return dict(self.items).get(name)


@dataclass(frozen=True)
class DslResult:
    """DSL 结果声明（kind 决定渲染器与 spec 结构）.

    :param id: 结果项 id（模板内唯一）。
    :param kind: 视图种类（curve/table/text/cloud）。
    :param title: 结果页签标题。
    :param spec: 种类相关配置（curve 的 x/y、table 的 columns、cloud 的 ref/field）。
    """

    id: str
    kind: str
    title: str
    spec: dict[str, Any]


@dataclass(frozen=True)
class DslReportSection:
    """报告章节（标题 + 正文 + 可选图表引用）."""

    title: str
    text: str = ""
    figure: str = ""
    table: str = ""


@dataclass(frozen=True)
class DslReport:
    """报告形式声明（章节序列 + 导出格式）."""

    sections: tuple[DslReportSection, ...] = ()
    exports: tuple[str, ...] = ("html",)


@dataclass(frozen=True)
class DslDocs:
    """图文说明（模板应用页的引导面板）."""

    text: str = ""
    image: str = ""


@dataclass(frozen=True)
class DslTemplate(Template):
    """DSL 分析模板（节点图模板 + 参数/结果/报告/文档/主题扩展）.

    继承 :class:`Template` 使既有注册表、执行器对 DSL 模板零成本复用；
    新增字段承载 DSL 扩展信息，``pipeline`` 中的 ``$name`` 参数引用在
    构造时以声明默认值代入（校验期即合法）。
    """

    icon: str = ""
    theme: str = ""
    dsl_params: tuple[DslParamGroup, ...] = ()
    dsl_results: tuple[DslResult, ...] = ()
    report: DslReport | None = None
    docs: DslDocs | None = None
    #: 原始节点参数表（保留 ``$name`` 引用；构造期代入默认值会销毁引用，
    #: 重绑定须以本表为源），格式 ``(节点id, 原始参数表)`` 序列
    raw_params: tuple[tuple[str, dict[str, Any]], ...] = ()

    # ------------------------------------------------------------------ DSL 查询

    def dsl_param(self, name: str) -> DslParam | None:
        """按名查 DSL 参数（跨分组扁平命名空间）；不存在返回 None."""
        for group in self.dsl_params:
            param = group.param(name)
            if param is not None:
                return param
        return None

    def param_defaults(self) -> dict[str, float | int | str]:
        """全部非派生参数的默认值表（``$name`` 代入执行用）."""
        return {
            name: param.value
            for group in self.dsl_params
            for name, param in group.items
            if not param.derived and param.value is not None
        }

    def evaluate(self, values: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """求值完整参数命名空间（输入参数 + 表达式派生量）.

        :param values: 用户输入覆盖（缺省参数用声明默认值补齐）。
        :raises ParamError: 派生表达式非法、引用未声明变量或循环依赖。
        """
        merged: dict[str, Any] = {**self.param_defaults(), **dict(values or {})}
        derived: dict[str, DslParam] = {
            name: param for group in self.dsl_params for name, param in group.items if param.derived
        }
        resolved: dict[str, Any] = {}
        for name in derived:
            _resolve_derived(name, derived, merged, resolved, visiting=())
        merged.update(resolved)
        return merged

    def bind_params(self, values: Mapping[str, Any]) -> Template:
        """以用户参数值代入 ``$`` 引用生成可执行模板（节点参数整体重写）.

        以 :attr:`raw_params` 保留的原始引用为源重新代入（派生参数先经
        :meth:`evaluate` 求值，``$派生量`` 引用同样可绑定）；未声明 ``$``
        引用的节点参数保持构造期已代入的默认值。

        :param values: 参数名 -> 值（缺省参数用声明默认值补齐）。
        """
        merged = self.evaluate(values)
        raw_by_node = dict(self.raw_params)
        nodes = tuple(
            TemplateNode(
                id=n.id,
                type_id=n.type_id,
                params=_substitute_node_params(n.type_id, raw_by_node.get(n.id, n.params), merged, self.id),
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

    # ------------------------------------------------------------------ 解析

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> DslTemplate:
        """由字典构造并校验 DSL 模板；定义非法抛 :class:`TemplateError`."""
        try:
            template = cls._build(data)
        except KeyError as exc:
            raise TemplateError(f"DSL 模板定义缺字段: {exc}") from exc
        except TypeError as exc:
            raise TemplateError(f"DSL 模板定义类型错误: {exc}") from exc
        except ParamError as exc:
            raise TemplateError(f"DSL 模板派生参数非法: {exc}") from exc
        template.validate()
        return template

    @classmethod
    def _build(cls, data: Mapping[str, Any]) -> DslTemplate:
        """构造字段（校验由 :meth:`from_mapping` 统一执行）."""
        meta = _expect_mapping(data.get("meta", data), "meta")
        theme = str(data.get("theme", ""))
        dsl_params = _parse_param_groups(data.get("params", {}))
        defaults = {
            name: p.value for group in dsl_params for name, p in group.items if not p.derived and p.value is not None
        }
        # 派生参数以默认值命名空间求值（含 $派生量 引用的构造期代入）
        derived: dict[str, DslParam] = {name: p for group in dsl_params for name, p in group.items if p.derived}
        resolved: dict[str, Any] = {}
        for name in derived:
            _resolve_derived(name, derived, defaults, resolved, visiting=())
        defaults.update(resolved)
        pipeline = data.get("pipeline", data.get("nodes"))
        if pipeline is None:
            raise TemplateError("DSL 模板应含 'pipeline' 计算过程声明")
        pipeline_list = _expect_list(pipeline, "pipeline", "<root>")
        # 原始参数表先留存（$ 引用），再以默认值代入构造可校验节点
        raw_params = tuple((str(raw["id"]), dict(raw.get("params", {}))) for raw in pipeline_list)
        nodes = tuple(
            TemplateNode(
                id=str(raw["id"]),
                type_id=str(raw["type"]),
                params=_substitute_node_params(
                    str(raw["type"]), dict(raw.get("params", {})), defaults, str(meta.get("id", ""))
                ),
                inputs=dict(raw.get("inputs", {})),
            )
            for raw in pipeline_list
        )
        dsl_results = _parse_results(data.get("results", ()))
        return cls(
            id=str(meta["id"]),
            name=str(meta["name"]),
            nodes=nodes,
            discipline=str(meta.get("discipline", "general")),
            description=str(meta.get("description", "")),
            tags=tuple(str(t) for t in meta.get("tags", ())),
            param_groups=(),  # DSL 用 dsl_params 扁平命名空间，不复用节点参数引用
            results=tuple(r.spec["ref"] for r in dsl_results if r.kind == "cloud" and "ref" in r.spec),
            icon=str(meta.get("icon", "")),
            theme=theme,
            dsl_params=dsl_params,
            dsl_results=dsl_results,
            report=_parse_report(data.get("report")),
            docs=_parse_docs(data.get("docs")),
            raw_params=raw_params,
        )


# ---------------------------------------------------------------------- 解析辅助


def _parse_param_groups(raw: Any) -> tuple[DslParamGroup, ...]:
    """解析 params 分组声明（``{组键: {label, items: {名: 声明}}}``）."""
    if not raw:
        return ()
    if not isinstance(raw, Mapping):
        raise TemplateError("DSL 'params' 应为分组对象")
    groups: list[DslParamGroup] = []
    for key, value in raw.items():
        group = _expect_mapping(value, f"params.{key}")
        items_raw = _expect_mapping(group.get("items", {}), f"params.{key}.items")
        items: list[tuple[str, DslParam]] = []
        for name, raw_item in items_raw.items():
            item = {"value": raw_item} if isinstance(raw_item, (int, float, str)) else raw_item  # 简写：直接给默认值
            items.append((str(name), _parse_param(_expect_mapping(item, f"params.{key}.items.{name}"), str(name))))
        if not items:
            raise TemplateError(f"参数分组 {key!r} 不含任何参数")
        groups.append(DslParamGroup(label=str(group.get("label", key)), items=tuple(items)))
    return tuple(groups)


def _parse_param(raw: Mapping[str, Any], name: str) -> DslParam:
    """解析单个参数声明（数值字段须为数值，min<=max）.

    YAML 1.1 规范下无符号指数（``1.0e4``）解析为字符串，此处对可转数值的
    字符串做宽容转换（``float`` 成功即接受），降低模板作者书写负担。
    """
    value = _coerce_value(raw.get("value"))
    lo, hi = _bound(raw.get("min"), name, "min"), _bound(raw.get("max"), name, "max")
    if lo is not None and hi is not None and lo > hi:
        raise TemplateError(f"参数 {name!r} 范围非法: min={lo} > max={hi}")
    step = _as_float(raw.get("step"), f"参数 {name!r} step")
    return DslParam(
        label=str(raw.get("label", name)),
        value=value,
        unit=str(raw.get("unit", "")),
        min=lo,
        max=hi,
        step=step,
        expr=str(raw.get("expr", "")),
    )


def _bound(value: Any, name: str, which: str) -> float | None:
    """解析范围端点（None/数值/可转数值字符串）."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TemplateError(f"参数 {name!r} {which} 应为数值")
    result = _as_float(value, f"参数 {name!r} {which}")
    return None if result is None else result


def _as_float(value: Any, where: str) -> float | None:
    """宽容数值转换（None 透传；bool/不可转字符串报错）."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TemplateError(f"{where} 应为数值")
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise TemplateError(f"{where} 应为数值") from exc
    return float(value)


def _coerce_value(value: Any) -> float | int | str | None:
    """参数默认值宽容转换：可转数值的字符串转数值（``"2.1e5"``→``2.1e5``），其余原样."""
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _parse_results(raw: Any) -> tuple[DslResult, ...]:
    """解析 results 结果声明列表."""
    if not raw:
        return ()
    results: list[DslResult] = []
    seen: set[str] = set()
    for item in _expect_list(raw, "results", "<root>"):
        spec_raw = _expect_mapping(item, "results[]")
        kind = str(spec_raw.get("kind", ""))
        if kind not in _RESULT_KINDS:
            raise TemplateError(f"结果 kind {kind!r} 非法，应为 {list(_RESULT_KINDS)} 之一")
        rid = str(spec_raw.get("id", ""))
        if not rid:
            raise TemplateError("结果声明须含非空 id")
        if rid in seen:
            raise TemplateError(f"结果 id 重复: {rid!r}")
        seen.add(rid)
        spec = {k: v for k, v in spec_raw.items() if k not in ("id", "kind", "title")}
        _validate_result_spec(kind, spec, rid)
        results.append(DslResult(id=rid, kind=kind, title=str(spec_raw.get("title", rid)), spec=spec))
    return tuple(results)


def _validate_result_spec(kind: str, spec: Mapping[str, Any], rid: str) -> None:
    """按 kind 校验结果声明结构."""
    if kind == "curve" and not ("x" in spec and "y" in spec):
        raise TemplateError(f"曲线结果 {rid!r} 须声明 x/y 数据引用")
    if kind == "table" and "columns" not in spec:
        raise TemplateError(f"表格结果 {rid!r} 须声明 columns")
    if kind == "cloud" and "ref" not in spec:
        raise TemplateError(f"云图结果 {rid!r} 须声明 ref 节点引用")


def _parse_report(raw: Any) -> DslReport | None:
    """解析 report 报告声明."""
    if raw is None:
        return None
    data = _expect_mapping(raw, "report")
    sections = []
    for item in _expect_list(data.get("sections", []), "report.sections", "report"):
        section = _expect_mapping(item, "report.sections[]")
        sections.append(
            DslReportSection(
                title=str(section.get("title", "")),
                text=str(section.get("text", "")),
                figure=str(section.get("figure", "")),
                table=str(section.get("table", "")),
            )
        )
    exports = tuple(str(e) for e in data.get("exports", ("html",)))
    unknown = set(exports) - {"html", "md"}
    if unknown:
        raise TemplateError(f"报告导出格式 {sorted(unknown)} 不受支持（可选 html/md）")
    return DslReport(sections=tuple(sections), exports=exports)


def _parse_docs(raw: Any) -> DslDocs | None:
    """解析 docs 图文说明声明."""
    if raw is None:
        return None
    data = _expect_mapping(raw, "docs")
    intro = _expect_mapping(data.get("intro", {}), "docs.intro")
    return DslDocs(text=str(intro.get("text", "")), image=str(intro.get("image", "")))


def _substitute_node_params(
    type_id: str,
    params: Mapping[str, Any],
    values: Mapping[str, Any],
    template_id: str,
) -> dict[str, Any]:
    """代入节点参数 ``$`` 引用；compute.sweep 的 body 子图部分保留原始引用.

    body 中的 ``$var``（var 为扫描变量名）指向扫描变量（运行期由节点
    函数逐值代入），构造期与绑定期均保留原样，否则会销毁扫描语义；
    body 中的其余 ``$name`` 引用按 DSL 参数命名空间正常代入（绑定期
    随用户输入更新），扫参子图由此共享模板参数。
    """
    if type_id == "compute.sweep":
        head = {key: value for key, value in params.items() if key != "body"}
        result = substitute_refs(head, values, template_id)
        if "body" in params:
            var = str(params.get("var", ""))
            result["body"] = _substitute_body(params["body"], values, var, template_id)
        return result
    return substitute_refs(params, values, template_id)


def _substitute_body(
    body: Any,
    values: Mapping[str, Any],
    var: str,
    template_id: str,
) -> Any:
    """深度代入 body 内 DSL 参数引用；扫描变量 ``$var`` 引用保留原样.

    :param body: body 声明（映射/列表递归，字符串按 ``$`` 引用处理）。
    :param values: DSL 参数命名空间（构造期为默认值，绑定期为用户输入）。
    :param var: 扫描变量名（其 ``$var`` 引用保留给运行期逐值代入）。
    :param template_id: 模板 id（错误消息用）。
    """
    if isinstance(body, str) and body.startswith("$"):
        name = body[1:]
        if name == var:
            return body
        if name in values:
            return values[name]
        raise TemplateError(f"模板 {template_id!r} 引用未声明的参数 {name!r}")
    if isinstance(body, Mapping):
        return {key: _substitute_body(value, values, var, template_id) for key, value in body.items()}
    if isinstance(body, list):
        return [_substitute_body(item, values, var, template_id) for item in body]
    return body


def substitute_refs(params: Mapping[str, Any], values: Mapping[str, Any], template_id: str) -> dict[str, Any]:
    """深度代入 ``$name`` 引用（映射/列表递归，标量字符串命中即替换）.

    未声明或派生量引用报 :class:`TemplateError`；供 DSL 构造/绑定与
    compute.sweep 运行期子图代入共用。
    """
    return {key: _substitute_value(value, values, template_id) for key, value in params.items()}


def _substitute_value(value: Any, values: Mapping[str, Any], template_id: str) -> Any:
    """替换单值：``$name`` 字符串取声明值，容器递归，其余原样."""
    if isinstance(value, str) and value.startswith("$"):
        name = value[1:]
        if name not in values:
            raise TemplateError(f"模板 {template_id!r} 引用未声明的参数 {name!r}")
        return values[name]
    if isinstance(value, Mapping):
        return {key: _substitute_value(item, values, template_id) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute_value(item, values, template_id) for item in value]
    return value


def _resolve_derived(
    name: str,
    derived: Mapping[str, DslParam],
    inputs: dict[str, Any],
    resolved: dict[str, Any],
    visiting: tuple[str, ...],
) -> Any:
    """递归求值派生参数（依赖其它派生量时先解依赖；环报 ParamError）.

    :param name: 待求值派生参数名。
    :param derived: 全部派生参数声明表。
    :param inputs: 输入参数值（含默认值）。
    :param resolved: 已求值派生量缓存（就地更新）。
    :param visiting: 求值路径（环检测）。
    """
    if name in resolved:
        return resolved[name]
    if name in visiting:
        chain = " -> ".join([*visiting, name])
        raise ParamError(f"派生参数循环依赖: {chain}")
    param = derived[name]
    # 先解派生依赖（表达式引用的其它派生量），再以完整命名空间求值
    for dep in sorted(expr_names(param.expr) & set(derived)):
        if dep not in resolved:
            _resolve_derived(dep, derived, inputs, resolved, (*visiting, name))
    namespace = {**inputs, **resolved}
    try:
        value = safe_eval(param.expr, namespace)
    except ParamError as exc:
        raise ParamError(f"派生参数 {name!r} 求值失败: {exc}") from exc
    except NameError as exc:
        raise ParamError(f"派生参数 {name!r} 引用未声明变量: {exc}") from exc
    resolved[name] = value
    return value


def _expect_mapping(value: Any, where: str) -> Mapping[str, Any]:
    """要求值为映射对象."""
    if not isinstance(value, Mapping):
        raise TemplateError(f"{where} 应为对象，得到 {type(value).__name__}")
    return value


def _expect_list(data: Any, key: str, where: str) -> list[Any]:
    """要求数据为列表."""
    if not isinstance(data, list):
        raise TemplateError(f"{where} 的 {key!r} 应为列表")
    return data


# ---------------------------------------------------------------------- 载入


def dsl_from_yaml(text: str) -> DslTemplate:
    """由 YAML 文本解析 DSL 模板（YAML 是 JSON 超集，两类文本均可）."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise TemplateError(f"DSL 模板 YAML 解析失败: {exc}") from exc
    if not isinstance(data, Mapping):
        raise TemplateError("DSL 模板顶层应为对象")
    return DslTemplate.from_mapping(data)


def load_dsl(path: Path) -> DslTemplate:
    """由文件加载 DSL 模板（按扩展名分派 YAML/JSON 解析）."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TemplateError(f"DSL 模板文件读取失败 {path}: {exc}") from exc
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TemplateError(f"DSL 模板 JSON 解析失败: {exc}") from exc
        if not isinstance(data, Mapping):
            raise TemplateError("DSL 模板顶层应为对象")
        return DslTemplate.from_mapping(data)
    try:
        return dsl_from_yaml(text)
    except TemplateError as exc:
        raise TemplateError(f"DSL 模板文件 {path.name} 非法: {exc}") from exc
