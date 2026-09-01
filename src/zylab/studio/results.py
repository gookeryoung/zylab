"""DSL 结果视图数据解析：results 声明 + 节点输出载荷 -> 标准化视图数据.

四种视图声明的取数规则（引用格式 ``"节点id[.字段路径]"``，路径按映射
键逐级下行）：

- ``curve``：``x`` 引用取横轴序列；``y`` 单引用或引用列表（多序列，
  序列名取引用末段）；``x_label``/``y_label`` 可选；
- ``table``：``columns`` 为 ``{title, ref}`` 对象或引用简写（标题取
  末段）列表，各列序列按行转置成表；
- ``text``：``text`` 为 ``str.format`` 模板，``values`` 提供占位符 ->
  引用绑定；
- ``cloud``：``ref`` 指向解算节点（载荷为解对象，由既有云图视图渲染）。

解析失败的根因（节点未运行/路径不存在/长度不匹配）统一抛
:class:`TemplateError`，调用方（GUI/报告生成）按错误提示呈现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .dsl import DslResult
from .errors import TemplateError

__all__ = [
    "CloudData",
    "CurveData",
    "CurveSeries",
    "TableData",
    "TextData",
    "ViewData",
    "build_result",
]


@dataclass(frozen=True)
class CurveSeries:
    """单条曲线序列.

    :param name: 序列名（图例显示）。
    :param x: 横轴取值序列。
    :param y: 纵轴取值序列（与 x 等长）。
    """

    name: str
    x: tuple[Any, ...]
    y: tuple[Any, ...]


@dataclass(frozen=True)
class CurveData:
    """曲线视图数据.

    :param title: 视图标题。
    :param x_label: 横轴标签。
    :param y_label: 纵轴标签。
    :param series: 曲线序列表。
    """

    title: str
    x_label: str = ""
    y_label: str = ""
    series: tuple[CurveSeries, ...] = ()


@dataclass(frozen=True)
class TableData:
    """表格视图数据.

    :param title: 视图标题。
    :param columns: 列标题表。
    :param rows: 行数据（与列数等宽）。
    """

    title: str
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()


@dataclass(frozen=True)
class TextData:
    """文本视图数据.

    :param title: 视图标题。
    :param text: 正文（已格式化）。
    """

    title: str
    text: str = ""


@dataclass(frozen=True)
class CloudData:
    """云图视图数据（指向解算节点，载荷为解对象）.

    :param title: 视图标题。
    :param node_id: 解算节点 id。
    :param field: 场量声明（``temperature``/``voltage``/``displacement``/``stress``，
        缺省按载荷类型自适应）。
    :param cmap: 色带键（默认 ``jet``，见 :func:`zylab.fea.viewdata.cmap_keys`）。
    :param deform: 位移放大系数（位移场绘制变形轮廓，默认 1.0 真实变形）。
    :param payload: 节点输出载荷（解对象，报告 SVG 渲染用）。
    """

    title: str
    node_id: str = ""
    field: str = ""
    cmap: str = "jet"
    deform: float = 1.0
    payload: Any = None


#: 视图数据联合类型
ViewData = CurveData | TableData | TextData | CloudData


def build_result(result: DslResult, outputs: Mapping[str, Any]) -> ViewData:
    """按结果声明从节点输出载荷解析视图数据.

    :param result: DSL 结果声明（kind 分发）。
    :param outputs: 节点 id -> 输出载荷（运行结果缓存）。
    :raises TemplateError: 引用无法解析、序列长度不匹配或格式化缺占位符。
    """
    if result.kind == "curve":
        return _build_curve(result, outputs)
    if result.kind == "table":
        return _build_table(result, outputs)
    if result.kind == "text":
        return _build_text(result, outputs)
    ref = str(result.spec.get("ref", ""))
    node_id = ref.partition(".")[0]
    if node_id not in outputs:
        raise TemplateError(f"结果引用 {ref!r} 的节点 {node_id!r} 无输出（尚未运行）")
    return CloudData(
        title=result.title,
        node_id=node_id,
        field=str(result.spec.get("field", "")),
        cmap=str(result.spec.get("cmap", "jet")),
        deform=float(result.spec.get("deform", 1.0)),
        payload=outputs[node_id],
    )


def _build_curve(result: DslResult, outputs: Mapping[str, Any]) -> CurveData:
    """解析曲线声明：x 单序列 + y 单/多序列."""
    spec = result.spec
    x_values = _resolve_sequence(str(spec["x"]), outputs)
    y_raw = spec["y"]
    y_refs = [y_raw] if isinstance(y_raw, str) else [str(ref) for ref in y_raw]
    if not y_refs:
        raise TemplateError(f"曲线结果 {result.id!r} 的 y 引用为空")
    series: list[CurveSeries] = []
    for ref in y_refs:
        y_values = _resolve_sequence(ref, outputs)
        if len(y_values) != len(x_values):
            raise TemplateError(
                f"曲线结果 {result.id!r} 序列 {ref!r} 长度 {len(y_values)} 与 x 长度 {len(x_values)} 不匹配"
            )
        series.append(CurveSeries(name=ref.rpartition(".")[2], x=x_values, y=y_values))
    return CurveData(
        title=result.title,
        x_label=str(spec.get("x_label", "")),
        y_label=str(spec.get("y_label", "")),
        series=tuple(series),
    )


def _build_table(result: DslResult, outputs: Mapping[str, Any]) -> TableData:
    """解析表格声明：columns 引用各列序列，转置为行."""
    spec = result.spec
    columns_raw = spec["columns"]
    if not isinstance(columns_raw, list) or not columns_raw:
        raise TemplateError(f"表格结果 {result.id!r} 的 columns 应为非空列表")
    titles: list[str] = []
    column_values: list[tuple[Any, ...]] = []
    for entry in columns_raw:
        if isinstance(entry, Mapping):
            ref = str(entry["ref"])
            titles.append(str(entry.get("title", ref.rpartition(".")[2])))
        else:
            ref = str(entry)
            titles.append(ref.rpartition(".")[2])
        column_values.append(_resolve_sequence(ref, outputs))
    lengths = {len(values) for values in column_values}
    if len(lengths) > 1:
        raise TemplateError(f"表格结果 {result.id!r} 各列长度不一致: {sorted(lengths)}")
    rows = tuple(tuple(values[i] for values in column_values) for i in range(lengths.pop()))
    return TableData(title=result.title, columns=tuple(titles), rows=rows)


def _build_text(result: DslResult, outputs: Mapping[str, Any]) -> TextData:
    """解析文本声明：str.format 模板 + 占位符引用绑定."""
    spec = result.spec
    template = str(spec.get("text", ""))
    bindings = spec.get("values", {})
    if not isinstance(bindings, Mapping):
        raise TemplateError(f"文本结果 {result.id!r} 的 values 应为对象")
    subs = {str(name): _resolve_path(str(ref), outputs) for name, ref in bindings.items()}
    try:
        text = template.format(**subs)
    except (KeyError, IndexError, ValueError) as exc:
        raise TemplateError(f"文本结果 {result.id!r} 格式化失败: {exc}") from exc
    return TextData(title=result.title, text=text)


# ------------------------------------------------------------------ 引用解析


def _resolve_path(ref: str, outputs: Mapping[str, Any]) -> Any:
    """按 ``"节点id.字段路径"`` 引用取值.

    路径逐级下行：映射按键取值；序列（list/tuple）按数字段取下标
    （支持负索引，如 ``sweep.series.y.-1`` 取末项）；其余对象（解
    对象等）按公开属性取值（如 ``solve.t_max``/``solve.times``）。
    段与当前值类型不匹配或不存在时报错。
    """
    node_id, _, rest = ref.partition(".")
    if node_id not in outputs:
        raise TemplateError(f"结果引用 {ref!r} 的节点 {node_id!r} 无输出（尚未运行）")
    value: Any = outputs[node_id]
    for segment in rest.split(".") if rest else ():
        value = _descend(value, segment, ref)
    return _plain(value)


def _descend(value: Any, segment: str, ref: str) -> Any:
    """路径单段下行（映射键/序列或数组索引/对象公开属性；不匹配/越界报错）."""
    if isinstance(value, Mapping) and segment in value:
        return value[segment]
    if isinstance(value, (list, tuple)) and _index_text(segment) and -len(value) <= int(segment) < len(value):
        return value[int(segment)]
    if isinstance(value, np.ndarray) and _index_text(segment):
        index = int(segment)
        if -value.shape[0] <= index < value.shape[0]:
            return value[index]
    if not isinstance(value, (Mapping, list, tuple, np.ndarray)) and not segment.startswith("_"):
        attribute = getattr(value, segment, None)
        if attribute is not None:
            return attribute
    raise TemplateError(f"结果引用 {ref!r} 无法解析: {segment!r} 不存在")


def _index_text(segment: str) -> bool:
    """段是否为合法整数文本（含负号；空串/单横杠均非）."""
    return segment.lstrip("-").isdigit()


def _resolve_sequence(ref: str, outputs: Mapping[str, Any]) -> tuple[Any, ...]:
    """按引用取序列值（数组/列表收敛为元组）.

    载荷已经 :func:`_resolve_path` 的 :func:`_plain` 收敛（ndarray ->
    list），此处只需处理 tuple/list，元素再做一次标量收敛。
    """
    value = _resolve_path(ref, outputs)
    if isinstance(value, (tuple, list)):
        return tuple(_plain(item) for item in value)
    raise TemplateError(f"结果引用 {ref!r} 应为序列，得到 {type(value).__name__}")


def _plain(value: Any) -> Any:
    """收敛 numpy 标量/零维数组为 Python 内建值（可 pickle/可格式化）."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
