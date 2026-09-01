"""DSL 报告生成器：模板 + 节点输出 -> Markdown/HTML 报告（Qt-free）.

复用 :func:`zylab.studio.results.build_result` 把 ``results`` 声明解析为
标准化视图数据，再按报告载体拼装：

- 曲线（CurveData）→ 纯 Python 拼装的 SVG（零绘图依赖），Markdown 以
  base64 data URI 内嵌，HTML 直接内联；
- 表格（TableData）→ Markdown 管道表 / HTML ``<table>``；
- 文本（TextData）→ 段落；
- 云图（CloudData）→ 占位说明（云图由 GUI 解算视图渲染，报告以文字
  指明节点）。

报告结构（``report`` 声明）：标题 + 参数概览表 + 章节序列（标题/正文/
图表按 results id 引用）；未声明 report 时自动收录全部 results 兜底。
"""

from __future__ import annotations

import base64
from html import escape
from typing import Any, Mapping

from .dsl import DslReportSection, DslTemplate
from .errors import TemplateError
from .results import CurveData, TableData, TextData, ViewData, build_result

__all__ = ["build_html", "build_markdown"]

#: 曲线序列色表（与 GUI 图例配色风格一致的循环色）
_CURVE_COLORS = ("#4c8bf5", "#e4572e", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf")

_SVG_W = 640  # SVG 画布宽（像素）
_SVG_H = 400  # SVG 画布高（像素）
_SVG_PAD_L = 64  # 左边距（纵轴标签）
_SVG_PAD_R = 16
_SVG_PAD_T = 16
_SVG_PAD_B = 48  # 底边距（横轴标签 + 刻度）


def build_markdown(template: DslTemplate, outputs: Mapping[str, Any], values: Mapping[str, Any] | None = None) -> str:
    """生成 Markdown 报告.

    :param template: DSL 模板（meta/params/report 声明）。
    :param outputs: 节点 id -> 输出载荷（运行结果）。
    :param values: 用户输入参数（缺省用声明默认值）。
    :raises TemplateError: results 引用无法解析或章节引用缺失。
    """
    views = _render_views(template, outputs)
    lines: list[str] = [f"# {template.name}", ""]
    if template.description:
        lines += [template.description, ""]
    lines += _md_param_section(template, values)
    resolved = template.evaluate(values)
    for section, view in _report_sections(template, views):
        lines += [f"## {section.title}", ""]
        text = _format_section_text(section, resolved)
        if text:
            lines += [text, ""]
        if view is not None:
            lines += _md_view(view)
    return "\n".join(lines).rstrip() + "\n"


def build_html(template: DslTemplate, outputs: Mapping[str, Any], values: Mapping[str, Any] | None = None) -> str:
    """生成 HTML 报告（自包含单文件：内联样式 + data URI 图，可离线打开）.

    参数与结构同 :func:`build_markdown`。
    """
    views = _render_views(template, outputs)
    parts: list[str] = ["<h1>", escape(template.name), "</h1>"]
    if template.description:
        parts += ["<p>", escape(template.description), "</p>"]
    parts += _html_param_section(template, values)
    resolved = template.evaluate(values)
    for section, view in _report_sections(template, views):
        parts += ["<h2>", escape(section.title), "</h2>"]
        text = _format_section_text(section, resolved)
        if text:
            parts += ["<p>", escape(text), "</p>"]
        if view is not None:
            parts.append(_html_view(view))
    body = "".join(parts)
    return (
        '<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">'
        f"<title>{escape(template.name)}</title><style>{_CSS}</style></head>"
        f'<body class="report">{body}</body></html>'
    )


#: HTML 报告内联样式（浅色纸张风格，打印友好）
_CSS = (
    "body.report{font-family:'Microsoft YaHei',sans-serif;max-width:860px;"
    "margin:32px auto;padding:0 24px;color:#222;background:#fff}"
    "body.report h1{font-size:22px;border-bottom:2px solid #4c8bf5;padding-bottom:8px}"
    "body.report h2{font-size:17px;color:#333;margin-top:28px}"
    "body.report table{border-collapse:collapse;margin:12px 0}"
    "body.report th,body.report td{border:1px solid #ccc;padding:5px 14px;text-align:center}"
    "body.report th{background:#f2f5fa}"
    "body.report img{max-width:100%}"
)


# ------------------------------------------------------------------ 视图解析


def _render_views(template: DslTemplate, outputs: Mapping[str, Any]) -> dict[str, ViewData]:
    """把模板 results 声明整体解析为视图数据（id -> ViewData）."""
    return {result.id: build_result(result, outputs) for result in template.dsl_results}


def _report_sections(
    template: DslTemplate, views: Mapping[str, ViewData]
) -> list[tuple[DslReportSection, ViewData | None]]:
    """报告章节序列：声明优先，未声明 report 时按 results 兜底自拼章节."""
    if template.report is not None:
        sections: list[tuple[DslReportSection, ViewData | None]] = []
        for section in template.report.sections:
            view = _resolve_section_view(template, section, views)
            sections.append((section, view))
        return sections
    return [(DslReportSection(title=view.title), view) for view in views.values() if not isinstance(view, TextData)] + [
        (DslReportSection(title=view.title, text=view.text), None)
        for view in views.values()
        if isinstance(view, TextData)
    ]


def _resolve_section_view(
    template: DslTemplate, section: DslReportSection, views: Mapping[str, ViewData]
) -> ViewData | None:
    """按章节 figure/table 引用取视图数据（引用非法时报错）."""
    ref = section.figure or section.table
    if not ref:
        return None
    if ref not in views:
        raise TemplateError(f"模板 {template.id!r} 报告章节 {section.title!r} 引用未声明的结果 {ref!r}")
    view = views[ref]
    if section.figure and not isinstance(view, CurveData):
        raise TemplateError(f"报告章节 {section.title!r} 的 figure 引用 {ref!r} 应为曲线结果")
    if section.table and not isinstance(view, TableData):
        raise TemplateError(f"报告章节 {section.title!r} 的 table 引用 {ref!r} 应为表格结果")
    return view


# ------------------------------------------------------------------ 参数概览


def _format_section_text(section: DslReportSection, resolved: Mapping[str, Any]) -> str:
    """章节正文占位符绑定（``{name}`` -> 求值后的 DSL 参数）.

    结果数值引用走 ``results`` 的 text 声明（其 values 支持节点输出
    引用）；章节正文占位符只绑定参数命名空间，缺绑定报错。
    """
    if not section.text:
        return ""
    try:
        return section.text.format(**resolved)
    except (KeyError, IndexError, ValueError) as exc:
        raise TemplateError(f"报告章节 {section.title!r} 正文格式化失败: {exc}") from exc


def _md_param_section(template: DslTemplate, values: Mapping[str, Any] | None) -> list[str]:
    """Markdown 参数概览表（含派生量；无参数时输出空段）."""
    rows = _param_rows(template, values)
    if not rows:
        return []
    lines = ["## 参数", "", "| 参数 | 数值 | 单位 |", "| --- | --- | --- |"]
    lines += [f"| {escape(label)} | {_fmt(value)} | {unit} |" for label, value, unit in rows]
    return [*lines, ""]


def _html_param_section(template: DslTemplate, values: Mapping[str, Any] | None) -> list[str]:
    """HTML 参数概览表."""
    rows = _param_rows(template, values)
    if not rows:
        return []
    trs = "".join(
        f"<tr><td>{escape(label)}</td><td>{_fmt(value)}</td><td>{escape(unit)}</td></tr>" for label, value, unit in rows
    )
    return ["<h2>参数</h2><table><tr><th>参数</th><th>数值</th><th>单位</th></tr>", trs, "</table>"]


def _param_rows(template: DslTemplate, values: Mapping[str, Any] | None) -> list[tuple[str, Any, str]]:
    """参数行（label, value, unit）：求值完整命名空间后按声明顺序取值."""
    resolved = template.evaluate(values)
    return [
        (param.label or name, resolved[name], param.unit)
        for group in template.dsl_params
        for name, param in group.items
        if name in resolved
    ]


# ------------------------------------------------------------------ 单视图渲染


def _md_view(view: ViewData) -> list[str]:
    """Markdown 视图块（曲线图 data URI / 管道表 / 段落）."""
    if isinstance(view, CurveData):
        uri = _svg_data_uri(_curve_svg(view))
        return [f"![{view.title}]({uri})", ""]
    if isinstance(view, TableData):
        lines = [f"**{view.title}**", "", "| " + " | ".join(view.columns) + " |", "|" + " --- |" * len(view.columns)]
        lines += ["| " + " | ".join(_fmt(cell) for cell in row) + " |" for row in view.rows]
        return [*lines, ""]
    if isinstance(view, TextData):
        return [view.text, ""]
    return [f"（云图结果 {view.node_id!r} 由应用界面渲染）", ""]


def _html_view(view: ViewData) -> str:
    """HTML 视图块."""
    if isinstance(view, CurveData):
        return f'<img alt="{escape(view.title)}" src="{_svg_data_uri(_curve_svg(view))}">'
    if isinstance(view, TableData):
        head = "".join(f"<th>{escape(name)}</th>" for name in view.columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{escape(_fmt(cell))}</td>" for cell in row) + "</tr>" for row in view.rows
        )
        return f"<table><tr>{head}</tr>{body}</table>"
    if isinstance(view, TextData):
        return f"<p>{escape(view.text)}</p>"
    return f"<p>（云图结果 {escape(view.node_id)} 由应用界面渲染）</p>"


def _fmt(value: Any) -> str:
    """数值格式化（浮点 6 位有效数字，其余 str）."""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _svg_data_uri(svg: str) -> str:
    """SVG 文本转 base64 data URI（自包含可离线渲染）."""
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


# ------------------------------------------------------------------ 曲线 SVG


def _curve_svg(data: CurveData) -> str:
    """纯 Python 拼装曲线 SVG（多序列折线 + 轴标签 + 图例）.

    无绘图库依赖：坐标按全部序列的取值范围归一化到画布（退化区间
    保护为 ±1），序列色取 :data:`_CURVE_COLORS` 循环。
    """
    xs = [x for series in data.series for x in series.x]
    ys = [y for series in data.series for y in series.y]
    if not xs or not ys:
        raise TemplateError(f"曲线结果 {data.title!r} 无数据点")
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    if x_min == x_max:
        x_min, x_max = x_min - 1.0, x_max + 1.0
    if y_min == y_max:
        y_min, y_max = y_min - 1.0, y_max + 1.0
    plot_w = _SVG_W - _SVG_PAD_L - _SVG_PAD_R
    plot_h = _SVG_H - _SVG_PAD_T - _SVG_PAD_B
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{_SVG_W}" height="{_SVG_H}">']
    parts.append(f'<rect width="{_SVG_W}" height="{_SVG_H}" fill="#ffffff"/>')
    parts.append(
        f'<rect x="{_SVG_PAD_L}" y="{_SVG_PAD_T}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#cccccc"/>'
    )
    for index, series in enumerate(data.series):
        points = " ".join(
            f"{_SVG_PAD_L + (x - x_min) / (x_max - x_min) * plot_w:.1f},"
            f"{_SVG_PAD_T + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h:.1f}"
            for x, y in zip(series.x, series.y)
        )
        color = _CURVE_COLORS[index % len(_CURVE_COLORS)]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    parts += _svg_axes(data, (x_min, x_max, y_min, y_max), (plot_w, plot_h))
    parts += _svg_legend(data)
    parts.append("</svg>")
    return "".join(parts)


def _svg_axes(data: CurveData, bounds: tuple[float, float, float, float], plot: tuple[int, int]) -> list[str]:
    """轴标签与首尾刻度文本（轴名 + 数据范围）.

    :param bounds: ``(x_min, x_max, y_min, y_max)`` 数据范围。
    :param plot: ``(plot_w, plot_h)`` 绘图区尺寸。
    """
    x_min, x_max, y_min, y_max = bounds
    plot_w, plot_h = plot

    def text(x: float, y: float, content: str, anchor: str = "middle") -> str:
        return (
            f'<text x="{x:.0f}" y="{y:.0f}" font-size="12" fill="#666" text-anchor="{anchor}">{escape(content)}</text>'
        )

    parts: list[str] = []
    if data.x_label:
        parts.append(text(_SVG_PAD_L + plot_w / 2, _SVG_H - 6, data.x_label))
    if data.y_label:
        parts.append(
            f'<text x="14" y="{_SVG_PAD_T + plot_h / 2:.0f}" font-size="12" fill="#666" '
            f'transform="rotate(-90 14 {_SVG_PAD_T + plot_h / 2:.0f})" text-anchor="middle">{escape(data.y_label)}</text>'
        )
    parts.append(text(_SVG_PAD_L, _SVG_H - _SVG_PAD_B + 16, _fmt(x_min), "start"))
    parts.append(text(_SVG_PAD_L + plot_w, _SVG_H - _SVG_PAD_B + 16, _fmt(x_max), "end"))
    parts.append(text(_SVG_PAD_L - 6, _SVG_PAD_T + plot_h, _fmt(y_min), "end"))
    parts.append(text(_SVG_PAD_L - 6, _SVG_PAD_T + 10, _fmt(y_max), "end"))
    return parts


def _svg_legend(data: CurveData) -> list[str]:
    """图例（右上角色块 + 序列名）."""
    parts: list[str] = []
    for index, series in enumerate(data.series):
        color = _CURVE_COLORS[index % len(_CURVE_COLORS)]
        x = _SVG_W - _SVG_PAD_R - 140
        y = _SVG_PAD_T + 8 + index * 18
        parts.append(f'<rect x="{x}" y="{y}" width="12" height="12" fill="{color}"/>')
        parts.append(
            f'<text x="{x - 6}" y="{y + 11}" font-size="12" fill="#333" text-anchor="end">{escape(series.name)}</text>'
        )
    return parts
