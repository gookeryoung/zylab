"""DSL 报告生成器：模板 + 节点输出 -> Markdown/HTML 报告（Qt-free）.

复用 :func:`zylab.studio.results.build_result` 把 ``results`` 声明解析为
标准化视图数据，再按报告载体拼装：

- 曲线（CurveData）→ 纯 Python 拼装的 SVG（零绘图依赖），Markdown 以
  base64 data URI 内嵌，HTML 直接内联；
- 云图（CloudData）→ 纯 Python 拼装的 SVG（节点标量场着色：2D 单元
  填充 + 线框，3D 等轴测投影散点 + 线框，附色标），渲染失败回落
  文字占位（报告总能生成）；
- 表格（TableData）→ Markdown 管道表 / HTML ``<table>``；
- 文本（TextData）→ 段落。

报告结构（``report`` 声明）：标题 + 参数概览表 + 章节序列（标题/正文/
图表按 results id 引用，figure 可引用曲线或云图）；未声明 report 时自动
收录全部 results 兜底。
"""

from __future__ import annotations

import base64
from html import escape
from typing import Any, Mapping

import numpy as np

from zylab.fea.viewdata import cmap_lut, deformed_coords, mesh_edges, nodal_stress_field, project3d, scalar_colors

from .dsl import DslReportSection, DslTemplate
from .errors import TemplateError
from .results import CloudData, CurveData, TableData, TextData, ViewData, build_result

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
    if section.figure and not isinstance(view, (CurveData, CloudData)):
        raise TemplateError(f"报告章节 {section.title!r} 的 figure 引用 {ref!r} 应为曲线或云图结果")
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
    """Markdown 视图块（曲线/云图 data URI / 管道表 / 段落）."""
    if isinstance(view, CurveData):
        uri = _svg_data_uri(_curve_svg(view))
        return [f"![{view.title}]({uri})", ""]
    if isinstance(view, TableData):
        lines = [f"**{view.title}**", "", "| " + " | ".join(view.columns) + " |", "|" + " --- |" * len(view.columns)]
        lines += ["| " + " | ".join(_fmt(cell) for cell in row) + " |" for row in view.rows]
        return [*lines, ""]
    if isinstance(view, TextData):
        return [view.text, ""]
    return [*_cloud_md(view), ""]


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
    svg, fallback = _cloud_svg(view)
    if svg is None:
        return f"<p>{escape(fallback)}</p>"
    return f'<img alt="{escape(view.title)}" src="{_svg_data_uri(svg)}">'


def _cloud_md(view: CloudData) -> list[str]:
    """Markdown 云图块：SVG 成功内嵌 data URI，失败回落文字占位."""
    svg, fallback = _cloud_svg(view)
    if svg is None:
        return [fallback]
    return [f"![{view.title}]({_svg_data_uri(svg)})"]


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


# ------------------------------------------------------------------ 云图 SVG

#: 云图场量显示名（field 声明 -> 标注文本）
_CLOUD_FIELD_LABELS = {"temperature": "温度", "voltage": "电压", "displacement": "位移模", "stress": "应力"}

_CLOUD_W = 640  # 云图 SVG 画布宽（像素）
_CLOUD_H = 420  # 云图 SVG 画布高（像素）
_CLOUD_BAR_W = 20  # 色标条宽
_CLOUD_BAR_H = 260  # 色标条高
_CLOUD_R = 3.0  # 3D 节点散点半径


def _cloud_svg(view: CloudData) -> tuple[str | None, str]:
    """云图 SVG（节点标量场着色 + 色标）；载荷不可渲染时返回占位文字.

    :param view: 云图视图数据（payload 为解对象）。
    :returns: ``(svg 文本, 占位文字)``——渲染成功占位为空描述，失败 svg 为 None。
    """
    try:
        return _build_cloud_svg(view), ""
    except (AttributeError, IndexError, TypeError, ValueError, TemplateError):
        return None, f"（云图结果 {view.node_id!r} 载荷暂不支持报告渲染，由应用界面查看）"


def _build_cloud_svg(view: CloudData) -> str:
    """拼装云图 SVG（2D 单元填充 / 3D 等轴测投影散点 + 线框 + 色标）."""
    payload = view.payload
    if payload is None or isinstance(payload, Mapping):
        raise TypeError("云图载荷缺失或非解对象")
    mesh = payload.mesh
    values, label = _cloud_field(payload, view.field, mesh.n_nodes)
    coords = _cloud_coords(payload, mesh, values, view.deform)
    colors = scalar_colors(values, cmap=view.cmap)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_CLOUD_W}" height="{_CLOUD_H}">',
        f'<rect width="{_CLOUD_W}" height="{_CLOUD_H}" fill="#ffffff"/>',
        f'<text x="{_SVG_PAD_L}" y="24" font-size="14" fill="#333">{escape(view.title)} · {escape(label)}</text>',
    ]
    if mesh.dim >= 3:
        parts += _cloud_3d(mesh, coords, colors)
    else:
        parts += _cloud_2d(mesh, coords, colors)
    parts.append(_cloud_colorbar(values, label, view.cmap))
    parts.append("</svg>")
    return "".join(parts)


def _cloud_field(payload: Any, field: str, n_nodes: int) -> tuple[np.ndarray, str]:
    """提取节点标量场（field 显式声明优先，缺省按载荷属性自适应）.

    :returns: ``(n_nodes,) 标量数组, 场量显示名``。
    """
    declared = field or _auto_field(payload)
    if declared == "temperature":
        return _last_frame(np.asarray(payload.temperatures), n_nodes), _CLOUD_FIELD_LABELS[declared]
    if declared == "voltage":
        return np.asarray(payload.voltages), _CLOUD_FIELD_LABELS[declared]
    if declared == "displacement":
        disp = np.asarray(payload.displacements, dtype=float)
        if disp.ndim == 2 and disp.shape[0] != n_nodes:  # (n_dofs, n_times) 全时程：取末帧
            disp = disp[:, -1].reshape(n_nodes, -1)
        dim = payload.mesh.dim
        return np.linalg.norm(disp[:, :dim], axis=1), _CLOUD_FIELD_LABELS[declared]
    if declared == "stress":
        return nodal_stress_field(payload), _CLOUD_FIELD_LABELS[declared]
    raise TemplateError(f"云图场量 {declared!r} 不受支持（可选 temperature/voltage/displacement/stress）")


def _auto_field(payload: Any) -> str:
    """按载荷属性自适应场量（温度优先，其次位移/电压/应力）."""
    for name, attribute in (
        ("temperature", "temperatures"),
        ("displacement", "displacements"),
        ("voltage", "voltages"),
        ("stress", "element_results"),
    ):
        if getattr(payload, attribute, None) is not None:
            return name
    raise TypeError("载荷不含可渲染的节点标量场")


def _last_frame(values: np.ndarray, n_nodes: int) -> np.ndarray:
    """瞬态场 ``(n_frames, n_nodes)`` 取末帧，节点场原样返回."""
    if values.ndim == 2 and values.shape[0] != n_nodes:
        return values[-1]
    return values


def _cloud_coords(payload: Any, mesh: Any, values: np.ndarray, deform: float) -> np.ndarray:
    """云图绘制坐标：位移场叠加变形（放大 ``deform`` 倍），其余原坐标."""
    if values.shape == (mesh.n_nodes,) and getattr(payload, "displacements", None) is not None:
        disp = np.asarray(payload.displacements, dtype=float)
        if disp.ndim == 2 and disp.shape[0] != mesh.n_nodes:
            disp = disp[:, -1].reshape(mesh.n_nodes, -1)
        return deformed_coords(mesh, disp, scale=deform)
    return np.asarray(mesh.coords, dtype=float)


def _cloud_frame(coords: np.ndarray) -> tuple[np.ndarray, float]:
    """坐标归一化到绘图区（保持纵横比，中心对齐），返回屏幕坐标与缩放系数."""
    plot_w = _CLOUD_W - _SVG_PAD_L - _SVG_PAD_R - 96  # 右侧留出色标区
    plot_h = _CLOUD_H - _SVG_PAD_T - _SVG_PAD_B - 16
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    span_x = x_max - x_min if x_max > x_min else 1.0
    span_y = y_max - y_min if y_max > y_min else 1.0
    scale = min(plot_w / span_x, plot_h / span_y)
    offset_x = _SVG_PAD_L + (plot_w - span_x * scale) / 2
    offset_y = _SVG_PAD_T + 16 + (plot_h - span_y * scale) / 2
    screen = np.column_stack(
        (
            offset_x + (coords[:, 0] - x_min) * scale,
            offset_y + (1.0 - (coords[:, 1] - y_min) / span_y) * span_y * scale,
        )
    )
    return screen, scale


def _cloud_2d(mesh: Any, coords: np.ndarray, colors: np.ndarray) -> list[str]:
    """2D 云图：闭合单元多边形填充（节点平均色）+ 全网格线框."""
    screen, _ = _cloud_frame(coords)
    parts: list[str] = []
    for block in mesh.blocks:
        conn = np.asarray(block.conn)
        is_line = block.etype.value in ("truss2", "beam2") or conn.shape[1] <= 2
        for row in conn:
            points = " ".join(f"{screen[n, 0]:.1f},{screen[n, 1]:.1f}" for n in row)
            fill = _rgb(colors[list(row)].mean(axis=0))
            if is_line:  # 线单元：粗线段按场着色
                parts.append(f'<polyline points="{points}" fill="none" stroke="{fill}" stroke-width="4"/>')
            else:
                parts.append(f'<polygon points="{points}" fill="{fill}" stroke="none"/>')
    edges = mesh_edges(mesh)
    for a, b in edges:
        parts.append(
            f'<line x1="{screen[a, 0]:.1f}" y1="{screen[a, 1]:.1f}" '
            f'x2="{screen[b, 0]:.1f}" y2="{screen[b, 1]:.1f}" stroke="#00000033" stroke-width="0.6"/>'
        )
    return parts


def _cloud_3d(mesh: Any, coords: np.ndarray, colors: np.ndarray) -> list[str]:
    """3D 云图：等轴测投影，边线框 + 按观察深度排序的节点散点着色."""
    xy, depth = project3d(coords)
    screen, _ = _cloud_frame(xy)
    parts: list[str] = []
    edges = mesh_edges(mesh)
    for a, b in edges:
        parts.append(
            f'<line x1="{screen[a, 0]:.1f}" y1="{screen[a, 1]:.1f}" '
            f'x2="{screen[b, 0]:.1f}" y2="{screen[b, 1]:.1f}" stroke="#00000022" stroke-width="0.6"/>'
        )
    for index in np.argsort(depth)[::-1]:  # 远者先画，近者覆盖
        parts.append(
            f'<circle cx="{screen[index, 0]:.1f}" cy="{screen[index, 1]:.1f}" '
            f'r="{_CLOUD_R:.1f}" fill="{_rgb(colors[index])}"/>'
        )
    return parts


def _cloud_colorbar(values: np.ndarray, label: str, cmap: str) -> str:
    """右侧色标（渐变条 + 场名 + 最值标注）."""
    bar_x = _CLOUD_W - _SVG_PAD_R - _CLOUD_BAR_W - 52
    bar_y = (_CLOUD_H - _CLOUD_BAR_H) / 2
    parts = [
        f'<text x="{bar_x + _CLOUD_BAR_W / 2:.0f}" y="{bar_y - 10:.0f}" font-size="12" fill="#333" '
        f'text-anchor="middle">{escape(label)}</text>'
    ]
    segments = 16
    lut = cmap_lut(cmap, samples=segments)
    for index in range(segments):
        y = bar_y + _CLOUD_BAR_H * (1.0 - (index + 1) / segments)
        parts.append(
            f'<rect x="{bar_x:.0f}" y="{y:.0f}" width="{_CLOUD_BAR_W}" '
            f'height="{_CLOUD_BAR_H / segments + 0.5:.1f}" fill="{_rgb(lut[index])}"/>'
        )
    parts.append(
        f'<rect x="{bar_x:.0f}" y="{bar_y:.0f}" width="{_CLOUD_BAR_W}" height="{_CLOUD_BAR_H}" fill="none" stroke="#999"/>'
    )
    parts.append(
        f'<text x="{bar_x + _CLOUD_BAR_W + 6:.0f}" y="{bar_y + 12:.0f}" font-size="11" fill="#333" '
        f'text-anchor="start">{_fmt(float(values.max()))}</text>'
    )
    parts.append(
        f'<text x="{bar_x + _CLOUD_BAR_W + 6:.0f}" y="{bar_y + _CLOUD_BAR_H:.0f}" font-size="11" fill="#333" '
        f'text-anchor="start">{_fmt(float(values.min()))}</text>'
    )
    return "".join(parts)


def _rgb(color: np.ndarray) -> str:
    """浮点 RGB 转 ``#rrggbb`` 色值文本."""
    return "#{:02x}{:02x}{:02x}".format(*(round(max(0.0, min(1.0, c)) * 255) for c in color))
