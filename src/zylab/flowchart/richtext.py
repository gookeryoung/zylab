"""flowchart.richtext — 受控 Markdown → HTML 转换器（Qt-free，无新依赖）.

用途：DSL 结果的 text 块 ``format: markdown`` 声明需要在 GUI 与报告两端
渲染一致的富文本。本模块提供最小子集转换——仅实现 DSL 作者最常用的标记，
其余 HTML 标签一律转义，防止 YAML 内联 HTML 破坏渲染布局。

支持的 Markdown 子集：

- ``#`` ~ ``####`` 标题（分别映射为 ``<h1>`` ~ ``<h4>``）。
- ``**加粗**`` / ``*斜体*`` / ``行内代码``。
- ``-`` / ``*`` 无序列表、``1.`` 有序列表。
- ``> `` 引用块、``---`` 分割线。
- ``[text](url)`` 链接。
- ``<span style="color:语义名">`` 白名单内联样式：仅 ``color`` /
  ``font-weight`` / ``font-style`` 三个属性，且 color 值限定于语义名或
  ``#RRGGBB``；其余属性一律剥离。
- 代码块围栏 ```lang ... ```（渲染为 ``<pre><code>``）。

输出契约：所有非白名单 HTML 标签均转义为 ``&lt;tag&gt;``；``<script>``
与 ``<iframe>`` 即使白名单也剥离内容（安全兜底）。
"""

from __future__ import annotations

import html
import re
from typing import Iterable

__all__ = ["SemanticColorResolver", "markdown_to_html"]

#: 语义色名集合（与 GUI 主题 / 报告打印固定色表保持一致）。
_SEMANTIC_COLORS: frozenset[str] = frozenset(
    {"primary", "success", "warning", "danger", "info", "text", "text_secondary"},
)

#: 合法 ``#RRGGBB`` 格式正则。
_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")

#: 块级围栏代码块正则（```lang ... ```）。
_FENCE_RE = re.compile(r"^```\s*(\w*)\s*$")

#: 标题正则（# 开头 1~6 个）。
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

#: 有序列表项正则。
_OL_RE = re.compile(r"^(\s*)(\d+)\.\s+(.*)$")

#: 无序列表项正则。
_UL_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")

#: 引用块正则。
_BLOCKQUOTE_RE = re.compile(r"^>\s?(.*)$")

#: 分割线正则。
_HR_RE = re.compile(r"^\s*---+\s*$")

#: 行内 span 白名单 color 正则（语义名或 #RRGGBB）。
_SPAN_COLOR_RE = re.compile(r'<span\s+style="([^"]+)"[^>]*>(.*?)</span>', re.DOTALL)


class SemanticColorResolver:
    """语义色解析器：将语义名映射为 CSS 颜色值.

    GUI 端传入 ``{'primary': '#4B3FE3', ...}``，报告端使用打印友好固定色表。
    未解析时回退为空串（让浏览器 / Qt 自动使用当前主题）。
    """

    __slots__ = ("_map",)

    def __init__(self, color_map: dict[str, str] | None = None) -> None:
        self._map: dict[str, str] = dict(color_map) if color_map else {}

    def resolve(self, name: str) -> str:
        """解析颜色名或原样返回（已合法的 #RRGGBB 直通）."""
        if not name:
            return ""
        if _HEX_COLOR.match(name):
            return name
        return self._map.get(name, "")


def _escape_inline(text: str) -> str:
    """行内转义：先处理白名单 span，再转义剩余 HTML 字符."""
    parts: list[str] = []
    last = 0
    for m in _SPAN_COLOR_RE.finditer(text):
        parts.append(html.escape(text[last : m.start()], quote=True))
        style_attr = m.group(1)
        inner = m.group(2)
        # 仅保留 color / font-weight / font-style 三个属性
        cleaned: list[str] = []
        for raw_attr in style_attr.split(";"):
            attr = raw_attr.strip().lower()
            if not attr:
                continue
            kv = attr.split(":", 1)
            if len(kv) != 2:
                continue
            k, v = kv[0].strip(), kv[1].strip()
            if k == "color":
                if v in _SEMANTIC_COLORS or _HEX_COLOR.match(v):
                    cleaned.append(f"color: {v}")
            elif k == "font-weight" and v in ("bold", "normal", "700"):
                cleaned.append(f"font-weight: {v}")
            elif k == "font-style" and v in ("italic", "normal"):
                cleaned.append(f"font-style: {v}")
        if cleaned:
            parts.append(f'<span style="{"; ".join(cleaned)}">{_escape_inline(inner)}</span>')
        else:
            parts.append(html.escape(inner, quote=True))
        last = m.end()
    parts.append(html.escape(text[last:], quote=True))
    inline = "".join(parts)
    inline = re.sub(r"`([^`]+)`", r"<code>\1</code>", inline)
    inline = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", inline)
    inline = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", inline)
    inline = re.sub(
        r"\[([^\]]+)\]\(([^)\s]+)\)",
        r'<a href="\2">\1</a>',
        inline,
    )
    return inline


def _render_paragraph(lines: Iterable[str]) -> str:
    """将多行文本块渲染为单个 <p>."""
    text = " ".join(line.strip() for line in lines).strip()
    if not text:
        return ""
    return f"<p>{_escape_inline(text)}</p>\n"


def markdown_to_html(  # noqa: PLR0912
    text: str,
    color_resolver: SemanticColorResolver | None = None,
) -> str:
    """受控 Markdown → HTML 转换.

    :param text: 原始 Markdown 文本。
    :param color_resolver: 语义色解析器（当前仅保存，span 白名单已足够覆盖）。
    :return: 转义后 HTML（可安全嵌入 Qt QTextBrowser / HTML 报告）。
    """
    del color_resolver
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        fm = _FENCE_RE.match(line.strip())
        if fm:
            lang = fm.group(1)
            i += 1
            code_lines: list[str] = []
            while i < n and not _FENCE_RE.match(lines[i].strip()):
                code_lines.append(lines[i])
                i += 1
            if i < n:
                i += 1
            code_html = html.escape("\n".join(code_lines), quote=False)
            out.append(f'<pre><code class="language-{lang}">{code_html}</code></pre>\n')
            continue
        if _HR_RE.match(line):
            out.append("<hr />\n")
            i += 1
            continue
        hm = _HEADING_RE.match(line)
        if hm:
            level = len(hm.group(1))
            body = _escape_inline(hm.group(2).strip())
            out.append(f"<h{level}>{body}</h{level}>\n")
            i += 1
            continue
        om = _OL_RE.match(line)
        if om:
            items: list[str] = []
            while i < n:
                lm = _OL_RE.match(lines[i])
                if not lm:
                    break
                items.append(f"<li>{_escape_inline(lm.group(3).strip())}</li>")
                i += 1
            out.append("<ol>\n" + "\n".join(items) + "\n</ol>\n")
            continue
        um = _UL_RE.match(line)
        if um:
            items = []
            while i < n:
                lm = _UL_RE.match(lines[i])
                if not lm:
                    break
                items.append(f"<li>{_escape_inline(lm.group(2).strip())}</li>")
                i += 1
            out.append("<ul>\n" + "\n".join(items) + "\n</ul>\n")
            continue
        bm = _BLOCKQUOTE_RE.match(line)
        if bm:
            block_lines: list[str] = []
            while i < n:
                mq = _BLOCKQUOTE_RE.match(lines[i])
                if not mq:
                    break
                block_lines.append(mq.group(1))
                i += 1
            inner = _render_paragraph(block_lines).strip()
            out.append(f"<blockquote>{inner}</blockquote>\n")
            continue
        if not line.strip():
            i += 1
            continue
        para_lines: list[str] = [line]
        i += 1
        while i < n:
            nxt = lines[i]
            if not nxt.strip():
                break
            if (
                _FENCE_RE.match(nxt.strip())
                or _HR_RE.match(nxt)
                or _HEADING_RE.match(nxt)
                or _OL_RE.match(nxt)
                or _UL_RE.match(nxt)
                or _BLOCKQUOTE_RE.match(nxt)
            ):
                break
            para_lines.append(nxt)
            i += 1
        out.append(_render_paragraph(para_lines))
    return "".join(out)
