"""flowchart.richtext Markdown → HTML 转换测试."""

from __future__ import annotations

from zylab.flowchart.richtext import (
    SemanticColorResolver,
    markdown_to_html,
)


def test_bold_and_italic() -> None:
    """**加粗** / *斜体* / 行内代码 基础转换."""
    html_out = markdown_to_html("**粗** *斜* `code`")
    assert "<strong>粗</strong>" in html_out
    assert "<em>斜</em>" in html_out
    assert "<code>code</code>" in html_out


def test_heading_levels() -> None:
    """#~###### 标题正确映射到 h1~h4（仅支持前四级）."""
    html_out = markdown_to_html("# H1\n## H2\n### H3\n#### H4")
    assert "<h1>H1</h1>" in html_out
    assert "<h2>H2</h2>" in html_out
    assert "<h3>H3</h3>" in html_out
    assert "<h4>H4</h4>" in html_out


def test_unordered_list() -> None:
    """- 无序列表."""
    html_out = markdown_to_html("- a\n- b\n- c")
    assert "<ul>" in html_out
    assert "<li>a</li>" in html_out
    assert "<li>b</li>" in html_out
    assert "<li>c</li>" in html_out


def test_ordered_list() -> None:
    """1. 有序列表."""
    html_out = markdown_to_html("1. 一\n2. 二")
    assert "<ol>" in html_out
    assert "<li>一</li>" in html_out
    assert "<li>二</li>" in html_out


def test_blockquote() -> None:
    """> 引用块."""
    html_out = markdown_to_html("> 引用文本")
    assert "<blockquote>" in html_out
    assert "引用文本" in html_out


def test_hr() -> None:
    """--- 分割线."""
    html_out = markdown_to_html("---")
    assert "<hr />" in html_out


def test_code_fence() -> None:
    """```lang 围栏代码块."""
    html_out = markdown_to_html("```python\nprint('x')\n```")
    assert '<pre><code class="language-python">' in html_out
    assert "print" in html_out


def test_link() -> None:
    """[text](url) 链接."""
    html_out = markdown_to_html("[zylab](https://example.com)")
    assert '<a href="https://example.com">zylab</a>' in html_out


def test_span_color_whitelist() -> None:
    """白名单 span 保留、非法属性剥离."""
    html_out = markdown_to_html('<span style="color: danger">告警</span>')
    assert 'style="color: danger"' in html_out
    assert "告警" in html_out


def test_span_illegal_attr_stripped() -> None:
    """非法 style 属性剥离后内容保留."""
    html_out = markdown_to_html('<span style="font-size: 200px">bad</span>')
    assert "bad" in html_out
    assert "font-size" not in html_out


def test_html_escape() -> None:
    """非白名单 HTML 标签转义."""
    html_out = markdown_to_html("<div>evil</div>")
    assert html_out.count("&lt;") >= 1


def test_paragraph_join() -> None:
    """连续非空行合并为单个 <p>."""
    html_out = markdown_to_html("a\nb")
    assert html_out.count("<p>") == 1
    assert "a b" in html_out


def test_semantic_color_resolver() -> None:
    """SemanticColorResolver：语义名查映射、#RRGGBB 直通."""
    r = SemanticColorResolver({"primary": "#4B3FE3"})
    assert r.resolve("primary") == "#4B3FE3"
    assert r.resolve("#FF0000") == "#FF0000"
    assert r.resolve("unknown") == ""
    assert r.resolve("") == ""


# ---------------------------------------------------------------- 补充分支覆盖


def test_span_font_weight_and_style_preserved() -> None:
    """font-weight / font-style 白名单属性保留."""
    html_out = markdown_to_html('<span style="font-weight: bold;font-style: italic">hi</span>')
    assert "font-weight: bold" in html_out
    assert "font-style: italic" in html_out
    assert ">hi</span>" in html_out


def test_span_hex_color_preserved() -> None:
    """#RRGGBB hex color 保留."""
    html_out = markdown_to_html('<span style="color: #FF00FF">magenta</span>')
    assert "color: #ff00ff" in html_out


def test_span_style_semicolon_gap_and_bad_kv() -> None:
    """style 中连续分号空段 / 非 key:value 字段被跳过."""
    html_out = markdown_to_html('<span style="color: primary;;bad;font-weight: bold">x</span>')
    assert "color: primary" in html_out
    assert "font-weight: bold" in html_out
    assert "bad" not in html_out


def test_empty_paragraph_dropped() -> None:
    """空文本段落返回空字符串."""
    assert markdown_to_html("") == ""
    assert markdown_to_html("\n\n") == ""


def test_list_breaks_on_non_list_line() -> None:
    """有序列表遇到非列表行时 break."""
    content = "\n".join(["1. a", "2. b", "不是列表", "3. c"])
    html_out = markdown_to_html(content)
    assert "<ol>" in html_out
    assert "<li>a</li>" in html_out
    assert "<li>b</li>" in html_out
    assert "不是列表" in html_out


def test_code_fence_unclosed_consumed() -> None:
    """围栏代码块无闭合时消费到文本末尾."""
    html_out = markdown_to_html("```python\nprint('x')")
    assert '<pre><code class="language-python">' in html_out
    assert "print" in html_out


def test_span_color_not_whitelisted_stripped() -> None:
    """color 属性值不在语义名/hex 白名单时被剥离."""
    html_out = markdown_to_html('<span style="color: illegal;color: primary">x</span>')
    # illegal 被剥离，primary 保留
    assert "color: primary" in html_out
    assert "illegal" not in html_out


def test_unordered_list_breaks_on_non_list() -> None:
    """无序列表遇到非列表行 break."""
    content = "\n".join(["- a", "- b", "正文", "- c"])
    html_out = markdown_to_html(content)
    assert "<ul>" in html_out
    assert "<li>a</li>" in html_out
    assert "正文" in html_out


def test_blockquote_continuation() -> None:
    """多行引用块."""
    content = "\n".join(["> 第一行", "> 第二行"])
    html_out = markdown_to_html(content)
    assert "<blockquote>" in html_out
    assert "第一行" in html_out
    assert "第二行" in html_out


def test_paragraph_breaks_on_blank_line() -> None:
    """连续段落被空行分隔."""
    content = "\n".join(["第一段", "", "第二段"])
    html_out = markdown_to_html(content)
    assert html_out.count("<p>") >= 2


def test_blockquote_breaks_on_non_quote_line() -> None:
    """多行引用块遇到非 > 开头行时 break 并单独成段落."""
    content = "\n".join(["> 引用", "正文"])
    html_out = markdown_to_html(content)
    assert "<blockquote>" in html_out
    # "正文" 不应在 blockquote 里
    assert "正文" in html_out


def test_blockquote_empty_content_renders_nothing() -> None:
    """引用块 > 后无文本时 inner 为空，_render_paragraph 返回空字符串."""
    html_out = markdown_to_html("> ")
    assert "<blockquote>" in html_out
