"""gui.highlight Python 语法高亮测试."""

from __future__ import annotations

import pytest

from zylab.gui import theme
from zylab.gui.highlight import PythonHighlighter
from zylab.gui.pages.notebook_page import CellWidget
from zylab.gui.qt_compat import QPlainTextEdit
from zylab.sci import new_cell


def _color_at(block, offset: int) -> str:
    """取块内指定偏移的高亮前景色（统一小写便于与主题令牌比较）."""
    for fmt_range in block.layout().formats():
        if fmt_range.start <= offset < fmt_range.start + fmt_range.length:
            color = fmt_range.format.foreground().color()
            return color.name().lower()
    return "#000000"


@pytest.mark.gui
def test_highlight_colors_by_token_kind(qtbot) -> None:
    """关键字/内置/字符串/注释/数字应各自着色（主题语义色）."""
    pal = theme.current_palette()
    widget = CellWidget(new_cell("if 1:  # note\nprint('s')"))
    qtbot.addWidget(widget)
    widget.highlighter.rehighlight()
    doc = widget.editor.document()

    first = doc.findBlock(0)
    text1 = first.text()
    assert _color_at(first, text1.index("if")) == pal.primary.lower()
    assert _color_at(first, text1.index("1")) == pal.danger_text.lower()
    assert _color_at(first, text1.index("#")) == pal.text_secondary.lower()

    second = doc.findBlockByNumber(1)
    text2 = second.text()
    assert _color_at(second, text2.index("print")) == pal.warning_text.lower()
    assert _color_at(second, text2.index("s")) == pal.success_text.lower()


@pytest.mark.gui
def test_highlighter_attached_to_editor(qtbot) -> None:
    """CellWidget 编辑器应挂载 PythonHighlighter."""
    widget = CellWidget(new_cell("a = 1"))
    qtbot.addWidget(widget)
    assert isinstance(widget.highlighter, PythonHighlighter)


@pytest.mark.gui
def test_triple_quote_multiline_string(qtbot) -> None:
    """三引号字符串跨块着色：未闭合行整行字符串色，闭合后恢复."""
    pal = theme.current_palette()
    editor = QPlainTextEdit()
    qtbot.addWidget(editor)
    highlighter = PythonHighlighter(editor.document())
    editor.setPlainText('s = """abc\ndef\nghi"""\nx = 1')
    highlighter.rehighlight()

    doc = editor.document()
    blocks = [doc.findBlockByNumber(i) for i in range(4)]
    assert [b.text() for b in blocks] == ['s = """abc', "def", 'ghi"""', "x = 1"]

    # 第 0 块 abc 起为字符串色；第 1 块整块在三引号内；第 2 块到闭合引号均字符串色
    assert _color_at(blocks[0], blocks[0].text().index("abc")) == pal.success_text.lower()
    assert _color_at(blocks[1], 1) == pal.success_text.lower()
    assert _color_at(blocks[2], 1) == pal.success_text.lower()
    # 第 3 块恢复：数字 1 数字色（块状态已复位）
    assert _color_at(blocks[3], blocks[3].text().index("1")) == pal.danger_text.lower()


@pytest.mark.gui
def test_plain_names_not_colored(qtbot) -> None:
    """普通用户名字不着色（保持默认前景色）."""
    widget = CellWidget(new_cell("my_var = 2"))
    qtbot.addWidget(widget)
    widget.highlighter.rehighlight()
    first = widget.editor.document().findBlock(0)
    assert _color_at(first, first.text().index("my_var")) == "#000000"
