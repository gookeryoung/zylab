"""Python 语法高亮（QSyntaxHighlighter）：关键字/内置/字符串/注释/数字.

颜色全部取自 :mod:`zylab.gui.theme` 语义色板，换主题无需修改本模块；
三引号字符串跨行时用块状态机跟踪（0 = 块外，1 = 块内）。
"""

from __future__ import annotations

import builtins
import keyword
import re

from . import theme
from .qt_compat import QColor, QSyntaxHighlighter, QTextCharFormat

__all__ = ["PythonHighlighter"]

_KEYWORDS = frozenset(keyword.kwlist)
_BUILTINS = frozenset(dir(builtins))

_TRIPLE_RE = re.compile(r'("""|\'\'\')')
_LINE_COMMENT_RE = re.compile(r"#.*$")
_STRING_RE = re.compile(r"[rbuf]{0,2}(\"[^\"\n]*\"|'[^'\n]*)")
_NAME_RE = re.compile(r"\b[A-Za-z_]\w*\b")
#: 0x 十六进制须置于十进制之前，否则 \b\d+ 会先行截断前缀 0
_NUMBER_RE = re.compile(r"\b0[xX][0-9a-fA-F]+\b|\b\d+(?:\.\d*)?(?:[eE][+-]?\d+)?\b")

#: 块状态：三引号字符串内
_STATE_IN_TRIPLE = 1


class PythonHighlighter(QSyntaxHighlighter):
    """Python 源码高亮器（挂在编辑器的 document 上）."""

    def __init__(self, parent) -> None:  # parent: QTextDocument
        """按当前主题色板构建各类词法格式."""
        super().__init__(parent)
        pal = theme.current_palette()
        self._fmt_keyword = self._make_format(pal.primary, bold=True)
        self._fmt_builtin = self._make_format(pal.warning_text)
        self._fmt_string = self._make_format(pal.success_text)
        self._fmt_comment = self._make_format(pal.text_secondary, italic=True)
        self._fmt_number = self._make_format(pal.danger_text)

    @staticmethod
    def _make_format(color: str, *, bold: bool = False, italic: bool = False) -> QTextCharFormat:
        """构造带颜色/字重/斜体的字符格式."""
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        font = fmt.font()
        font.setBold(bold)
        font.setItalic(italic)
        fmt.setFont(font)
        return fmt

    def highlightBlock(self, text: str) -> None:  # Qt 命名约定
        """逐块高亮：三引号跨行状态机 → 单行 token（注释/字符串/数字/名字）."""
        state = self.previousBlockState()
        in_triple = state == _STATE_IN_TRIPLE
        pos = 0
        if in_triple:
            # 上块在三引号内：本块内首个三引号即为闭合符（其后文本由主循环处理）
            match = _TRIPLE_RE.search(text)
            if match is not None:
                pos = match.end()
                self.setFormat(0, pos, self._fmt_string)
            else:
                self.setFormat(0, len(text), self._fmt_string)
                self.setCurrentBlockState(_STATE_IN_TRIPLE)
                return

        for match in _TRIPLE_RE.finditer(text):
            if match.start() < pos:
                continue
            closer = text.find(match.group(1), match.end())
            if closer >= 0:
                end = closer + len(match.group(1))
                self.setFormat(match.start(), end - match.start(), self._fmt_string)
                pos = end
            else:
                self.setFormat(match.start(), len(text) - match.start(), self._fmt_string)
                self.setCurrentBlockState(_STATE_IN_TRIPLE)
                return
        self.setCurrentBlockState(0)
        self._highlight_tokens(text)

    def _highlight_tokens(self, text: str) -> None:
        """单行 token 高亮：注释/字符串/数字/关键字/内置名."""
        comment = _LINE_COMMENT_RE.search(text)
        if comment:
            self.setFormat(comment.start(), len(text) - comment.start(), self._fmt_comment)

        for match in _STRING_RE.finditer(text):
            self.setFormat(match.start(), match.end() - match.start(), self._fmt_string)

        for match in _NUMBER_RE.finditer(text):
            self.setFormat(match.start(), match.end() - match.start(), self._fmt_number)

        for match in _NAME_RE.finditer(text):
            fmt = None
            if match.group() in _KEYWORDS:
                fmt = self._fmt_keyword
            elif match.group() in _BUILTINS:
                fmt = self._fmt_builtin
            if fmt is not None:
                self.setFormat(match.start(), match.end() - match.start(), fmt)
