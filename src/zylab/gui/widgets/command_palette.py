"""命令面板：VS Code 风格功能搜索 + 主题实时预览（Ctrl+Shift+P）.

- :class:`Command`：面板命令项（中文标题为主搜索键，英文 keywords 辅助）；
- :class:`CommandPalette`：无边框弹层（搜索框 + 结果列表），
  上下键导航、Enter 执行、Esc 关闭；键入 ``>`` 前缀或执行「选择主题」
  进入主题模式——上下键即时预览（VS Code 颜色主题语义），Enter 确认
  （由主窗口持久化），Esc 取消并还原原主题。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .. import theme
from ..qt_compat import (
    QDialog,
    QEvent,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPoint,
    Qt,
    QVBoxLayout,
    QWidget,
    Signal,
)

__all__ = ["Command", "CommandPalette"]

#: 列表行兜底高度（命令行项与主题行共用）：需容纳样式表字体行高 + 上下边距，
# 兜底值按 polish 后 13px 字体（行高约 18px）+ 12px 垂直边距取整
_ROW_H = 34


@dataclass(frozen=True)
class Command:
    """面板可执行命令项.

    Attributes:
        id: 唯一标识（重复注册同 id 覆盖定义、保持原顺序）。
        title: 中文标题（搜索主键）。
        callback: 无参执行回调。
        keywords: 英文辅助关键词（空格分隔）。
        shortcut: 展示用快捷键文本（空串不显示）。
    """

    id: str
    title: str
    callback: Callable[[], None]
    keywords: str = ""
    shortcut: str = ""


class CommandPalette(QDialog):
    """命令面板弹层：命令过滤执行 + 主题上下键实时预览.

    信号:
        theme_previewed: 主题模式下导航到某主题（主窗口应用预览，不持久化）。
        theme_confirmed: 主题模式下按 Enter 确认（主窗口应用并持久化）。
    """

    theme_previewed = Signal(str)
    theme_confirmed = Signal(str)

    _COMMAND_MODE = 0
    _THEME_MODE = 1

    def __init__(self, parent: QWidget) -> None:
        """构建面板 UI（搜索框 + 结果列表，附着于主窗口头部下方）."""
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint)
        self._commands: list[Command] = []
        self._by_id: dict[str, Command] = {}
        self._mode = self._COMMAND_MODE
        self._original_theme = theme.current_palette().name
        self._build_ui()

    # ------------------------------------------------------------------ 公共接口

    def register(self, command: Command) -> None:
        """注册命令（同 id 覆盖定义并保持首次注册位置）."""
        self._by_id[command.id] = command
        for index, existing in enumerate(self._commands):
            if existing.id == command.id:
                self._commands[index] = command
                return
        self._commands.append(command)

    def open_commands(self) -> None:
        """命令模式打开面板（清空搜索，全量命令列表）."""
        self._set_search_text("")
        self._populate_commands("")
        self._place_and_show()

    def open_theme_picker(self) -> None:
        """主题选择模式打开面板（上下键实时预览，Enter 确认，Esc 还原）."""
        self._set_search_text(">")
        self._populate_themes("")
        self._place_and_show()

    # ------------------------------------------------------------------ 内部

    def _build_ui(self) -> None:
        """组装搜索框与结果列表."""
        self.setObjectName("commandPalette")
        self.setFixedWidth(480)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        self._search = QLineEdit(objectName="paletteSearch")
        self._search.setPlaceholderText("输入命令关键词搜索；键入 > 选择主题")
        self._search.textChanged.connect(self._apply_filter)
        self._search.installEventFilter(self)
        root.addWidget(self._search)
        self._list = QListWidget(objectName="paletteList")
        self._list.itemActivated.connect(self._activate_current)
        self._list.currentRowChanged.connect(self._on_row_changed)
        root.addWidget(self._list)

    def _set_search_text(self, text: str) -> None:
        """设置搜索框文本（阻塞信号，模式切换由调用方显式填充列表）."""
        self._search.blockSignals(True)
        self._search.setText(text)
        self._search.blockSignals(False)

    def _place_and_show(self) -> None:
        """定位于父窗口头部条下方水平居中并显示（焦点入搜索框）."""
        parent = self.parentWidget()
        if parent is not None:
            x = (parent.width() - self.width()) // 2
            self.move(parent.mapToGlobal(QPoint(x, theme.HEADER_HEIGHT)))
        self._search.setFocus()
        self.show()

    def _apply_filter(self, text: str) -> None:
        """按搜索词分发：``>`` 前缀进主题模式，否则命令过滤."""
        if text.startswith(">"):
            self._populate_themes(text[1:].strip().lower())
        else:
            self._populate_commands(text.strip())

    def _populate_commands(self, query: str) -> None:
        """命令模式列表：全部搜索词命中（标题 + 英文关键词）才显示."""
        self._mode = self._COMMAND_MODE
        self._list.clear()
        terms = query.lower().split()
        for command in self._commands:
            haystack = f"{command.title} {command.keywords}".lower()
            if all(term in haystack for term in terms):
                self._add_command_row(command)
        self._select_first()

    def _add_command_row(self, command: Command) -> None:
        """命令行项：标题居左，快捷键提示居右（次级色）."""
        item = QListWidgetItem(self._list)
        item.setData(Qt.UserRole, ("command", command.id))
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)
        layout.addWidget(QLabel(command.title, objectName="paletteTitle"))
        layout.addStretch()
        if command.shortcut:
            layout.addWidget(QLabel(command.shortcut, objectName="paletteHint"))
        hint = row.sizeHint()
        hint.setHeight(max(hint.height(), _ROW_H))  # 兜底行高，防样式表字体度量偏小致行间遮挡
        item.setSizeHint(hint)
        self._list.addItem(item)
        self._list.setItemWidget(item, row)

    def _populate_themes(self, query: str) -> None:
        """主题模式列表：记录原主题供取消还原，默认选中当前主题（不触发预览）."""
        self._mode = self._THEME_MODE
        self._original_theme = theme.current_palette().name
        self._list.clear()
        current = theme.current_palette().name
        target = 0
        for name, pal in theme.THEMES.items():
            text = pal.display_name + ("（当前）" if name == current else "")
            if query and query not in text.lower() and query not in name:
                continue
            if name == current:
                target = self._list.count()
            item = QListWidgetItem(text, self._list)
            item.setData(Qt.UserRole, ("theme", name))
            hint = item.sizeHint()
            hint.setHeight(max(hint.height(), _ROW_H))  # 与命令行项统一的舒适行高
            item.setSizeHint(hint)
        # 静默选中当前主题：导航预览从当前态开始，无闪烁
        self._list.blockSignals(True)
        self._list.setCurrentRow(target)
        self._list.blockSignals(False)

    def _select_first(self) -> None:
        """默认选中首项（空列表时静默）."""
        if self._list.count() > 0:
            self._list.setCurrentRow(0)

    def _on_row_changed(self, row: int) -> None:
        """主题模式下导航即实时预览（命令模式无副作用）."""
        if self._mode != self._THEME_MODE or row < 0:
            return
        info = self._list.item(row).data(Qt.UserRole)
        if info is not None and info[0] == "theme":
            self.theme_previewed.emit(info[1])

    def _activate_current(self, item: QListWidgetItem) -> None:
        """执行当前项：命令关闭面板后执行；主题确认并交主窗口持久化."""
        info = item.data(Qt.UserRole)
        if info is None:
            return
        kind, value = info
        self.close()
        if kind == "command":
            self._by_id[value].callback()
        else:
            self.theme_confirmed.emit(value)

    def keyPressEvent(self, event) -> None:  # Qt 命名约定
        """Esc 关闭：主题模式取消预览并还原原主题."""
        if event.key() == Qt.Key_Escape:
            original = self._original_theme
            self.close()
            if self._mode == self._THEME_MODE and theme.current_palette().name != original:
                self.theme_previewed.emit(original)
            return
        super().keyPressEvent(event)

    def eventFilter(self, obj, event) -> bool:  # Qt 命名约定
        """搜索框键盘直达：上下键移动选择，Enter 执行当前项."""
        if obj is self._search and event.type() == QEvent.KeyPress:
            key = event.key()
            if key in (Qt.Key_Up, Qt.Key_Down):
                step = 1 if key == Qt.Key_Down else -1
                row = max(0, min(self._list.currentRow() + step, self._list.count() - 1))
                self._list.setCurrentRow(row)
                return True
            if key in (Qt.Key_Return, Qt.Key_Enter):
                item = self._list.currentItem()
                if item is not None:
                    self._activate_current(item)
                return True
        return super().eventFilter(obj, event)
