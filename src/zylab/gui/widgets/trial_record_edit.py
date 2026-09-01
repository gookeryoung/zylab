"""升降法逐发试验记录输入控件（DSL 定制参数控件）.

:class:`TrialRecordEdit` 为「刺激量 + 响应/不响应」逐发记录提供便捷
录入：按升降规则（响应降一步、不响应升一步）自动推算下一发刺激量，
表格可逐发修正实测水平与响应（真实记录偏离规则时），并序列化为
``"3.20 O, 3.15 X, ..."`` 文本——``text``/``setText``/``textChanged``
与 QLineEdit 接口兼容，DSL 参数表单零适配接入（参数声明
``widget: response_sequence``），后端由
:func:`zylab.reliability.parse_trial_records` 解析。
"""

from __future__ import annotations

from zylab.reliability import ReliabilityError, parse_trial_records

from .. import theme
from ..qt_compat import (
    QAbstractItemView,
    QDoubleSpinBox,
    QHBoxLayout,
    QPushButton,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    Signal,
)

__all__ = ["TrialRecordEdit"]

#: 记录表最大可视行数（超出滚动，避免长序列撑爆表单）
_TABLE_MAX_HEIGHT = 220


class TrialRecordEdit(QWidget):
    """逐发试验记录输入（QLineEdit 兼容接口）.

    信号:
        textChanged: 记录序列化文本变化（与 QLineEdit 同名信号语义一致）。
    """

    textChanged = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化：建议参数行（初始刺激量/步长）+ 操作按钮 + 记录表格."""
        super().__init__(parent)
        self._records: list[tuple[float, int]] = []
        self._updating = False
        self._build_ui()

    # ------------------------------------------------------------------ 对外

    def text(self) -> str:
        """序列化记录文本（``"3.20 O, 3.15 X"``，逗号分隔逐发）."""
        return ", ".join(f"{level:.6g} {'O' if hit else 'X'}" for level, hit in self._records)

    def setText(self, text: str) -> None:
        """解析记录文本并重建表格（格式非法时清空）."""
        try:
            levels, responses = parse_trial_records(text)
        except ReliabilityError:
            levels, responses = [], []
        self._records = [(float(level), int(hit)) for level, hit in zip(levels, responses)]
        self._rebuild_table()
        self.textChanged.emit(self.text())

    def set_suggest_step(self, step: float) -> None:
        """外部同步建议步长（表单 step 参数变化时联动建议网格）."""
        if step > 0.0:
            self._step_spin.setValue(float(step))

    # ------------------------------------------------------------------ 内部

    def _build_ui(self) -> None:
        """构建界面：建议参数 + 追加/撤销/清空按钮 + 记录表格."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACING_SM)
        controls = QHBoxLayout()
        controls.setSpacing(theme.SPACING_SM)
        self._start_spin = self._make_spin(3.2, "初始刺激量")
        self._step_spin = self._make_spin(0.05, "步长 d")
        controls.addWidget(self._start_spin)
        controls.addWidget(self._step_spin)
        add_hit = QPushButton("＋响应 O")
        add_hit.setToolTip("追加一发响应记录，按升降规则自动降一个步长")
        add_hit.clicked.connect(lambda: self._append(1))
        add_miss = QPushButton("＋不响应 X")
        add_miss.setToolTip("追加一发不响应记录，按升降规则自动升一个步长")
        add_miss.clicked.connect(lambda: self._append(0))
        undo = QPushButton("撤销")
        undo.setToolTip("删除最后一发记录")
        undo.clicked.connect(self._remove_last)
        clear = QPushButton("清空")
        clear.setToolTip("清空全部记录")
        clear.clicked.connect(self._clear)
        self._add_hit_btn, self._add_miss_btn, self._undo_btn, self._clear_btn = add_hit, add_miss, undo, clear
        for button in (add_hit, add_miss, undo, clear):
            controls.addWidget(button)
        controls.addStretch()
        layout.addLayout(controls)
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(("发序", "刺激量", "响应"))
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setMaximumHeight(_TABLE_MAX_HEIGHT)
        self._table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._table)

    def _make_spin(self, value: float, tooltip: str) -> QDoubleSpinBox:
        """构建建议参数输入框（初始刺激量/步长共用）."""
        spin = QDoubleSpinBox()
        spin.setDecimals(4)
        spin.setRange(-1.0e6, 1.0e6)
        spin.setSingleStep(0.05)
        spin.setValue(value)
        spin.setToolTip(tooltip)
        spin.setMinimumWidth(88)
        return spin

    def _append(self, hit: int) -> None:
        """按升降规则推算下一发刺激量并追加记录（响应降、不响应升）."""
        if self._records:
            last_level, last_hit = self._records[-1]
            level = last_level - self._step_spin.value() if last_hit else last_level + self._step_spin.value()
        else:
            level = self._start_spin.value()
        self._records.append((round(float(level), 6), hit))
        self._rebuild_table()
        self.textChanged.emit(self.text())

    def _remove_last(self) -> None:
        """删除最后一发记录."""
        if not self._records:
            return
        self._records.pop()
        self._rebuild_table()
        self.textChanged.emit(self.text())

    def _clear(self) -> None:
        """清空全部记录."""
        if not self._records:
            return
        self._records.clear()
        self._rebuild_table()
        self.textChanged.emit(self.text())

    def _rebuild_table(self) -> None:
        """按内部记录重建表格（发序只读，刺激量/响应可修正）."""
        self._updating = True
        try:
            self._table.setRowCount(len(self._records))
            for row, (level, hit) in enumerate(self._records):
                index_item = self._cell_item(str(row + 1))
                index_item.setFlags(index_item.flags() & ~Qt.ItemIsEditable)
                self._table.setItem(row, 0, index_item)
                self._table.setItem(row, 1, self._cell_item(f"{level:.6g}"))
                self._table.setItem(row, 2, self._cell_item("O" if hit else "X"))
        finally:
            self._updating = False

    @staticmethod
    def _cell_item(text: str) -> QTableWidgetItem:
        """构建居中数据单元格（默认可编辑，发序列另行去除可编辑标志）."""
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        return item

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """单元格编辑落地：校验并回写内部记录（非法输入回退原值）."""
        if self._updating or item is None:
            return
        row, column = item.row(), item.column()
        if row >= len(self._records) or column == 0:
            return
        level, hit = self._records[row]
        try:
            if column == 1:
                level = float(item.text())
            else:
                mark = item.text().strip().upper()
                if mark in ("O", "1"):
                    hit = 1
                elif mark in ("X", "0"):
                    hit = 0
                else:
                    raise ValueError(item.text())
        except ValueError:
            self._set_cell(row, column, f"{level:.6g}" if column == 1 else ("O" if hit else "X"))
            return
        self._records[row] = (level, hit)
        self._set_cell(row, column, f"{level:.6g}" if column == 1 else ("O" if hit else "X"))
        self.textChanged.emit(self.text())

    def _set_cell(self, row: int, column: int, text: str) -> None:
        """回写单元格文本（绕过 itemChanged 递归）."""
        self._updating = True
        try:
            item = self._table.item(row, column)
            if item is not None:
                item.setText(text)
        finally:
            self._updating = False
