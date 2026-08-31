"""变量浏览器组件：三列模型（类型列带标签）+ 标签委托 + 变量详情对话框.

- :class:`VarTableModel`：变量表数据模型（列：名称/类型/字节数），
  类型列经 :attr:`VarTableModel.TAGS_ROLE` 提供元素类型/形状标签；
- :class:`VarTagDelegate`：类型列委托，类型名后绘制圆角标签 chip；
- :class:`VarDetailDialog`：MATLAB 变量编辑器风格详情对话框（只读），
  标量/非数组用文本 repr，一维列向量、二维矩阵、多维数组提供
  前置维索引切换（slice 导航）+ 摘要统计（最小/最大/均值）。
"""

from __future__ import annotations

import numpy as np

from zylab.sci import VarInfo

from .. import theme
from ..qt_compat import (
    QAbstractItemView,
    QAbstractTableModel,
    QApplication,
    QColor,
    QDialog,
    QDialogButtonBox,
    QFont,
    QHBoxLayout,
    QLabel,
    QModelIndex,
    QRect,
    QSpinBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

__all__ = ["VarDetailDialog", "VarTableModel", "VarTagDelegate", "mono_font"]

_INVALID_INDEX = QModelIndex()

#: 详情表显示上限（超出截断并提示，避免万行级数组卡死界面）
_MAX_ROWS = 200
_MAX_COLS = 60


def mono_font() -> QFont:
    """等宽字体（与脚本页一致）."""
    return QFont(theme.FONT_MONO.strip('"').split(",")[0], 10)


def _fmt_cell(value: object) -> str:
    """数值单元格格式化（浮点 6 位有效数字，与 MATLAB 默认显示相近）."""
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".6g")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


class VarTableModel(QAbstractTableModel):
    """变量浏览器数据模型（列：名称/类型/字节数；形状与元素类型作类型列标签）."""

    _HEADERS = ("名称", "类型", "字节数")

    #: 类型列标签角色（tuple[str, ...]：元素类型 + 形状，空项省略）
    TAGS_ROLE = Qt.UserRole + 1

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化空模型."""
        super().__init__(parent)
        self._vars: list[VarInfo] = []

    def set_vars(self, infos: list[VarInfo]) -> None:
        """整体替换变量列表并刷新视图."""
        self.beginResetModel()
        self._vars = list(infos)
        self.endResetModel()

    def info_at(self, row: int) -> VarInfo | None:
        """取指定行的变量描述（越界返回 None）."""
        if 0 <= row < len(self._vars):
            return self._vars[row]
        return None

    def rowCount(self, parent: QModelIndex = _INVALID_INDEX) -> int:
        """行数（顶层）."""
        return 0 if parent.isValid() else len(self._vars)

    def columnCount(self, _parent: QModelIndex = _INVALID_INDEX) -> int:
        """列数."""
        return len(self._HEADERS)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> object:
        """单元格数据：类型列 DisplayRole 为类型名，标签经 TAGS_ROLE 提供给委托."""
        if not index.isValid():
            return None
        info = self._vars[index.row()]
        if role == Qt.DisplayRole:
            return (info.name, info.type_name, str(info.nbytes))[index.column()]
        if role == self.TAGS_ROLE and index.column() == 1:
            return tuple(tag for tag in (info.dtype, info.shape) if tag)
        if role == Qt.ForegroundRole:
            pal = theme.current_palette()
            return QColor(pal.text_secondary if info.builtin else pal.text_primary)
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> object:
        """表头数据."""
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self._HEADERS[section]
        return None


class VarTagDelegate(QStyledItemDelegate):
    """类型列委托：类型名文本后绘制元素类型/形状圆角标签."""

    def paint(self, painter, option: QStyleOptionViewItem, index) -> None:  # Qt 命名约定
        """先绘制默认背景，再画类型文本，右侧追加标签 chip."""
        tags = index.data(VarTableModel.TAGS_ROLE) or ()
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        style = opt.widget.style() if opt.widget is not None else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, opt.widget)
        pal = theme.current_palette()
        painter.save()
        painter.setPen(QColor(pal.text_primary))
        rect = option.rect.adjusted(6, 0, -2, 0)
        text = index.data(Qt.DisplayRole) or ""
        painter.drawText(rect, Qt.AlignVCenter | Qt.AlignLeft, text)
        # 标签 chip：小号字 + 圆角底色，空间不足时跳过
        chip_font = painter.font()
        chip_font.setPointSizeF(max(7.0, chip_font.pointSizeF() - 1.5))
        painter.setFont(chip_font)
        metrics = painter.fontMetrics()
        x = rect.x() + metrics.horizontalAdvance(text) + 8
        chip_height = metrics.height() + 4
        y = option.rect.y() + (option.rect.height() - chip_height) // 2
        for tag in tags:
            width = metrics.horizontalAdvance(tag) + 10
            if x + width > option.rect.right():
                break
            chip = QRect(x, y, width, chip_height)
            painter.setPen(QColor(pal.border))
            painter.setBrush(QColor(pal.bg_input))
            painter.drawRoundedRect(chip, 4, 4)
            painter.setPen(QColor(pal.text_secondary))
            painter.drawText(chip, Qt.AlignCenter, tag)
            x += width + 4
        painter.restore()


class VarDetailDialog(QDialog):
    """变量详情对话框（MATLAB 变量编辑器只读风格）.

    - 标量/非数组对象：等宽文本 repr（截断保护）；
    - 一维数组：索引 + 值 矩阵视图（列向量）；
    - 二维数组：行列索引表头的矩阵视图；
    - 多维数组（ndim>=3）：前置维逐维索引微调（slice 导航），
      后两维固定显示矩阵切片。
    """

    def __init__(self, info: VarInfo, value: object, parent: QWidget | None = None) -> None:
        """构建摘要区 + 按维度分发的详情视图.

        Args:
            info: 变量描述（名称/类型/形状/元素类型/字节数）。
            value: 变量当前值（取自内核命名空间）。
        """
        super().__init__(parent)
        self.setWindowTitle(f"变量详情 - {info.name}")
        self.resize(560, 600)
        self._info = info
        self._value = value
        self._ndim_spinners: list[QSpinBox] = []

        root = QVBoxLayout(self)
        root.addWidget(self._build_summary())
        if isinstance(value, np.ndarray) and value.ndim >= 1:
            if value.ndim >= 3:
                root.addWidget(self._build_slice_bar())
            root.addWidget(self._build_matrix(), stretch=1)
            self._refresh_matrix()
        else:
            root.addWidget(self._build_text_view(), stretch=1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ------------------------------------------------------------ 摘要与构建

    def _build_summary(self) -> QLabel:
        """摘要行：名称/类型/形状/元素类型/字节数 + 数值统计."""
        info = self._info
        parts = [f"名称 {info.name}", f"类型 {info.type_name}"]
        if info.shape:
            parts.append(f"形状 {info.shape}")
        if info.dtype:
            parts.append(f"元素类型 {info.dtype}")
        parts.append(f"字节数 {info.nbytes}")
        stats = _stats_text(self._value)
        if stats:
            parts.append(stats)
        label = QLabel(" · ".join(parts))
        label.setWordWrap(True)
        return label

    def _build_slice_bar(self) -> QWidget:
        """多维数组前置维索引条（每维一个微调框，改动即刷新切片）."""
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        value: np.ndarray = self._value  # type: ignore[assignment]
        for dim, size in enumerate(value.shape[:-2]):
            layout.addWidget(QLabel(f"第 {dim + 1} 维"))
            spin = QSpinBox()
            spin.setRange(0, size - 1)
            spin.valueChanged.connect(lambda _v, d=dim: self._on_slice_changed(d))
            layout.addWidget(spin)
            self._ndim_spinners.append(spin)
        layout.addStretch(1)
        return bar

    def _build_text_view(self) -> QLabel:
        """非数组详情：等宽 repr 文本（超长截断保护）."""
        text = self._info.preview or repr(self._value)
        if len(text) > 2000:
            text = text[:2000] + "\n…（超长截断）"
        label = QLabel(f"<pre style='margin:0'>{text}</pre>")
        label.setFont(mono_font())
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return label

    # ------------------------------------------------------------ 矩阵视图

    def _on_slice_changed(self, _dim: int) -> None:
        """前置维索引变动：重建矩阵切片（同维联动保持互斥索引无意义，直接全刷）."""
        self._refresh_matrix()

    def _current_slice(self) -> np.ndarray:
        """按前置维索引取当前二维切片."""
        value: np.ndarray = self._value  # type: ignore[assignment]
        idx = tuple(spin.value() for spin in self._ndim_spinners)
        return value[idx] if idx else value

    def _build_matrix(self) -> QTableWidget:
        """构建矩阵视图（一维列向量/二维矩阵/多维当前切片共用）."""
        self._table = QTableWidget()
        self._table.setFont(mono_font())
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        return self._table

    def _refresh_matrix(self) -> None:
        """按当前切片重建矩阵表（行列索引作表头，超限截断并提示）."""
        arr = self._current_slice()
        if arr.ndim == 0:  # 前置维索引后剩单个元素
            matrix, cols = arr.reshape(1, 1), 1
        elif arr.ndim == 1:
            matrix, cols = arr.reshape(-1, 1), 1
        else:
            matrix, cols = arr, arr.shape[-1]
        rows = matrix.shape[0]
        shown_rows = min(rows, _MAX_ROWS)
        shown_cols = min(cols, _MAX_COLS)
        self._table.setRowCount(shown_rows)
        self._table.setColumnCount(shown_cols)
        self._table.setHorizontalHeaderLabels([str(c) for c in range(shown_cols)])
        self._table.setVerticalHeaderLabels([str(r) for r in range(shown_rows)])
        for r in range(shown_rows):
            for c in range(shown_cols):
                self._table.setItem(r, c, QTableWidgetItem(_fmt_cell(matrix[r, c])))
        truncated = rows > shown_rows or cols > shown_cols
        base = f"变量详情 - {self._info.name}"
        self.setWindowTitle(f"{base}（仅显示前 {shown_rows} 行 × {shown_cols} 列）" if truncated else base)


def _stats_text(value: object) -> str:
    """数值 ndarray 的统计摘要（最小/最大/均值，空集或非数值返回空串）."""
    if not isinstance(value, np.ndarray) or value.size == 0:
        return ""
    if not np.issubdtype(value.dtype, np.number):
        return ""
    finite = value[np.isfinite(value)]  # 全 NaN/Inf 时统计降级为空
    if finite.size == 0:
        return ""
    return f"最小 {_fmt_cell(np.min(finite))} · 最大 {_fmt_cell(np.max(finite))} · 均值 {_fmt_cell(np.mean(finite))}"
