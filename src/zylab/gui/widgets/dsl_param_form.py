"""DSL 参数表单：按 DslTemplate 的 params 声明自动生成输入 UI.

- 数值参数（float/int 默认值）：QDoubleSpinBox（单位后缀/范围/步进）；
- 文本参数（str 默认值）：QLineEdit；
- 派生参数（``expr`` 非空）：只读行，值随其它输入实时重算
  （:meth:`DslTemplate.evaluate`），模板应用页据此展示派生量。

与 :class:`~zylab.gui.widgets.param_form.ParamForm`（工作流图节点参数
表单）互补：本表单消费 DSL 声明的扁平参数命名空间，非节点参数引用。
"""

from __future__ import annotations

from typing import Any

from zylab.studio.dsl import DslParam, DslTemplate

from .. import theme
from ..qt_compat import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

__all__ = ["DslParamForm"]

#: 数值参数缺省范围（声明未给 min/max 时兜底，双 spin 框共用）
_DEFAULT_MIN = -1.0e9
_DEFAULT_MAX = 1.0e9


class DslParamForm(QWidget):
    """DSL 声明驱动参数表单（分组框 + 行编辑/派生只读行）."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化空表单."""
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD)
        self._layout.setSpacing(theme.SPACING_SM)
        self._template: DslTemplate | None = None
        self._spins: dict[str, QDoubleSpinBox] = {}
        self._edits: dict[str, QLineEdit] = {}
        self._derived_labels: dict[str, QLabel] = {}

    def set_template(self, template: DslTemplate) -> None:
        """按 DSL 参数分组重建表单（清空旧控件）."""
        self._template = template
        self._spins.clear()
        self._edits.clear()
        self._derived_labels.clear()
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for group in template.dsl_params:
            box = QGroupBox(group.label)
            form = QFormLayout(box)
            form.setSpacing(theme.SPACING_SM)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            for name, param in group.items:
                self._add_row(form, name, param)
            self._layout.addWidget(box)
        self._layout.addStretch()
        self._refresh_derived()

    def values(self) -> dict[str, Any]:
        """收集用户输入（非派生参数；派生量由表达式运行期求值）."""
        values: dict[str, Any] = {name: spin.value() for name, spin in self._spins.items()}
        values.update({name: edit.text() for name, edit in self._edits.items()})
        return values

    def set_fields_enabled(self, enabled: bool) -> None:
        """运行中禁用全部输入（防运行期参数漂移）."""
        for spin in self._spins.values():
            spin.setEnabled(enabled)
        for edit in self._edits.values():
            edit.setEnabled(enabled)

    # ------------------------------------------------------------------ 内部

    def _add_row(self, form: QFormLayout, name: str, param: DslParam) -> None:
        """添加单个参数行（数值/文本输入或派生只读行）."""
        label = param.label or name
        if param.derived:
            value_label = QLabel("—", objectName="derivedValue")
            value_label.setToolTip(f"派生表达式: {param.expr}")
            form.addRow(label, value_label)
            self._derived_labels[name] = value_label
            return
        if isinstance(param.value, str):
            edit = QLineEdit(str(param.value))
            edit.setToolTip(param.unit)
            edit.textChanged.connect(self._refresh_derived)
            form.addRow(label, edit)
            self._edits[name] = edit
            return
        spin = self._make_spin(name, param)
        spin.valueChanged.connect(lambda _value: self._refresh_derived())
        form.addRow(label, spin)
        self._spins[name] = spin

    def _make_spin(self, name: str, param: DslParam) -> QDoubleSpinBox:
        """按声明构建数值输入框（范围/步进/单位后缀，tooltip 带表达式说明）."""
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        low = param.min if param.min is not None else _DEFAULT_MIN
        high = param.max if param.max is not None else _DEFAULT_MAX
        spin.setRange(low, high)
        spin.setSingleStep(param.step if param.step is not None else _default_step(low, high))
        if param.value is not None and not isinstance(param.value, str):
            spin.setValue(float(param.value))
        if param.unit:
            spin.setSuffix(f" {param.unit}")
        tooltip = f"参数 {name}"
        if param.min is not None and param.max is not None:
            tooltip += f"\n范围 [{param.min}, {param.max}]"
        spin.setToolTip(tooltip)
        spin.setMinimumWidth(96)
        return spin

    def _refresh_derived(self) -> None:
        """输入变化后重算派生参数并刷新只读行（求值失败显示错误提示）."""
        if self._template is None or not self._derived_labels:
            return
        try:
            resolved = self._template.evaluate(self.values())
        except Exception:  # 表达式依赖不全/非法：保持占位，运行时统一报错
            for label in self._derived_labels.values():
                label.setText("—")
            return
        for name, label in self._derived_labels.items():
            value = resolved.get(name)
            label.setText(f"{value:.6g}" if isinstance(value, float) else str(value))


def _default_step(low: float, high: float) -> float:
    """缺省步进：范围量级的 1%（至少 0.001）."""
    return max(abs(high - low) / 100.0, 0.001)
