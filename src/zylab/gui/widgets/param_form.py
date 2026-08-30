"""参数表单生成器：按 ParamSpec 自动构建输入控件（QDoubleSpinBox/QSpinBox）.

模板 ``ui.param_groups`` 声明暴露给用户的参数分组（``"node_id.param_key"`` 引用）；
表单按组渲染 QGroupBox + QFormLayout，取值变化经 ``param_edited`` 信号上报
（由工作台页写入 WorkflowGraph.set_param，级联失效由图负责）。
"""

from __future__ import annotations

from zylab.studio import ParamGroup, ParamSpec, ParamType, WorkflowGraph

from .. import theme
from ..qt_compat import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QVBoxLayout,
    QWidget,
    Signal,
)

__all__ = ["ParamForm"]


class ParamForm(QWidget):
    """schema 驱动的参数表单（按工作流图 + 模板分组自动生成）."""

    #: 参数被编辑（node_id, param_key, 新值）
    param_edited = Signal(str, str, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化空表单."""
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        # 四周留出 MD 边距，避免分组框直贴滚动区/窗口右缘（与左栏模板库面板内边距一致）
        self._layout.setContentsMargins(theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD)
        self._layout.setSpacing(theme.SPACING_SM)
        self._fields: dict[tuple[str, str], QDoubleSpinBox] = {}
        self._graph: WorkflowGraph | None = None

    def set_graph(self, graph: WorkflowGraph, groups: tuple[ParamGroup, ...]) -> None:
        """按图与分组重建表单（清空旧控件）."""
        self._graph = graph
        self._fields.clear()
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not groups:  # 模板未声明分组时平铺全部参数
            groups = tuple(
                ParamGroup(title=node.name, params=tuple(f"{node.id}.{p.key}" for p in node.spec.params))
                for node in graph.nodes()
                if node.spec.params
            )
        for group in groups:
            box = QGroupBox(group.title)
            form = QFormLayout(box)
            form.setSpacing(theme.SPACING_SM)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            for ref in group.params:
                node_id, _, key = ref.partition(".")
                self._add_row(form, graph, node_id, key)
            self._layout.addWidget(box)
        self._layout.addStretch()

    def _add_row(self, form: QFormLayout, graph: WorkflowGraph, node_id: str, key: str) -> None:
        """添加单个参数行（标签含单位，tooltip 含说明与取值范围）."""
        node = graph.node(node_id)
        spec = node.spec.param(key)
        spin = self._make_spin(spec)
        spin.setValue(node.params[key])
        spin.valueChanged.connect(lambda value, nid=node_id, k=key: self.param_edited.emit(nid, k, value))
        label = spec.label + (f"（{spec.unit}）" if spec.unit else "")
        spin.setToolTip(f"{node.name} · {spec.label}\n{spec.doc}\n范围 [{spec.minimum}, {spec.maximum}]".strip())
        form.addRow(label, spin)
        self._fields[(node_id, key)] = spin

    @staticmethod
    def _make_spin(spec: ParamSpec) -> QDoubleSpinBox:
        """按参数类型创建输入框（整数用 0 位小数的 DoubleSpinBox 统一接口）."""
        spin = QDoubleSpinBox()
        if spec.param_type is ParamType.INT:
            spin.setDecimals(0)
        else:
            spin.setDecimals(6)
        spin.setRange(float(spec.minimum), float(spec.maximum))
        spin.setSingleStep(float(spec.step))
        return spin

    def refresh_values(self) -> None:
        """从图同步全部参数值到控件（blockSignals 防回环触发）."""
        if self._graph is None:
            return
        for (node_id, key), spin in self._fields.items():
            value = float(self._graph.node(node_id).params[key])
            if spin.value() != value:
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)

    def set_fields_enabled(self, enabled: bool) -> None:
        """运行中禁用全部输入（防止运行期变更破坏图一致性）."""
        for spin in self._fields.values():
            spin.setEnabled(enabled)
