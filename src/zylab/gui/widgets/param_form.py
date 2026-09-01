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

#: 可生成数值输入行的参数类型（STR/MAP 为 compute 节点内部结构参数，不暴露）
_NUMERIC_TYPES = (ParamType.FLOAT, ParamType.INT)


class ParamForm(QWidget):
    """schema 驱动的参数表单（按工作流图 + 模板分组自动生成）."""

    #: 参数被编辑（node_id, param_key, 新值）
    param_edited = Signal(str, str, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化空表单."""
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        # 四周留出 MD 边距，避免分组框直贴滚动区/窗口右缘（与工具栏面板内边距一致）
        self._layout.setContentsMargins(theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD)
        self._layout.setSpacing(theme.SPACING_SM)
        self._fields: dict[tuple[str, str], QDoubleSpinBox] = {}
        self._graph: WorkflowGraph | None = None
        # 分组行表：(QGroupBox, 该组内行引用列表 [("node_id.key", spin)])，用于按选中节点过滤
        self._group_rows: list[tuple[QGroupBox, list[tuple[str, QDoubleSpinBox]]]] = []

    def set_graph(self, graph: WorkflowGraph, groups: tuple[ParamGroup, ...]) -> None:
        """按图与分组重建表单（清空旧控件），默认显示全部参数."""
        self._graph = graph
        self._fields.clear()
        self._group_rows.clear()
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not groups:  # 模板未声明分组时平铺全部数值参数
            groups = tuple(
                ParamGroup(
                    title=node.name,
                    params=tuple(f"{node.id}.{p.key}" for p in node.spec.params if p.param_type in _NUMERIC_TYPES),
                )
                for node in graph.nodes()
                if any(p.param_type in _NUMERIC_TYPES for p in node.spec.params)
            )
        for group in groups:
            box = QGroupBox(group.title)
            form = QFormLayout(box)
            form.setSpacing(theme.SPACING_SM)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            rows: list[tuple[str, QDoubleSpinBox]] = []
            for ref in group.params:
                node_id, _, key = ref.partition(".")
                if graph.node(node_id).spec.param(key).param_type not in _NUMERIC_TYPES:
                    continue  # STR/MAP 结构参数不生成输入行
                spin = self._add_row(form, graph, node_id, key)
                rows.append((ref, spin))
            self._group_rows.append((box, rows))
            self._layout.addWidget(box)
        self._layout.addStretch()
        self.show_all()

    def _add_row(self, form: QFormLayout, graph: WorkflowGraph, node_id: str, key: str) -> QDoubleSpinBox:
        """添加单个参数行并返回输入框（单位入框内后缀，tooltip 含说明与取值范围）."""
        node = graph.node(node_id)
        spec = node.spec.param(key)
        spin = self._make_spin(spec)
        # 单位放框内后缀而非标签：标签过长会把表单最小宽撑出滚动区视口，右侧被裁剪
        if spec.unit:
            spin.setSuffix(f" {spec.unit}")
        spin.setValue(node.params[key])
        spin.valueChanged.connect(lambda value, nid=node_id, k=key: self.param_edited.emit(nid, k, value))
        spin.setToolTip(f"{node.name} · {spec.label}\n{spec.doc}\n范围 [{spec.minimum}, {spec.maximum}]".strip())
        form.addRow(spec.label, spin)
        self._fields[(node_id, key)] = spin
        return spin

    # ------------------------------------------------------------------ 选择过滤

    def show_node(self, node_id: str) -> None:
        """仅显示指定节点的参数行（Workbench 单选环节语义）.

        隐藏行经 label + 输入框 setVisible(False)（QFormLayout 自动跳过
        隐藏行）；全部行被过滤掉的分组整体隐藏。
        """
        self._filter_rows(lambda ref: ref.partition(".")[0] == node_id)

    def show_all(self) -> None:
        """显示全部参数行（全选语义，set_graph 后的默认态）."""
        self._filter_rows(lambda _ref: True)

    def _filter_rows(self, predicate) -> None:
        """按谓词过滤行可见性：行引用 "node_id.key" 命中谓词才显示."""
        for box, rows in self._group_rows:
            any_visible = False
            for ref, spin in rows:
                visible = predicate(ref)
                any_visible = any_visible or visible
                spin.setVisible(visible)
                label = spin.parentWidget().layout().labelForField(spin) if spin.parentWidget() else None
                if label is not None:
                    label.setVisible(visible)
            box.setVisible(any_visible)

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
        # QDoubleSpinBox 的 minimumSizeHint 按最大值全位数（如 999999999.000000）计算，
        # 大范围 + 6 位小数会把表单最小宽撑到 ~600px，超出滚动区视口后右侧被裁剪
        # （横向滚动已禁用）；显式最小宽度覆盖该提示，字段仍可随面板伸展。
        spin.setMinimumWidth(96)
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
