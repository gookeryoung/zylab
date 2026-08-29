"""工作台页：模板库 + 竖向流式画布 + 参数表单 + 结果视图四区整合.

交互模型（对标 ANSYS Workbench 单元格语义）：
- 模板库双击/单击实例化模板 → 画布生成节点图，源节点即时在进程内建模预览；
- 「运行全部」按拓扑序级联求解（UP_TO_DATE 节点命中缓存自动跳过）；
- 单击节点查看其结果（模型线框/云图/曲线）；双击节点运行到该节点；
- 右键节点：运行到此 / 强制重跑 / 查看结果；
- 参数编辑即级联失效（画布徽标变为待运行），运行中表单禁用。
"""

from __future__ import annotations

import importlib
from typing import Callable

import numpy as np

from zylab.core.executor import EventKind
from zylab.studio import (
    ModelBundle,
    NodeRunEvent,
    NodeState,
    TemplateRegistry,
    WorkflowGraph,
    WorkflowRunner,
)

from .. import theme
from ..qt_compat import (
    QFrame,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QObject,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    Qt,
    QTimer,
    QVBoxLayout,
    QWidget,
    Signal,
)
from ..widgets.node_canvas import NodeCanvasWidget
from ..widgets.param_form import ParamForm
from ..widgets.result_view import ResultView

__all__ = ["StudioPage"]


def _resolve_target(target: str) -> Callable:
    """解析 ``"module:func"`` 全限定名（源节点预览的进程内直调）."""
    module_name, _, attr = target.partition(":")
    return getattr(importlib.import_module(module_name), attr)


class _StudioBridge(QObject):
    """runner 事件桥：监听线程 -> Qt 信号（跨线程自动队列到主线程）."""

    node_started = Signal(str)
    node_progress = Signal(str, float, str)
    node_result = Signal(str, object)
    node_failed = Signal(str, str)

    def dispatch(self, event: NodeRunEvent) -> None:
        """将节点运行事件转译为 Qt 信号."""
        if event.kind is EventKind.STARTED:
            self.node_started.emit(event.node_id)
        elif event.kind is EventKind.PROGRESS:
            progress, message = event.payload
            self.node_progress.emit(event.node_id, progress, message)
        elif event.kind is EventKind.RESULT:
            self.node_result.emit(event.node_id, event.payload)
        elif event.kind is EventKind.ERROR:
            self.node_failed.emit(event.node_id, str(event.payload))


class StudioPage(QWidget):
    """分析工作台页（模板配置化多学科计算工具）."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化工作台页：模板注册表 + 四区布局 + 首个模板实例化."""
        super().__init__(parent)
        self._registry = TemplateRegistry.with_builtin()
        self._graph: WorkflowGraph | None = None
        self._runner: WorkflowRunner | None = None
        self._bridge = _StudioBridge()

        self._build_ui()
        self._connect()
        self._template_list.setCurrentRow(0)  # 触发首个模板实例化

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        """组装三栏布局（模板库/画布+结果/参数表单）."""
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_library_panel())
        splitter.setCollapsible(0, False)

        center = QSplitter(Qt.Vertical)
        self._canvas = NodeCanvasWidget()
        center.addWidget(self._canvas)
        self._result_view = ResultView()
        center.addWidget(self._result_view)
        center.setStretchFactor(0, 3)
        center.setStretchFactor(1, 2)
        splitter.addWidget(center)

        right = QScrollArea()
        right.setWidgetResizable(True)
        right.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right.setFrameShape(QFrame.NoFrame)
        right.setMinimumWidth(300)
        right.setMaximumWidth(360)
        self._param_form = ParamForm()
        right.setWidget(self._param_form)
        splitter.addWidget(right)
        splitter.setCollapsible(2, False)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([240, 760, 320])

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(splitter)

    def _build_library_panel(self) -> QWidget:
        """左栏：模板库 + 运行控制."""
        panel = QWidget()
        panel.setMinimumWidth(220)
        panel.setMaximumWidth(280)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD)
        layout.setSpacing(theme.SPACING_MD)

        lib_box = QGroupBox("模板库")
        lib_layout = QVBoxLayout(lib_box)
        self._template_list = QListWidget(objectName="templateList")
        for template in self._registry.list():
            item = QListWidgetItem(template.name)
            item.setData(Qt.UserRole, template.id)
            item.setToolTip(template.description)
            self._template_list.addItem(item)
        lib_layout.addWidget(self._template_list)
        layout.addWidget(lib_box, stretch=1)

        run_box = QGroupBox("运行")
        run_layout = QVBoxLayout(run_box)
        run_layout.setSpacing(theme.SPACING_SM)
        self._run_all_button = QPushButton("运行全部")
        self._run_all_button.setMinimumHeight(36)
        self._cancel_button = QPushButton("取消")
        self._cancel_button.setEnabled(False)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._status_label = QLabel("就绪", objectName="secondaryText")
        self._status_label.setWordWrap(True)
        run_layout.addWidget(self._run_all_button)
        run_layout.addWidget(self._cancel_button)
        run_layout.addWidget(self._progress)
        run_layout.addWidget(self._status_label)
        layout.addWidget(run_box)
        return panel

    def _connect(self) -> None:
        """连接信号槽."""
        self._template_list.currentRowChanged.connect(self._on_template_selected)
        self._run_all_button.clicked.connect(self._on_run_all)
        self._cancel_button.clicked.connect(self._on_cancel)
        self._canvas.node_clicked.connect(self._on_node_clicked)
        self._canvas.node_double_clicked.connect(self._on_node_double_clicked)
        self._canvas.node_context_menu.connect(self._on_node_context_menu)
        self._param_form.param_edited.connect(self._on_param_edited)
        self._bridge.node_started.connect(self._on_node_started)
        self._bridge.node_progress.connect(self._on_node_progress)
        self._bridge.node_result.connect(self._on_node_result)
        self._bridge.node_failed.connect(self._on_node_failed)

    # ------------------------------------------------------------------ 模板实例化

    def _on_template_selected(self, row: int) -> None:
        """实例化模板：建图 + 画布/表单装配 + 源节点进程内建模预览."""
        if row < 0:
            return
        if self._runner is not None and self._runner.running:
            # 运行中禁止切换模板：回退列表选择
            self._template_list.blockSignals(True)
            self._template_list.setCurrentRow(self._active_row)
            self._template_list.blockSignals(False)
            return
        self._shutdown_runner()
        self._active_row = row
        template = self._registry.get(self._template_list.item(row).data(Qt.UserRole))
        self._graph = WorkflowGraph(template)
        self._canvas.set_graph(self._graph)
        self._param_form.set_graph(self._graph, template.param_groups)
        self._result_view.clear("尚未求解 —— 点击「运行全部」开始")
        self._status_label.setText(template.description or template.name)
        self._progress.setValue(0)
        self._preview_sources()

    def _preview_sources(self) -> None:
        """源节点进程内即时建模（纯数值、毫秒级），画布与结果视图立即呈现."""
        if self._graph is None:
            return
        for node in self._graph.nodes():
            if node.spec.inputs:
                continue
            try:
                result = _resolve_target(node.spec.target)({}, node.params)
            except Exception as exc:  # 预览失败不阻断：标记 FAILED 由画布呈现
                self._graph.mark_failed(node.id, f"{type(exc).__name__}: {exc}")
                continue
            self._graph.mark_result(node.id, result, 0.0)
            self._result_view.show_mesh(result)
        self._canvas.refresh_states()

    # ------------------------------------------------------------------ 运行控制

    def _on_run_all(self) -> None:
        """运行全部过期节点."""
        if self._graph is None:
            return
        self._runner = WorkflowRunner(self._graph)  # 每次运行自建，避免执行器复用状态
        self._set_running_ui(True)
        self._runner.run_all(self._bridge.dispatch)

    def _on_cancel(self) -> None:
        """取消当前运行."""
        if self._runner is not None:
            self._runner.cancel()
        self._set_running_ui(False)
        self._status_label.setText("已取消")
        self._canvas.refresh_states()

    def _run_node(self, node_id: str) -> None:
        """级联运行到目标节点（含过期上游）."""
        if self._graph is None or (self._runner is not None and self._runner.running):
            return
        self._runner = WorkflowRunner(self._graph)
        self._set_running_ui(True)
        self._runner.run_node(node_id, self._bridge.dispatch)

    def _set_running_ui(self, running: bool) -> None:
        """运行态 UI 切换（按钮/表单禁用）."""
        self._run_all_button.setEnabled(not running)
        self._cancel_button.setEnabled(running)
        self._param_form.set_fields_enabled(not running)
        if running:
            self._status_label.setText("运行中…")

    # ------------------------------------------------------------------ 节点事件（主线程）

    def _on_node_started(self, _node_id: str) -> None:
        """节点开始执行：刷新画布状态与动画."""
        self._canvas.refresh_states()

    def _on_node_progress(self, node_id: str, progress: float, message: str) -> None:
        """更新进度条与状态标签."""
        if self._graph is None:
            return
        self._progress.setValue(int(progress * 100))
        self._status_label.setText(f"{self._graph.node(node_id).name}: {message}")

    def _on_node_result(self, node_id: str, _result: object) -> None:
        """节点完成：刷新画布，结果节点自动呈现，队列排空后恢复 UI."""
        self._canvas.refresh_states()
        if self._graph is not None and node_id in self._graph.template.results:
            self._show_node_result(node_id)
        QTimer.singleShot(0, self._sync_idle_ui)

    def _on_node_failed(self, _node_id: str, message: str) -> None:
        """节点失败：画布标记 + 结果视图显示错误."""
        self._canvas.refresh_states()
        self._result_view.show_error(message)
        self._status_label.setText("运行失败")
        QTimer.singleShot(0, self._sync_idle_ui)

    def _sync_idle_ui(self) -> None:
        """队列排空后恢复运行控件（事件先于 handle 清空，延迟到事件循环下一轮判定）."""
        if self._runner is not None and not self._runner.running:
            self._set_running_ui(False)
            if "失败" not in self._status_label.text():
                self._status_label.setText("运行完成")

    # ------------------------------------------------------------------ 节点交互

    def _on_node_clicked(self, node_id: str) -> None:
        """单击选中：已完成的节点显示其结果."""
        self._show_node_result(node_id)

    def _on_node_double_clicked(self, node_id: str) -> None:
        """双击：已有结果查看结果，否则运行到该节点."""
        if self._graph is None:
            return
        if self._graph.node(node_id).state is NodeState.UP_TO_DATE:
            self._show_node_result(node_id)
        else:
            self._run_node(node_id)

    def _build_node_menu(self, node_id: str) -> QMenu:
        """构建节点上下文菜单（查看结果仅已完成时可用）."""
        node = self._graph.node(node_id)
        menu = QMenu(self)
        menu.addAction("运行到此节点")
        menu.addAction("强制重新运行")
        action_view = menu.addAction("查看结果")
        action_view.setEnabled(node.state is NodeState.UP_TO_DATE)
        return menu

    def _on_node_context_menu(self, node_id: str, global_pos) -> None:
        """右键菜单：运行到此 / 强制重跑 / 查看结果."""
        if self._graph is None:
            return
        menu = self._build_node_menu(node_id)
        chosen = menu.exec(global_pos)
        if chosen is None:
            return
        if chosen.text() == "运行到此节点":
            self._run_node(node_id)
        elif chosen.text() == "强制重新运行":
            self._graph.invalidate(node_id)
            self._canvas.refresh_states()
            self._run_node(node_id)
        elif chosen.text() == "查看结果":
            self._show_node_result(node_id)

    def _show_node_result(self, node_id: str) -> None:
        """在结果视图呈现节点输出（模型线框 / 分析解）."""
        if self._graph is None:
            return
        result = self._graph.node(node_id).result
        if result is None:
            return
        if isinstance(result, ModelBundle):
            self._result_view.show_mesh(result)
        else:
            self._result_view.show_solution(result, self._reference_load(node_id))

    def _reference_load(self, node_id: str) -> float:
        """屈曲临界载荷的参考载荷（上游模型工况的节点力模长合计）."""
        if self._graph is None:
            return 1.0
        for ref in self._graph.node(node_id).inputs.values():
            bundle = self._graph.node(ref.partition(".")[0]).result
            if isinstance(bundle, ModelBundle):
                total = float(sum(np.linalg.norm(load.forces) for load in bundle.case.loads))
                return total if total > 0.0 else 1.0
        return 1.0

    def _on_param_edited(self, node_id: str, key: str, value: object) -> None:
        """参数编辑：写入图（级联失效）并刷新画布."""
        if self._graph is None:
            return
        self._graph.set_param(node_id, key, value)
        self._canvas.refresh_states()
        self._status_label.setText("参数已修改，需重新运行")

    # ------------------------------------------------------------------ 生命周期

    def refresh_theme(self) -> None:
        """主题切换后重刷画布与结果视图."""
        self._canvas.refresh_theme()
        self._result_view.refresh_theme()

    def _shutdown_runner(self) -> None:
        """关闭现有 runner（模板切换时）."""
        if self._runner is not None:
            self._runner.shutdown()
            self._runner = None

    def shutdown(self) -> None:
        """关闭后台执行器（主窗口关闭时调用）."""
        self._shutdown_runner()
