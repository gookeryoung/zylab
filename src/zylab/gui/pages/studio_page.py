"""工作台页：Workbench 风格三区整合（工具栏 + 系统画布/结果 + 参数表单 + 底部状态栏）.

交互模型（对标 ANSYS Workbench）：
- 顶部工具栏：模板下拉选择 + 模板/工程操作图标按钮；
- 中央系统画布：整个模板为一个组合框（标题栏点击或 Ctrl+A 全选），
  环节单元单击显示其参数与结果，双击运行到该节点，右键菜单求解
  （单元：运行到此 / 强制重跑 / 查看结果；空白：运行全部）；
- 右侧参数表单：单击环节只显示该环节参数，全选显示全部参数；
- 底部状态栏：状态文本 + 进度条 + 取消按钮（进度条右侧）；
- 模板另存（用户模板存 data_dir/templates/*.json）、工程保存/打开（.zprj 内嵌模板+参数）。
"""

from __future__ import annotations

import importlib
import logging
import re
from pathlib import Path
from typing import Callable

import numpy as np

from zylab import __version__
from zylab.core import default_data_dir
from zylab.core.errors import ProjectFileError
from zylab.core.executor import EventKind
from zylab.core.project import Project
from zylab.studio import (
    ConductionBundle,
    ModelBundle,
    NodeRunEvent,
    NodeState,
    Template,
    TemplateError,
    TemplateRegistry,
    WorkflowGraph,
    WorkflowRunner,
    save_template,
)

from .. import theme
from ..icons import nav_icon
from ..qt_compat import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QObject,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSize,
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

logger = logging.getLogger(__name__)


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

    def __init__(self, parent: QWidget | None = None, data_dir: Path | None = None) -> None:
        """初始化工作台页：模板注册表（内置 + 用户目录 + 插件）+ Workbench 布局."""
        super().__init__(parent)
        self._data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
        self._registry = TemplateRegistry.with_builtin()
        self._registry.load_dir(self._data_dir / "templates")
        self._registry.load_entry_points()
        self._graph: WorkflowGraph | None = None
        self._runner: WorkflowRunner | None = None
        self._active_row = -1
        self._bridge = _StudioBridge()

        self._build_ui()
        self._connect()
        # QComboBox 首项在 blockSignals 填充期间已置 currentIndex=0，
        # 再 setCurrentIndex(0) 同值不发射信号，须显式实例化首个模板
        self._on_template_selected(0)

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        """组装布局：顶部工具栏 + 中央画布/结果 + 右侧参数 + 底部状态栏."""
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_toolbar())

        splitter = QSplitter(Qt.Horizontal)
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
        splitter.setCollapsible(1, False)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([900, 320])
        root.addWidget(splitter, stretch=1)

        root.addWidget(self._build_bottom_bar())

    def _build_toolbar(self) -> QWidget:
        """顶部工具栏：模板下拉 + 模板/工程操作图标按钮."""
        bar = QFrame(objectName="toolBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(theme.SPACING_MD, theme.SPACING_SM, theme.SPACING_MD, theme.SPACING_SM)
        layout.setSpacing(theme.SPACING_SM)
        layout.addWidget(QLabel("模板"))
        self._template_combo = QComboBox(objectName="templateCombo")
        self._template_combo.setMinimumWidth(240)
        self._template_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._reload_template_list()
        layout.addWidget(self._template_combo, stretch=1)
        layout.addStretch()

        icon_size = QSize(16, 16)
        self._save_template_button = self._tool_button("另存为模板")
        self._save_project_button = self._tool_button("保存工程 (.zprj)")
        self._open_project_button = self._tool_button("打开工程 (.zprj)")
        self._refresh_tool_icons()
        for btn in (self._save_template_button, self._save_project_button, self._open_project_button):
            btn.setIconSize(icon_size)
            layout.addWidget(btn)
        return bar

    def _build_bottom_bar(self) -> QWidget:
        """底部状态栏：状态文本 + 进度条 + 取消按钮（进度条右侧）."""
        bar = QFrame(objectName="bottomBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(theme.SPACING_MD, theme.SPACING_XS, theme.SPACING_MD, theme.SPACING_XS)
        layout.setSpacing(theme.SPACING_SM)
        self._status_label = QLabel("就绪", objectName="secondaryText")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label, stretch=1)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedWidth(220)
        layout.addWidget(self._progress)
        self._cancel_button = QPushButton("取消")
        self._cancel_button.setEnabled(False)
        layout.addWidget(self._cancel_button)
        return bar

    @staticmethod
    def _tool_button(tooltip: str) -> QPushButton:
        """构建工具行纯图标按钮（tooltip 承载完整语义，省横向空间）."""
        btn = QPushButton(objectName="toolBtn")
        btn.setToolTip(tooltip)
        return btn

    def _refresh_tool_icons(self) -> None:
        """按当前主题重绘工具行图标（单色剪影随按钮文字色着色）."""
        pal = theme.current_palette()
        for name, btn in (
            ("save_as_template", self._save_template_button),
            ("save_project", self._save_project_button),
            ("open_project", self._open_project_button),
        ):
            btn.setIcon(nav_icon(name, pal.text_on_primary))

    def _connect(self) -> None:
        """连接信号槽."""
        self._template_combo.currentIndexChanged.connect(self._on_template_selected)
        self._cancel_button.clicked.connect(self._on_cancel)
        self._save_template_button.clicked.connect(self._on_save_template_as)
        self._save_project_button.clicked.connect(self._on_save_project)
        self._open_project_button.clicked.connect(self._on_open_project)
        self._canvas.node_clicked.connect(self._on_node_clicked)
        self._canvas.node_double_clicked.connect(self._on_node_double_clicked)
        self._canvas.node_context_menu.connect(self._on_node_context_menu)
        self._canvas.background_context_menu.connect(self._on_background_context_menu)
        self._canvas.all_selected.connect(self._on_all_selected)
        self._param_form.param_edited.connect(self._on_param_edited)
        self._bridge.node_started.connect(self._on_node_started)
        self._bridge.node_progress.connect(self._on_node_progress)
        self._bridge.node_result.connect(self._on_node_result)
        self._bridge.node_failed.connect(self._on_node_failed)

    def _reload_template_list(self, select_id: str | None = None) -> None:
        """重建模板下拉（注册表变化后）；select_id 非空时选中并触发实例化."""
        self._template_combo.blockSignals(True)
        self._template_combo.clear()
        for template in self._registry.list():
            self._template_combo.addItem(template.name, template.id)
            self._template_combo.setItemData(self._template_combo.count() - 1, template.description, Qt.ToolTipRole)
        self._template_combo.blockSignals(False)
        if select_id is not None:
            index = self._template_combo.findData(select_id)
            if index >= 0:
                # 同值 setCurrentIndex 不发射信号，blockSignals 后显式触发
                self._template_combo.blockSignals(True)
                self._template_combo.setCurrentIndex(index)
                self._template_combo.blockSignals(False)
                self._on_template_selected(index)

    # ------------------------------------------------------------------ 模板实例化

    def _on_template_selected(self, row: int) -> None:
        """模板选择入口（运行中回退选择）."""
        if row < 0:
            return
        if self._runner is not None and self._runner.running:
            # 运行中禁止切换模板：回退下拉选择
            self._template_combo.blockSignals(True)
            self._template_combo.setCurrentIndex(self._active_row)
            self._template_combo.blockSignals(False)
            return
        self._active_row = row
        template = self._registry.get(self._template_combo.itemData(row))
        self._instantiate(template)

    def _instantiate(self, template: Template) -> None:
        """实例化模板：建图 + 画布/表单装配 + 源节点进程内建模预览."""
        self._shutdown_runner()
        self._graph = WorkflowGraph(template)
        self._canvas.set_graph(self._graph)
        self._param_form.set_graph(self._graph, template.param_groups)
        self._result_view.clear("尚未求解 —— 右键画布选择「运行全部」开始")
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
                logger.warning("模型预览失败: %s", exc)
                self._graph.mark_failed(node.id, f"{type(exc).__name__}: {exc}")
                continue
            self._graph.mark_result(node.id, result, 0.0)
            self._result_view.show_mesh(result)
        self._canvas.refresh_states()

    # ------------------------------------------------------------------ 模板与工程

    def _template_with_current_params(self) -> Template:
        """当前模板叠加图内最新参数."""
        assert self._graph is not None  # 调用方保证
        return self._graph.template.with_params({n.id: dict(n.params) for n in self._graph.nodes()})

    def _save_template_as(self, name: str) -> Template | None:
        """将当前图（含参数）另存为用户模板并注册；空名或空图返回 None."""
        name = name.strip()
        if not name or self._graph is None:
            return None
        base = re.sub(r"\W+", "_", name).strip("_") or "template"
        existing = {t.id for t in self._registry.list()}
        candidate = f"user.{base}"
        suffix = 2
        while candidate in existing:
            candidate = f"user.{base}_{suffix}"
            suffix += 1
        current = self._template_with_current_params()
        template = Template(
            id=candidate,
            name=name,
            nodes=current.nodes,
            discipline=current.discipline,
            description=current.description,
            tags=current.tags,
            param_groups=current.param_groups,
            results=current.results,
        )
        save_template(template, self._data_dir / "templates" / f"{candidate}.json")
        self._registry.register(template)
        self._reload_template_list(select_id=template.id)
        return template

    def _save_project(self, path: Path) -> None:
        """保存工程：模板（含当前参数）内嵌 .zprj（自包含，不依赖模板库）."""
        template = self._template_with_current_params()
        with Project.create(path, name=template.name, app_version=__version__) as proj:
            proj.write_json("model", "workflow", template.to_dict())
        self._status_label.setText(f"工程已保存: {path.name}")

    def _load_project(self, path: Path) -> None:
        """打开工程：内嵌模板注册并实例化."""
        try:
            with Project.open(path) as proj:
                data = proj.read_json("model", "workflow")
            template = Template.from_dict(data)
        except (ProjectFileError, TemplateError) as exc:
            self._status_label.setText(f"工程打开失败: {exc}")
            return
        self._registry.register(template, replace=True)
        self._reload_template_list(select_id=template.id)
        self._status_label.setText(f"工程已打开: {path.name}")

    def _on_save_template_as(self) -> None:
        """对话框：另存为模板."""
        if self._graph is None:
            return
        from ..qt_compat import QInputDialog

        name, ok = QInputDialog.getText(self, "另存为模板", "模板名称:", text=f"{self._graph.template.name} 副本")
        if ok:
            template = self._save_template_as(name)
            if template is not None:
                self._status_label.setText(f"模板已保存: {template.name}")

    def _on_save_project(self) -> None:
        """对话框：保存工程."""
        if self._graph is None:
            return
        from ..qt_compat import QFileDialog

        path_str, _ = QFileDialog.getSaveFileName(self, "保存工程", "workflow.zprj", "zylab 工程 (*.zprj)")
        if path_str:
            self._save_project(Path(path_str))

    def _on_open_project(self) -> None:
        """对话框：打开工程."""
        from ..qt_compat import QFileDialog

        path_str, _ = QFileDialog.getOpenFileName(self, "打开工程", "", "zylab 工程 (*.zprj)")
        if path_str:
            self._load_project(Path(path_str))

    # ------------------------------------------------------------------ 运行控制

    def _on_run_all(self) -> None:
        """运行全部过期节点（右键画布空白「运行全部」入口）."""
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
        """运行态 UI 切换（取消按钮/模板下拉/表单禁用）."""
        self._cancel_button.setEnabled(running)
        self._template_combo.setEnabled(not running)
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
        """单击选中：参数面板只显示该环节参数；已完成的节点同时显示其结果."""
        self._param_form.show_node(node_id)
        self._show_node_result(node_id)

    def _on_all_selected(self) -> None:
        """全选（Ctrl+A / 组合框标题栏）：参数面板显示全部参数."""
        self._param_form.show_all()

    def _on_node_double_clicked(self, node_id: str) -> None:
        """双击：已有结果查看结果，否则运行到该节点."""
        if self._graph is None:
            return
        if self._graph.node(node_id).state is NodeState.UP_TO_DATE:
            self._show_node_result(node_id)
        else:
            self._run_node(node_id)

    def _build_node_menu(self, node_id: str) -> QMenu:
        """构建单元上下文菜单（查看结果仅已完成时可用）."""
        node = self._graph.node(node_id)
        menu = QMenu(self)
        menu.addAction("运行到此节点")
        menu.addAction("强制重新运行")
        action_view = menu.addAction("查看结果")
        action_view.setEnabled(node.state is NodeState.UP_TO_DATE)
        return menu

    def _build_background_menu(self) -> QMenu:
        """构建画布空白区上下文菜单（求解入口）."""
        menu = QMenu(self)
        menu.addAction("运行全部")
        menu.addAction("全选参数")
        return menu

    def _on_node_context_menu(self, node_id: str, global_pos) -> None:
        """右键单元：运行到此 / 强制重跑 / 查看结果."""
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

    def _on_background_context_menu(self, global_pos) -> None:
        """右键画布空白：运行全部 / 全选参数."""
        if self._graph is None:
            return
        menu = self._build_background_menu()
        chosen = menu.exec(global_pos)
        if chosen is None:
            return
        if chosen.text() == "运行全部":
            self._on_run_all()
        elif chosen.text() == "全选参数":
            self._canvas.select_all()  # 经 all_selected 信号联动参数面板

    def _show_node_result(self, node_id: str) -> None:
        """在结果视图呈现节点输出（模型线框 / 分析解）."""
        if self._graph is None:
            return
        result = self._graph.node(node_id).result
        if result is None:
            return
        if isinstance(result, (ModelBundle, ConductionBundle)):
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
        """主题切换后重刷工具行图标、画布与结果视图."""
        self._refresh_tool_icons()
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
