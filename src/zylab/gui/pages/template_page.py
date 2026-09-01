"""模板应用页：加载 DSL 模板 -> 定制化计算界面 -> 运行 -> 结果/报告.

布局（左右分栏 + 底部运行条）：

- 左：模板说明（docs 声明）+ DSL 参数表单（:class:`DslParamForm`）；
- 右：结果多 TAB（每条 ``results`` 声明一页；curve/table/text 由
  :class:`DslResultView` 渲染，cloud 路由到既有解算视图
  :class:`~zylab.gui.widgets.result_view.ResultView`）；
- 底：加载模板 / 运行（主色）/ 导出报告（按 ``report.exports`` 声明
  写 Markdown/HTML）。

运行在线程中执行（``bind_params`` -> ``run_workflow`` 进程内拓扑序），
结果经 Qt 信号队列回主线程渲染；模板声明的主题经 ``theme_requested``
信号由主窗口应用（预览语义，不落盘）。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from zylab.studio import (
    ConductionBundle,
    ModelBundle,
    build_html,
    build_markdown,
    run_workflow,
)
from zylab.studio.dsl import DslTemplate, load_dsl
from zylab.studio.errors import StudioError, TemplateError
from zylab.studio.results import CloudData, build_result

from .. import theme
from ..icons import nav_icon
from ..qt_compat import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    Qt,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    Signal,
)
from ..widgets.dsl_param_form import DslParamForm
from ..widgets.dsl_result_view import DslResultView
from ..widgets.result_view import ResultView

__all__ = ["TemplatePage"]

#: 模板文件过滤器（YAML/JSON 双载体）
_TEMPLATE_FILTER = "DSL 模板 (*.yaml *.yml *.json);;所有文件 (*)"


def _builtin_dsl_templates() -> list[DslTemplate]:
    """注册表中的 DSL 模板（内置资产 + 用户目录，供下拉快捷加载）."""
    from zylab.core.config import default_data_dir
    from zylab.studio.registry import TemplateRegistry

    registry = TemplateRegistry.with_builtin()
    registry.load_dir(default_data_dir() / "templates")
    return [t for t in registry.list() if isinstance(t, DslTemplate)]


class TemplatePage(QWidget):
    """DSL 模板应用页（加载/参数化/运行/结果/报告导出）."""

    #: 状态栏提示（主窗口转发）
    status_message = Signal(str)
    #: 模板声明主题请求（主窗口以预览语义应用）
    theme_requested = Signal(str)
    #: 后台运行完成（outputs 载荷表, 首个错误串）
    run_finished = Signal(object, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化：占位界面（加载模板后重建）."""
        super().__init__(parent)
        self._template: DslTemplate | None = None
        self._outputs: dict[str, Any] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self.run_finished.connect(self._on_run_finished)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_body(), stretch=1)
        root.addWidget(self._build_run_bar())

    # ------------------------------------------------------------------ 布局

    def _build_body(self) -> QWidget:
        """左右分栏：左参数滚动区 | 右结果多 TAB."""
        splitter = QSplitter(Qt.Horizontal)
        self._param_scroll = QScrollArea()
        self._param_scroll.setWidgetResizable(True)
        self._param_scroll.setFrameShape(QScrollArea.NoFrame)
        self._param_panel = QWidget()
        self._param_layout = QVBoxLayout(self._param_panel)
        self._param_layout.setContentsMargins(theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD)
        self._param_layout.setSpacing(theme.SPACING_MD)
        self._docs_label = QLabel("", objectName="secondaryText")
        self._docs_label.setWordWrap(True)
        self._docs_label.setVisible(False)
        self._param_layout.addWidget(self._docs_label)
        self._param_form = DslParamForm()
        self._param_layout.addWidget(self._param_form)
        self._param_layout.addStretch()
        self._param_scroll.setWidget(self._param_panel)
        self._param_scroll.setMinimumWidth(280)
        splitter.addWidget(self._param_scroll)

        self._tabs = QTabWidget()
        self._placeholder = QLabel("加载模板后此处显示定制化计算界面", objectName="secondaryText")
        self._placeholder.setWordWrap(True)
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._tabs.addTab(self._placeholder, "结果")
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 900])
        return splitter

    def _build_run_bar(self) -> QWidget:
        """底部运行条：内置模板下拉/加载/运行/导报告 + 状态提示."""
        bar = QWidget(objectName="runBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(theme.SPACING_MD, theme.SPACING_SM, theme.SPACING_MD, theme.SPACING_SM)
        self._builtin_combo = QComboBox()
        self._builtin_combo.addItem("内置模板…")
        for template in _builtin_dsl_templates():
            self._builtin_combo.addItem(template.name, template)
        self._builtin_combo.currentIndexChanged.connect(self._on_builtin_selected)
        self._load_btn = QPushButton("加载模板")
        self._load_btn.clicked.connect(self.load_template_file)
        self._run_btn = QPushButton("运行")
        self._run_btn.setIcon(nav_icon("play", theme.current_palette().text_on_primary))
        self._run_btn.clicked.connect(self.run)
        self._run_btn.setEnabled(False)
        self._export_btn = QPushButton("导出报告", objectName="flatBtn")
        self._export_btn.clicked.connect(self.export_report)
        self._export_btn.setEnabled(False)
        self._status_label = QLabel("未加载模板", objectName="secondaryText")
        layout.addWidget(self._builtin_combo)
        layout.addWidget(self._load_btn)
        layout.addWidget(self._run_btn)
        layout.addWidget(self._export_btn)
        layout.addStretch()
        layout.addWidget(self._status_label)
        return bar

    def _on_builtin_selected(self, index: int) -> None:
        """内置模板下拉选择：加载选中 DSL 模板（占位项不动）."""
        template = self._builtin_combo.itemData(index)
        if not isinstance(template, DslTemplate):
            return
        self.load_template(template)

    # ------------------------------------------------------------------ 模板加载

    def load_template_file(self) -> None:
        """文件对话框选择 YAML/JSON 模板并加载."""
        path, _selected = QFileDialog.getOpenFileName(self, "加载 DSL 模板", "", _TEMPLATE_FILTER)
        if not path:
            return
        try:
            template = load_dsl(Path(path))
        except TemplateError as exc:
            self.status_message.emit(f"模板加载失败: {exc}")
            return
        self.load_template(template)

    def load_template(self, template: DslTemplate) -> None:
        """应用模板：重建参数表单与结果页（运行前置就绪）."""
        self._template = template
        self._outputs = {}
        if template.docs is not None and template.docs.text:
            self._docs_label.setText(template.docs.text)
            self._docs_label.setVisible(True)
        else:
            self._docs_label.setVisible(False)
        self._param_form.set_template(template)
        self._rebuild_tabs()
        self._run_btn.setEnabled(True)
        self._export_btn.setEnabled(False)
        self._status_label.setText(f"已加载: {template.name}")
        if template.theme:
            self.theme_requested.emit(template.theme)
        self.status_message.emit(f"模板已加载: {template.name}")

    # ------------------------------------------------------------------ 运行

    def run(self) -> None:
        """后台线程运行模板（参数代入 + 拓扑序执行）；防重入."""
        if self._template is None or self._running:
            return
        try:
            values = self._template.evaluate(self._param_form.values())
            executable = self._template.bind_params(values)
        except StudioError as exc:  # 参数/派生表达式错误
            self.status_message.emit(f"参数错误: {exc}")
            return
        self._running = True
        self._param_form.set_fields_enabled(False)
        self._run_btn.setEnabled(False)
        self._status_label.setText("计算中…")
        self._thread = threading.Thread(target=self._run_worker, args=(executable,), daemon=True)
        self._thread.start()

    def _run_worker(self, executable: Any) -> None:
        """工作线程：进程内拓扑序执行，完成后发信号回主线程."""
        try:
            outcome = run_workflow(executable)
        except Exception as exc:  # 线程边界兜底（执行器外异常）
            self.run_finished.emit({}, f"{type(exc).__name__}: {exc}")
            return
        if not outcome.succeeded:
            self.run_finished.emit({}, outcome.first_error())
            return
        outputs = {o.node_id: o.result for o in outcome.outcomes if o.result is not None}
        self.run_finished.emit(outputs, "")

    def _on_run_finished(self, outputs: dict, error: str) -> None:
        """主线程渲染结果（成功建各结果页，失败提示首个错误）."""
        self._running = False
        self._param_form.set_fields_enabled(True)
        self._run_btn.setEnabled(self._template is not None)
        if error:
            self._outputs = {}
            self._rebuild_tabs()
            self._status_label.setText("运行失败")
            self.status_message.emit(f"运行失败: {error}")
            return
        self._outputs = outputs
        self._render_results()
        self._export_btn.setEnabled(True)
        self._status_label.setText("运行完成")
        self.status_message.emit("模板运行完成")

    # ------------------------------------------------------------------ 结果渲染

    def _rebuild_tabs(self) -> None:
        """重建结果页签（按模板 results 声明，占位正文）."""
        self._tabs.clear()
        if self._template is None or not self._template.dsl_results:
            self._tabs.addTab(self._placeholder, "结果")
            return
        for result in self._template.dsl_results:
            self._tabs.addTab(DslResultView(), result.title)

    def _render_results(self) -> None:
        """按输出载荷渲染各结果页（cloud 路由到解算视图）."""
        if self._template is None:
            return
        self._tabs.clear()
        for result in self._template.dsl_results:
            try:
                data = build_result(result, self._outputs)
            except TemplateError as exc:
                page = DslResultView()
                page.set_error(str(exc))
                self._tabs.addTab(page, result.title)
                continue
            if isinstance(data, CloudData):
                self._tabs.addTab(self._build_cloud_page(data), result.title)
            else:
                page = DslResultView()
                page.set_data(data)
                self._tabs.addTab(page, result.title)
        if self._tabs.count() == 0:  # 无 results 声明：保持占位页
            self._tabs.addTab(self._placeholder, "结果")

    def _build_cloud_page(self, data: CloudData) -> QWidget:
        """云图页：既有解算视图渲染（解对象/模型预览，失败回落错误页）."""
        view = ResultView()
        payload = data.payload
        if isinstance(payload, (ModelBundle, ConductionBundle)):
            view.show_mesh(payload)
        else:
            try:
                view.show_solution(payload)
            except Exception:  # 载荷非解对象：解算视图无法分发
                view.show_error(f"节点 {data.node_id!r} 输出暂不支持云图渲染")
        return view

    # ------------------------------------------------------------------ 报告导出

    def export_report(self) -> None:
        """按模板 report.exports 声明导出报告文件（md/html 多选保存）."""
        if self._template is None or not self._outputs:
            return
        exports = self._template.report.exports if self._template.report is not None else ("html",)
        default_stem = f"{self._template.id.replace('.', '_')}_报告"
        selected = "Markdown 报告 (*.md);;HTML 报告 (*.html)" if "md" in exports else "HTML 报告 (*.html)"
        path, chosen = QFileDialog.getSaveFileName(self, "导出报告", default_stem, selected)
        if not path:
            return
        try:
            if chosen.startswith("Markdown"):
                Path(path).write_text(build_markdown(self._template, self._outputs), encoding="utf-8")
            else:
                Path(path).write_text(build_html(self._template, self._outputs), encoding="utf-8")
        except OSError as exc:
            self.status_message.emit(f"报告导出失败: {exc}")
            return
        self.status_message.emit(f"报告已导出: {path}")

    # ------------------------------------------------------------------ 主题/生命周期

    def refresh_theme(self) -> None:
        """主题切换后刷新运行按钮图标."""
        self._run_btn.setIcon(nav_icon("play", theme.current_palette().text_on_primary))
