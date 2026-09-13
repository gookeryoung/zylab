"""参数化计算应用页：加载 DSL 参数化计算 -> 定制化计算界面 -> 运行 -> 结果/报告导出.

布局（左右分栏 + 底部运行条，三区边界以 1px 分隔线与卡片化面板呈现）：

- 左：QTabWidget 分页（第一页「参数」= DSL 参数表单
  :class:`DslParamForm`，第二页「说明」= 图文引导面板
  :class:`DocsPanel`，说明卡按模板 ``docs`` 声明动态显隐）；
- 右：结果多 TAB（普通结果默认合并「结果」流页；显式 ``group`` 为单独页签；
  cloud 路由到既有解算视图 :class:`~zylab.gui.widgets.result_view.ResultView`）；
- 底：加载参数化计算 / 运行（主色）/ 导出报告（按 ``report.exports`` 声明
  写 Markdown/HTML）。

运行在线程中执行（``bind_params`` -> ``run_workflow`` 进程内拓扑序），
结果经 Qt 信号队列回主线程渲染；参数化计算声明的主题经 ``theme_requested``
信号由主窗口应用（预览语义，不落盘）。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from zylab.flowchart import (
    ConductionBundle,
    ModelBundle,
    build_html,
    build_markdown,
    run_workflow,
)
from zylab.flowchart.dsl import DslTemplate, load_dsl
from zylab.flowchart.errors import FlowchartError, TemplateError
from zylab.flowchart.results import CloudData, build_result

from .. import theme
from ..icons import nav_icon
from ..qt_compat import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPalette,
    QPushButton,
    QScrollArea,
    QSplitter,
    Qt,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    Signal,
    exec_dialog,
)
from ..widgets.docs_panel import DocsPanel
from ..widgets.dsl_param_form import DslParamForm
from ..widgets.result_view import ResultView
from ..widgets.stream_view import ResultStreamView
from ..widgets.template_dialog import TemplateDialog

__all__ = ["TemplatePage"]

#: 参数化计算文件过滤器（YAML/JSON 双载体）
_TEMPLATE_FILTER = "DSL 参数化计算 (*.yaml *.yml *.json);;所有文件 (*)"

#: 默认无分组普通结果的页签名
_DEFAULT_GROUP = "结果"

#: 左侧栏 Tab 索引
_TAB_PARAM = 0
_TAB_DOCS = 1


def _builtin_dsl_templates() -> list[DslTemplate]:
    """注册表中的 DSL 参数化计算（内置资产 + 用户目录，供下拉快捷加载）."""
    from zylab.core.config import default_data_dir
    from zylab.flowchart.registry import TemplateRegistry

    registry = TemplateRegistry.with_builtin()
    registry.load_dir(default_data_dir() / "templates")
    return [t for t in registry.list() if isinstance(t, DslTemplate)]


class TemplatePage(QWidget):
    """DSL 参数化计算应用页（加载/参数化/运行/结果/报告导出）."""

    #: 状态栏提示（主窗口转发）
    status_message = Signal(str)
    #: 参数化计算声明主题请求（主窗口以预览语义应用）
    theme_requested = Signal(str)
    #: 后台运行完成（outputs 载荷表, 首个错误串）
    run_finished = Signal(object, str)
    #: 运行生命周期状态变更（主窗口右下 indicator 统一承载：idle/running/success/error）
    run_state_changed = Signal(str, str)  # (state, detail)

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化：占位界面（加载参数化计算后重建）."""
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
        """左右分栏：左侧 Tab 分页（参数/说明） | 右侧结果多 TAB."""
        splitter = QSplitter(Qt.Horizontal)
        self._param_scroll = QScrollArea()
        self._param_scroll.setWidgetResizable(True)
        self._param_scroll.setFrameShape(QScrollArea.NoFrame)
        self._param_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._param_panel = QFrame(objectName="sidePanel")
        self._param_layout = QVBoxLayout(self._param_panel)
        self._param_layout.setContentsMargins(theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD)
        self._param_layout.setSpacing(theme.SPACING_MD)

        # 左侧 Tab 分页：参数（默认） | 说明
        self._side_tabs = QTabWidget(objectName="sideTabs")
        self._param_form = DslParamForm()
        self._docs_panel = DocsPanel()
        # Windows Fusion 下 QTabWidget::pane QSS 不总能覆盖 QStackedWidget 子控件
        # 的重绘（尤其切 tab 或加载模板重建内容时），需设 setAutoFillBackground(True)
        # + 注入主题色 palette，避免非激活 tab 内容穿透重叠。
        self._param_form.setAutoFillBackground(True)
        self._docs_panel.setAutoFillBackground(True)
        self._param_form.setPalette(self._tab_page_palette())
        self._docs_panel.setPalette(self._tab_page_palette())
        self._side_tabs.addTab(self._param_form, "参数")
        self._side_tabs.addTab(self._docs_panel, "说明")
        # 初始隐藏说明页（加载模板后按 docs 声明决定显隐）
        self._side_tabs.setTabVisible(_TAB_DOCS, False)
        self._param_layout.addWidget(self._side_tabs)

        self._param_scroll.setWidget(self._param_panel)
        self._param_scroll.setMinimumWidth(300)
        splitter.addWidget(self._param_scroll)

        self._tabs = QTabWidget(objectName="resultTabs")
        self._placeholder = QLabel("加载后此处显示定制化计算界面", objectName="secondaryText")
        self._placeholder.setWordWrap(True)
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._tabs.addTab(self._placeholder, "结果")
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setSizes([360, 900])
        return splitter

    def _build_run_bar(self) -> QWidget:
        """底部运行条：参数化计算市场/加载/运行/导报告 + 状态提示."""
        bar = QFrame(objectName="runBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(theme.SPACING_MD, theme.SPACING_SM, theme.SPACING_MD, theme.SPACING_SM)
        layout.setSpacing(theme.SPACING_SM)
        self._market_btn = QPushButton("参数化计算市场")
        self._market_btn.clicked.connect(self.open_market)
        self._load_btn = QPushButton("加载参数化计算")
        self._load_btn.clicked.connect(self.load_template_file)
        self._run_btn = QPushButton("运行")
        self._run_btn.setIcon(nav_icon("play", theme.current_palette().text_on_primary))
        self._run_btn.clicked.connect(self.run)
        self._run_btn.setEnabled(False)
        self._export_btn = QPushButton("导出报告", objectName="flatBtn")
        self._export_btn.clicked.connect(self.export_report)
        self._export_btn.setEnabled(False)
        layout.addWidget(self._market_btn)
        layout.addWidget(self._load_btn)
        layout.addSpacing(theme.SPACING_MD)
        layout.addWidget(self._run_btn)
        layout.addWidget(self._export_btn)
        layout.addStretch()
        return bar

    def open_market(self) -> None:
        """打开参数化计算市场对话框（分组/搜索/详情），确认后加载选中 DSL 参数化计算."""
        templates = _builtin_dsl_templates()
        if not templates:
            self.status_message.emit("参数化计算市场为空：未发现 DSL 参数化计算")
            return
        dialog = TemplateDialog(templates, self)
        dialog.setWindowTitle("参数化计算市场")
        if exec_dialog(dialog) and dialog.selected_id is not None:
            template = next((t for t in templates if t.id == dialog.selected_id), None)
            if template is not None:
                self.load_template(template)

    # ------------------------------------------------------------------ 参数化计算加载

    def load_template_file(self) -> None:
        """文件对话框选择 YAML/JSON 参数化计算并加载."""
        path, _selected = QFileDialog.getOpenFileName(self, "加载 DSL 参数化计算", "", _TEMPLATE_FILTER)
        if not path:
            return
        try:
            template = load_dsl(Path(path))
        except TemplateError as exc:
            self.status_message.emit(f"加载失败: {exc}")
            return
        self.load_template(template)

    def load_template(self, template: DslTemplate) -> None:
        """应用参数化计算：重建说明卡/参数表单与结果页（运行前置就绪）."""
        self._template = template
        self._outputs = {}
        self._docs_panel.set_template(template)
        self._param_form.set_template(template)
        self._rebuild_tabs()
        self._run_btn.setEnabled(True)
        self._export_btn.setEnabled(False)
        self.run_state_changed.emit("idle", f"已加载: {template.name}")
        if template.theme:
            self.theme_requested.emit(template.theme)
        self.status_message.emit(f"已加载: {template.name}")
        # 说明页显隐：按模板 docs 声明有无内容直接判断
        docs = template.docs
        has_docs = bool(docs) and bool((docs.text or "").strip() or (docs.image or "").strip())
        self._side_tabs.setTabVisible(_TAB_DOCS, has_docs)
        self._side_tabs.setCurrentIndex(_TAB_PARAM)

    # ------------------------------------------------------------------ 运行

    def run(self) -> None:
        """后台线程运行参数化计算（参数代入 + 拓扑序执行）；防重入."""
        if self._template is None or self._running:
            return
        try:
            values = self._template.evaluate(self._param_form.values())
            executable = self._template.bind_params(values)
        except FlowchartError as exc:  # 参数/派生表达式错误
            self.status_message.emit(f"参数错误: {exc}")
            return
        self._running = True
        self._param_form.set_fields_enabled(False)
        self._run_btn.setEnabled(False)
        self.run_state_changed.emit("running", f"计算 {self._template.name}")
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
        """主线程渲染结果（成功建各结果页，失败提示首个错误）；成败发 run_finished 供主窗口 indicator 消费."""
        self._running = False
        self._param_form.set_fields_enabled(True)
        self._run_btn.setEnabled(self._template is not None)
        if error:
            self._outputs = {}
            self._rebuild_tabs()
            self.run_state_changed.emit("error", error)
            # 失败详情用临时消息 showMessage（3秒消失），成败图标由 indicator 承载
            self.status_message.emit(f"运行失败: {error}")
            return
        self._outputs = outputs
        self._render_results()
        self._export_btn.setEnabled(True)
        self.run_state_changed.emit("success", f"共 {len(self._outputs)} 个节点产出结果")

    # ------------------------------------------------------------------ 结果渲染

    def _result_pages(self) -> list[tuple[str, list[Any], bool]]:
        """按 ``group`` 声明聚合结果为页序列.

        返回 ``(页签名, 结果列表, 是否 cloud 独立页)`` 三元组。

        - 未声明 group 且非 cloud 的结果 → 默认「结果」流页；
        - 显式 ``group`` 且非 cloud → 按组名聚合为一页；
        - cloud 结果 → 独立页签（整页解算视图，不参与流）。

        无 results 声明时返回空列表（调用方回退占位页）。
        """
        if self._template is None:
            return []
        # 先聚合普通结果（按 group 或默认），cloud 独立
        stream_groups: dict[str, list[Any]] = {}
        cloud_results: list[Any] = []
        for result in self._template.dsl_results:
            if result.kind == "cloud":
                cloud_results.append(result)
            elif result.group:
                stream_groups.setdefault(result.group, []).append(result)
            else:
                stream_groups.setdefault(_DEFAULT_GROUP, []).append(result)
        pages: list[tuple[str, list[Any], bool]] = []
        for name, results in stream_groups.items():
            pages.append((name, results, False))
        for cloud in cloud_results:
            pages.append((cloud.title, [cloud], True))
        return pages

    def _rebuild_tabs(self) -> None:
        """重建结果页签（按组聚合，占位正文；单页隐藏页签条）."""
        self._tabs.clear()
        pages = self._result_pages()
        if not pages:
            self._tabs.addTab(self._placeholder, "结果")
        else:
            for title, _results, _is_cloud in pages:
                page = ResultStreamView()
                self._tabs.addTab(page, title)
        self._tabs.tabBar().setVisible(self._tabs.count() > 1)

    def _render_results(self) -> None:
        """按输出载荷渲染各结果页（流页分块，cloud 路由到解算视图）."""
        if self._template is None:
            return
        self._tabs.clear()
        for title, results, is_cloud in self._result_pages():
            if is_cloud:
                # cloud 页：路由到解算视图
                result = results[0]
                try:
                    data = build_result(result, self._outputs)
                except TemplateError as exc:
                    page = ResultStreamView()
                    page.set_error(str(exc))
                    self._tabs.addTab(page, title)
                    continue
                if isinstance(data, CloudData):
                    self._tabs.addTab(self._build_cloud_page(data), title)
                else:
                    # build_result 未产生 CloudData：用流页显示非 cloud 载荷错误
                    page = ResultStreamView()
                    page.set_error(f"节点 {getattr(result, 'node_id', '?')!r} 输出暂不支持云图渲染")
                    self._tabs.addTab(page, title)
                continue
            # 流页：逐块渲染，块级失败显示错误文本
            blocks: list[tuple[str, Any, str]] = []
            for result in results:
                try:
                    blocks.append((result.title, build_result(result, self._outputs), ""))
                except TemplateError as exc:
                    blocks.append((result.title, str(exc), "danger"))
            page = ResultStreamView()
            page.set_blocks(blocks)
            self._tabs.addTab(page, title)
        if self._tabs.count() == 0:  # 无 results 声明：保持占位页
            self._tabs.addTab(self._placeholder, "结果")
        self._tabs.tabBar().setVisible(self._tabs.count() > 1)

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
        """按参数化计算 report.exports 声明导出报告文件（md/html 多选保存）."""
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

    def _tab_page_palette(self) -> QPalette:
        """为 tab page 子控件构建 QPalette.Window = 主题 bg_app 的调色板.

        配合 ``setAutoFillBackground(True)`` 让 tab page 自身画不透明底，
        避免 Windows Fusion 下 QStackedWidget 非激活子控件穿透。
        """
        pal = QPalette()
        pal.setColor(QPalette.Window, theme.current_palette().bg_app)
        return pal

    def refresh_theme(self) -> None:
        """主题切换后刷新运行按钮图标、tab page 背景与说明卡正文配色."""
        self._run_btn.setIcon(nav_icon("play", theme.current_palette().text_on_primary))
        # tab page 子控件的 palette.Window 必须同步换色，否则 setAutoFillBackground
        # 仍按旧色画底（深→浅切了白，浅→深切了黑）。
        tab_pal = self._tab_page_palette()
        self._param_form.setPalette(tab_pal)
        self._docs_panel.setPalette(tab_pal)
        self._docs_panel.refresh_theme()
