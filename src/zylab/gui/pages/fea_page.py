"""FEA 分析页：内置示例模型 + 参数编辑 + 进程隔离后台求解 + 云图可视化.

支持静力（变形云图）与模态（频率表 + 振型云图切换）两类分析。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg

from zylab.core.executor import EventKind, ProcessExecutor, TaskEvent, TaskSpec
from zylab.fea import (
    Constraint,
    ElementBlock,
    ElementType,
    LinearElastic,
    Mesh,
    ModalSolution,
    NodalLoad,
    Section,
    StaticCase,
)
from zylab.fea.viewdata import deformed_coords, displacement_field, edge_segments, mesh_edges, scalar_colors

from .. import theme
from ..qt_compat import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QObject,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    Signal,
)

if TYPE_CHECKING:
    from zylab.fea import StaticSolution

__all__ = ["FeaPage"]

# 悬臂梁示例：L x H = 40 x 8，单元长宽比保持 1（规避 CPS4 剪切自锁放大）
_BEAM_LENGTH = 40.0
_BEAM_HEIGHT = 8.0
_BEAM_NX = 40
_BEAM_NY = 8
_BEAM_TIP_LOAD = -100.0  # 右边缘每节点竖向力


class _SolveBridge(QObject):
    """求解事件桥：executor 监听线程 -> Qt 信号（跨线程自动队列到主线程）."""

    progress = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str)

    def dispatch(self, event: TaskEvent) -> None:
        """将 executor 任务事件转译为 Qt 信号."""
        if event.kind is EventKind.PROGRESS:
            progress, message = event.payload
            self.progress.emit(progress, message)
        elif event.kind is EventKind.RESULT:
            self.finished.emit(event.payload)
        elif event.kind is EventKind.ERROR:
            payload = event.payload
            self.failed.emit(f"{payload.get('exc_type', '错误')}: {payload.get('message', '')}")


def build_cantilever_mesh() -> Mesh:
    """构建悬臂梁 Q4 结构网格（单元正方形，长宽比 1）."""
    xs = np.linspace(0.0, _BEAM_LENGTH, _BEAM_NX + 1)
    ys = np.linspace(0.0, _BEAM_HEIGHT, _BEAM_NY + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    conn = []
    for j in range(_BEAM_NY):
        for i in range(_BEAM_NX):
            n00 = j * (_BEAM_NX + 1) + i
            conn.append((n00, n00 + 1, n00 + _BEAM_NX + 2, n00 + _BEAM_NX + 1))
    block = ElementBlock(etype=ElementType.QUAD4, conn=np.asarray(conn), name="梁")
    return Mesh(coords=coords, blocks=(block,))


def build_cantilever_case(mesh: Mesh) -> StaticCase:
    """构建悬臂梁工况：左端固支，右边缘节点竖向载荷."""
    n_nodes = mesh.n_nodes
    fixed = tuple(Constraint(node=n, dofs=(0, 1)) for n in range(_BEAM_NY + 1))
    tip = tuple(NodalLoad(node=n, forces=(0.0, _BEAM_TIP_LOAD)) for n in range(n_nodes - (_BEAM_NY + 1), n_nodes))
    return StaticCase(constraints=fixed, loads=tip)


class FeaPage(QWidget):
    """FEA 分析页：左侧模型参数与求解控制，右侧变形云图."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化分析页并组装布局."""
        super().__init__(parent)
        self._executor: ProcessExecutor | None = None
        self._solution: StaticSolution | None = None
        self._modal_solution: ModalSolution | None = None
        self._bridge = _SolveBridge()

        self._build_ui()
        self._connect()
        self._render_initial()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        """组装左右分栏布局."""
        splitter = QSplitter(Qt.Horizontal)

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD)
        panel_layout.setSpacing(theme.SPACING_MD)
        panel_layout.addWidget(self._build_model_group())
        panel_layout.addWidget(self._build_params_group())
        panel_layout.addWidget(self._build_solve_group())
        panel_layout.addWidget(self._build_result_group())
        panel_layout.addStretch()
        panel.setMaximumWidth(320)
        splitter.addWidget(panel)

        self._plot = pg.PlotWidget(background=theme.current_palette().bg_app)
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setAspectLocked(True)
        self._plot.addLegend(offset=(12, 12))
        splitter.addWidget(self._plot)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 900])

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(splitter)

    def _build_model_group(self) -> QGroupBox:
        """构建模型选择组（含分析类型）."""
        box = QGroupBox("模型")
        form = QFormLayout(box)
        self._model_combo = QComboBox()
        self._model_combo.addItem("悬臂梁（Q4 平面应力）")
        form.addRow("示例", self._model_combo)
        self._analysis_combo = QComboBox()
        self._analysis_combo.addItem("静力", "static")
        self._analysis_combo.addItem("模态", "modal")
        form.addRow("分析类型", self._analysis_combo)
        self._model_info = QLabel("40 x 8 网格 · 369 节点 · 320 单元", objectName="secondaryText")
        form.addRow(self._model_info)
        return box

    def _build_params_group(self) -> QGroupBox:
        """构建材料/截面参数组."""
        box = QGroupBox("参数")
        form = QFormLayout(box)
        self._young_spin = self._make_spin(1.0e3, 1.0e12, 2.1e5, 1.0e5)
        self._poisson_spin = self._make_spin(0.0, 0.49, 0.3, 0.05)
        self._thickness_spin = self._make_spin(0.01, 100.0, 1.0, 0.1)
        self._density_spin = self._make_spin(1.0e-9, 1.0e5, 7.85, 0.1)
        self._n_modes_spin = QSpinBox()
        self._n_modes_spin.setRange(1, 50)
        self._n_modes_spin.setValue(6)
        form.addRow("弹性模量 E", self._young_spin)
        form.addRow("泊松比 ν", self._poisson_spin)
        form.addRow("厚度 t", self._thickness_spin)
        form.addRow("密度 ρ（模态）", self._density_spin)
        form.addRow("模态阶数（模态）", self._n_modes_spin)
        return box

    @staticmethod
    def _make_spin(minimum: float, maximum: float, value: float, step: float) -> QDoubleSpinBox:
        """构建数值输入框."""
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        return spin

    def _build_solve_group(self) -> QGroupBox:
        """构建求解控制组."""
        box = QGroupBox("求解")
        layout = QVBoxLayout(box)
        layout.setSpacing(theme.SPACING_SM)
        self._solve_button = QPushButton("求解")
        self._solve_button.setMinimumHeight(36)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._status_label = QLabel("就绪", objectName="secondaryText")
        layout.addWidget(self._solve_button)
        layout.addWidget(self._progress)
        layout.addWidget(self._status_label)
        return box

    def _build_result_group(self) -> QGroupBox:
        """构建结果摘要组（模态分析附频率表与振型切换）."""
        box = QGroupBox("结果")
        layout = QVBoxLayout(box)
        self._result_label = QLabel("尚未求解", objectName="resultText")
        self._result_label.setWordWrap(True)
        layout.addWidget(self._result_label)

        # 模态结果：频率表 + 振型序号切换
        self._freq_table = QTableWidget(objectName="freqTable")
        self._freq_table.setColumnCount(3)
        self._freq_table.setHorizontalHeaderLabels(("阶", "ω (rad/s)", "f (Hz)"))
        self._freq_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._freq_table.verticalHeader().setVisible(False)
        self._freq_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._freq_table.setVisible(False)
        layout.addWidget(self._freq_table)

        self._mode_spin = QSpinBox()
        self._mode_spin.setRange(1, 1)
        self._mode_spin.setVisible(False)
        layout.addWidget(self._mode_spin)
        return box

    def _connect(self) -> None:
        """连接信号槽."""
        self._solve_button.clicked.connect(self._on_solve)
        self._mode_spin.valueChanged.connect(self._render_mode_shape)
        self._bridge.progress.connect(self._on_progress)
        self._bridge.finished.connect(self._on_finished)
        self._bridge.failed.connect(self._on_failed)

    # ---------------------------------------------------------------- 渲染

    def _render_initial(self) -> None:
        """渲染未变形线框."""
        mesh = build_cantilever_mesh()
        edges = mesh_edges(mesh)
        segments = edge_segments(mesh.coords, edges)
        self._plot.plot(
            segments[:, :, 0].ravel(),
            segments[:, :, 1].ravel(),
            connect="pairs",
            pen=pg.mkPen(theme.current_palette().border, width=1),
            name="未变形",
        )

    def _render_solution(self, solution: StaticSolution) -> None:
        """渲染变形线框与节点位移模云图."""
        mesh = solution.mesh
        edges = mesh_edges(mesh)
        field = displacement_field(solution)
        scale = self._deform_scale(mesh, field)

        deformed = deformed_coords(mesh, solution.displacements, scale)
        segments = edge_segments(deformed, edges)
        self._plot.clear()
        self._plot.plot(
            segments[:, :, 0].ravel(),
            segments[:, :, 1].ravel(),
            connect="pairs",
            pen=pg.mkPen(theme.current_palette().primary, width=2),
            name=f"变形 (x{scale:.0f})",
        )
        colors = [pg.mkColor(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in scalar_colors(field)]
        self._plot.addItem(
            pg.ScatterPlotItem(
                x=deformed[:, 0],
                y=deformed[:, 1],
                size=4,
                brush=colors,
                pen=None,
                name="|u|",
            )
        )

    @staticmethod
    def _deform_scale(mesh: Mesh, field: np.ndarray) -> float:
        """变形放大系数：最大位移放大至模型尺度的 5%."""
        max_u = float(np.max(np.abs(field))) if field.size else 0.0
        if max_u < 1e-15:
            return 1.0
        span = float(np.ptp(mesh.coords, axis=0).max())
        return 0.05 * span / max_u

    def _render_modal(self, solution: ModalSolution) -> None:
        """填充频率表并渲染首阶振型云图."""
        self._freq_table.setRowCount(solution.n_modes)
        for i in range(solution.n_modes):
            self._freq_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self._freq_table.setItem(i, 1, QTableWidgetItem(f"{solution.frequencies[i]:.6g}"))
            self._freq_table.setItem(i, 2, QTableWidgetItem(f"{solution.frequencies_hz[i]:.6g}"))
        self._freq_table.setVisible(True)
        self._mode_spin.blockSignals(True)
        self._mode_spin.setRange(1, solution.n_modes)
        self._mode_spin.setValue(1)
        self._mode_spin.blockSignals(False)
        self._mode_spin.setVisible(True)
        self._result_label.setText(
            f"前 {solution.n_modes} 阶 · 基频 ω₁ = {solution.frequencies[0]:.6g} rad/s"
            f"（{solution.frequencies_hz[0]:.6g} Hz）"
        )
        self._render_mode_shape(1)

    def _render_mode_shape(self, index: int) -> None:
        """渲染第 index 阶（1 基）振型云图."""
        solution = self._modal_solution
        if solution is None or not 1 <= index <= solution.n_modes:
            return
        mesh = solution.mesh
        edges = mesh_edges(mesh)
        shape = solution.mode_shape(index - 1)
        field = np.linalg.norm(shape, axis=1)
        scale = self._deform_scale(mesh, field)

        deformed = deformed_coords(mesh, shape, scale)
        segments = edge_segments(deformed, edges)
        self._plot.clear()
        self._plot.plot(
            segments[:, :, 0].ravel(),
            segments[:, :, 1].ravel(),
            connect="pairs",
            pen=pg.mkPen(theme.current_palette().primary, width=2),
            name=f"第 {index} 阶振型 (x{scale:.0f})",
        )
        colors = [pg.mkColor(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in scalar_colors(field)]
        self._plot.addItem(
            pg.ScatterPlotItem(
                x=deformed[:, 0],
                y=deformed[:, 1],
                size=4,
                brush=colors,
                pen=None,
                name="|φ|",
            )
        )

    # ---------------------------------------------------------------- 求解

    @property
    def current_material(self) -> LinearElastic:
        """当前参数对应的线弹性材料（平面应力）."""
        return LinearElastic(
            e_modulus=self._young_spin.value(),
            poisson=self._poisson_spin.value(),
            density=self._density_spin.value(),
        )

    def _current_model(self) -> tuple[Mesh, list, list, StaticCase]:
        """按当前参数构建 (mesh, materials, sections, case)."""
        mesh = build_cantilever_mesh()
        materials = [self.current_material]
        sections = [Section(thickness=self._thickness_spin.value())]
        case = build_cantilever_case(mesh)
        return mesh, materials, sections, case

    def _on_solve(self) -> None:
        """按分析类型提交求解任务到进程执行器."""
        mesh, materials, sections, case = self._current_model()
        if self._executor is None:
            self._executor = ProcessExecutor()
        if self._analysis_combo.currentData() == "modal":
            spec = TaskSpec(
                target="zylab.fea.modal:solve_modal",
                args=(mesh, materials, sections, case.constraints),
                kwargs={"n_modes": self._n_modes_spin.value()},
            )
        else:
            spec = TaskSpec(
                target="zylab.fea.static:solve_static",
                args=(mesh, materials, sections, case),
            )
        self._solve_button.setEnabled(False)
        self._progress.setValue(0)
        self._status_label.setText("提交任务…")
        handle = self._executor.submit(spec)
        handle.add_listener(self._bridge.dispatch)

    def _on_progress(self, progress: float, message: str) -> None:
        """更新进度条与状态."""
        self._progress.setValue(int(progress * 100))
        self._status_label.setText(message)

    def _on_finished(self, solution: object) -> None:
        """按结果类型分发渲染（StaticSolution / ModalSolution）."""
        self._solve_button.setEnabled(True)
        self._status_label.setText("求解完成")
        self._set_result_error(False)
        if isinstance(solution, ModalSolution):
            self._solution = None
            self._modal_solution = solution
            self._render_modal(solution)
        else:
            self._solution = solution
            self._modal_solution = None
            self._freq_table.setVisible(False)
            self._mode_spin.setVisible(False)
            self._render_solution(solution)
            max_u = float(np.max(np.linalg.norm(solution.displacements, axis=1)))
            self._result_label.setText(f"最大位移 |u| = {max_u:.6g}\n应变能 = {solution.strain_energy:.6g}")

    def _on_failed(self, message: str) -> None:
        """显示求解失败信息."""
        self._solve_button.setEnabled(True)
        self._status_label.setText("求解失败")
        self._result_label.setText(message)
        self._set_result_error(True)

    def _set_result_error(self, error: bool) -> None:
        """切换结果摘要的错误着色（objectName 切换 + 重刷样式）."""
        self._result_label.setObjectName("errorText" if error else "resultText")
        style = self._result_label.style()
        style.unpolish(self._result_label)
        style.polish(self._result_label)

    def shutdown(self) -> None:
        """关闭后台执行器（主窗口关闭时调用）."""
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
