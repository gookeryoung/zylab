"""FEA 分析页：内置示例模型 + 参数编辑 + 进程隔离后台求解 + 云图可视化.

支持静力（变形云图）、模态（频率表 + 振型云图切换）与谐响应（频响曲线）三类分析。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg

from zylab.core.executor import EventKind, ProcessExecutor, TaskEvent, TaskSpec
from zylab.fea import (
    BucklingSolution,
    Constraint,
    ElementBlock,
    ElementType,
    HarmonicResponse,
    LinearElastic,
    Mesh,
    ModalSolution,
    NodalLoad,
    NonlinearSolution,
    Section,
    StaticCase,
)
from zylab.fea.viewdata import deformed_coords, edge_segments, mesh_edges, scalar_colors

from .. import theme
from ..qt_compat import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHeaderView,
    QLabel,
    QObject,
    QProgressBar,
    QPushButton,
    QScrollArea,
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


# 悬臂柱示例：竖直 BEAM2 梁，底部固支，顶部压缩载荷（屈曲基准）
_COLUMN_HEIGHT = 10.0
_COLUMN_N_ELEM = 20
_COLUMN_TIP_LOAD = -1.0  # 顶部单位压缩轴力（临界载荷 = 因子 × 1.0）


def build_column_mesh() -> Mesh:
    """构建竖直悬臂柱 BEAM2 梁网格（底部固支端为节点 0，顶端为最后节点）."""
    coords = np.array([[0.0, _COLUMN_HEIGHT * i / _COLUMN_N_ELEM] for i in range(_COLUMN_N_ELEM + 1)])
    conn = np.array([[i, i + 1] for i in range(_COLUMN_N_ELEM)], dtype=np.int64)
    block = ElementBlock(etype=ElementType.BEAM2, conn=conn, name="柱")
    return Mesh(coords=coords, blocks=(block,))


def build_column_case(mesh: Mesh) -> StaticCase:
    """构建悬臂柱工况：底部固支，顶部单位压缩轴力（沿 y 负向）."""
    top = mesh.n_nodes - 1
    return StaticCase(
        constraints=(Constraint(node=0, dofs=(0, 1, 2)),),
        loads=(NodalLoad(node=top, forces=(0.0, _COLUMN_TIP_LOAD, 0.0)),),
    )


# 两杆浅桁架示例：几何非线性大位移经典算例（顶点集中力下弦转角效应显著）
_TRUSS_HALF_SPAN = 5.0
_TRUSS_RISE = 0.5
_TRUSS_APEX_LOAD = -60.0  # 顶点竖向集中力（向下，接近极限点，非线性效应显著）


def build_truss_mesh() -> Mesh:
    """构建两杆浅桁架 TRUSS2 网格（顶点为节点 1）."""
    b, h = _TRUSS_HALF_SPAN, _TRUSS_RISE
    coords = np.array([[-b, 0.0], [0.0, h], [b, 0.0]])
    conn = np.array([[0, 1], [1, 2]], dtype=np.int64)
    block = ElementBlock(etype=ElementType.TRUSS2, conn=conn, name="桁架")
    return Mesh(coords=coords, blocks=(block,))


def build_truss_case(mesh: Mesh) -> StaticCase:
    """构建两杆桁架工况：两支座固支，顶点竖向集中力."""
    apex = mesh.n_nodes - 2  # 3 节点模型的中间顶点
    return StaticCase(
        constraints=(Constraint(node=0, dofs=(0, 1)), Constraint(node=2, dofs=(0, 1))),
        loads=(NodalLoad(node=apex, forces=(0.0, _TRUSS_APEX_LOAD)),),
    )


class FeaPage(QWidget):
    """FEA 分析页：左侧模型参数与求解控制，右侧变形云图."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化分析页并组装布局."""
        super().__init__(parent)
        self._executor: ProcessExecutor | None = None
        self._solution: StaticSolution | None = None
        self._modal_solution: ModalSolution | None = None
        self._harmonic_solution: HarmonicResponse | None = None
        self._buckling_solution: BucklingSolution | None = None
        self._nonlinear_solution: NonlinearSolution | None = None
        self._bridge = _SolveBridge()

        self._build_ui()
        self._connect()
        self._render_initial()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        """组装左右分栏布局（左面板入滚动区，纵向永不压缩）."""
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

        # 面板包滚动区：小窗口/高 DPI 下内容超高时出滚动条，
        # 而不是压缩表单行导致输入框变形（宽度区间由滚动区承载）
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(340)
        splitter.addWidget(scroll)
        splitter.setCollapsible(0, False)
        self._panel_scroll = scroll

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
        self._model_combo.addItem("悬臂柱（BEAM2 屈曲）")
        self._model_combo.addItem("两杆桁架（TRUSS2 非线性）")
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        form.addRow("示例", self._model_combo)
        self._analysis_combo = QComboBox()
        self._analysis_combo.addItem("静力", "static")
        self._analysis_combo.addItem("模态", "modal")
        self._analysis_combo.addItem("谐响应", "harmonic")
        self._analysis_combo.addItem("屈曲", "buckling")
        self._analysis_combo.addItem("几何非线性", "nonlinear")
        form.addRow("分析类型", self._analysis_combo)
        self._model_info = QLabel("40 x 8 网格 · 369 节点 · 320 单元", objectName="secondaryText")
        form.addRow(self._model_info)
        return box

    def _build_params_group(self) -> QGroupBox:
        """构建材料/截面参数组（行可见性按模型/分析类型联动）."""
        box = QGroupBox("参数")
        form = QFormLayout(box)
        self._params_form = form
        self._young_spin = self._make_spin(1.0e3, 1.0e12, 2.1e5, 1.0e5)
        self._poisson_spin = self._make_spin(0.0, 0.49, 0.3, 0.05)
        self._thickness_spin = self._make_spin(0.01, 100.0, 1.0, 0.1)
        self._density_spin = self._make_spin(1.0e-9, 1.0e5, 7.85, 0.1)
        self._n_modes_spin = QSpinBox()
        self._n_modes_spin.setRange(1, 50)
        self._n_modes_spin.setValue(6)
        self._fmax_spin = self._make_spin(1.0e-6, 1.0e6, 3.0, 0.5)
        self._n_freq_spin = QSpinBox()
        self._n_freq_spin.setRange(10, 2000)
        self._n_freq_spin.setValue(60)
        self._alpha_spin = self._make_spin(0.0, 1.0e6, 0.1, 0.05)
        self._beta_spin = self._make_spin(0.0, 1.0e3, 0.0, 0.01)
        self._increments_spin = QSpinBox()
        self._increments_spin.setRange(1, 100)
        self._increments_spin.setValue(10)
        form.addRow("弹性模量 E", self._young_spin)
        form.addRow("泊松比 ν", self._poisson_spin)
        form.addRow("厚度 t", self._thickness_spin)
        form.addRow("密度 ρ（动力学）", self._density_spin)
        form.addRow("模态阶数（模态）", self._n_modes_spin)
        form.addRow("扫频上限 ω（谐响应）", self._fmax_spin)
        form.addRow("扫频点数（谐响应）", self._n_freq_spin)
        form.addRow("阻尼 α（谐响应）", self._alpha_spin)
        form.addRow("阻尼 β（谐响应）", self._beta_spin)
        form.addRow("增量步数（非线性）", self._increments_spin)
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
        self._analysis_combo.currentIndexChanged.connect(self._update_param_visibility)
        self._update_param_visibility()
        self._mode_spin.valueChanged.connect(self._render_mode_shape)
        self._bridge.progress.connect(self._on_progress)
        self._bridge.finished.connect(self._on_finished)
        self._bridge.failed.connect(self._on_failed)

    # ---------------------------------------------------------------- 渲染

    def _render_initial(self) -> None:
        """渲染当前模型的未变形线框."""
        mesh, _materials, _sections, _case = self._current_model()
        edges = mesh_edges(mesh)
        segments = edge_segments(mesh.coords, edges)
        self._plot.clear()
        self._plot.plot(
            segments[:, :, 0].ravel(),
            segments[:, :, 1].ravel(),
            connect="pairs",
            pen=pg.mkPen(theme.current_palette().border, width=1),
            name="未变形",
        )

    def _render_solution(self, solution: StaticSolution) -> None:
        """渲染变形线框与节点位移模云图."""
        self._render_deformation(solution.mesh, solution.displacements)

    def _render_deformation(self, mesh: Mesh, displacements: np.ndarray) -> None:
        """渲染大位移变形云图（静力/几何非线性共用）."""
        self._restore_mesh_view()
        edges = mesh_edges(mesh)
        field = np.linalg.norm(displacements, axis=1)
        scale = self._deform_scale(mesh, field)

        deformed = deformed_coords(mesh, displacements, scale)
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
        self._freq_table.setHorizontalHeaderLabels(("阶", "ω (rad/s)", "f (Hz)"))
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
        """渲染第 index 阶（1 基）振型/屈曲模态云图（数据源：模态或屈曲解）."""
        self._restore_mesh_view()
        shape: np.ndarray | None = None
        label = ""
        mesh: Mesh | None = None
        if self._modal_solution is not None and 1 <= index <= self._modal_solution.n_modes:
            shape = self._modal_solution.mode_shape(index - 1)
            label = f"第 {index} 阶振型"
            mesh = self._modal_solution.mesh
        elif self._buckling_solution is not None and 1 <= index <= self._buckling_solution.n_modes:
            shape = self._buckling_solution.mode_shape(index - 1)
            label = f"第 {index} 阶屈曲"
            mesh = self._buckling_solution.mesh
        if shape is None or mesh is None:
            return
        edges = mesh_edges(mesh)
        translation = shape[:, :2]  # 梁含转角分量，仅取平动
        field = np.linalg.norm(translation, axis=1)
        scale = self._deform_scale(mesh, field)

        deformed = deformed_coords(mesh, translation, scale)
        segments = edge_segments(deformed, edges)
        self._plot.clear()
        self._plot.plot(
            segments[:, :, 0].ravel(),
            segments[:, :, 1].ravel(),
            connect="pairs",
            pen=pg.mkPen(theme.current_palette().primary, width=2),
            name=f"{label} (x{scale:.0f})",
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

    def _render_buckling(self, solution: BucklingSolution) -> None:
        """填充载荷因子表并渲染首阶屈曲模态."""
        self._freq_table.setHorizontalHeaderLabels(("阶", "载荷因子 λ", "临界载荷"))
        self._freq_table.setRowCount(solution.n_modes)
        reference = abs(_COLUMN_TIP_LOAD)
        for i in range(solution.n_modes):
            factor = float(solution.load_factors[i])
            self._freq_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self._freq_table.setItem(i, 1, QTableWidgetItem(f"{factor:.6g}"))
            self._freq_table.setItem(i, 2, QTableWidgetItem(f"{factor * reference:.6g}"))
        self._freq_table.setVisible(True)
        self._mode_spin.blockSignals(True)
        self._mode_spin.setRange(1, solution.n_modes)
        self._mode_spin.setValue(1)
        self._mode_spin.blockSignals(False)
        self._mode_spin.setVisible(True)
        self._result_label.setText(
            f"前 {solution.n_modes} 阶 · 一阶载荷因子 λ₁ = {solution.load_factors[0]:.6g}"
            f"（临界载荷 = λ × 参考载荷 {reference:.6g}）"
        )
        self._render_mode_shape(1)

    def _restore_mesh_view(self) -> None:
        """恢复云图视图状态（锁纵横比 + 线性轴，清除频响曲线的视图残留）."""
        self._plot.setAspectLocked(True)
        self._plot.setLogMode(y=False)

    @staticmethod
    def _tip_node(mesh: Mesh) -> int:
        """取末端中点节点（x 最大列中 y 居中者）作为频响观察点."""
        coords = mesh.coords
        tip_mask = coords[:, 0] >= coords[:, 0].max() - 1e-9
        tip_rows = np.flatnonzero(tip_mask)
        return int(tip_rows[np.argmin(np.abs(coords[tip_rows, 1] - np.mean(coords[tip_rows, 1])))])

    def _render_harmonic(self, solution: HarmonicResponse) -> None:
        """渲染频响曲线（观察点 |uy| 随 ω 变化，对数幅值轴）并标注峰值."""
        mesh = solution.mesh
        node = self._tip_node(mesh)
        dof_y = node * mesh.dofs_per_node + 1  # 观察点竖向分量
        amplitude = np.abs(solution.displacements[dof_y, :])
        omegas = solution.frequencies

        self._plot.clear()
        self._plot.setAspectLocked(False)  # 频响曲线恢复自由纵横比
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setLogMode(y=True)
        self._plot.addLegend(offset=(12, 12))
        self._plot.setLabel("bottom", "ω", units="rad/s")
        self._plot.setLabel("left", "|uy|（对数）")
        self._plot.plot(
            omegas,
            amplitude,
            pen=pg.mkPen(theme.current_palette().primary, width=2),
            name=f"节点 {node} |uy|",
        )
        peak_index = int(np.argmax(amplitude))
        self._plot.addItem(
            pg.ScatterPlotItem(
                x=[omegas[peak_index]],
                y=[amplitude[peak_index]],
                size=10,
                brush=pg.mkBrush(theme.current_palette().error_text),
                pen=None,
                name="峰值",
            )
        )
        self._result_label.setText(
            f"峰值 |uy| = {amplitude[peak_index]:.6g} @ ω = {omegas[peak_index]:.6g} rad/s"
            f"（{solution.n_frequencies} 个频率点）"
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
        """按当前模型选择构建 (mesh, materials, sections, case)."""
        index = self._model_combo.currentIndex()
        if index == 1:
            mesh = build_column_mesh()
            materials = [self.current_material]
            sections = [Section(area=1.0, inertia=1.0e-4)]
            case = build_column_case(mesh)
            return mesh, materials, sections, case
        if index == 2:
            mesh = build_truss_mesh()
            materials = [self.current_material]
            sections = [Section(area=1.0)]
            case = build_truss_case(mesh)
            return mesh, materials, sections, case
        mesh = build_cantilever_mesh()
        materials = [self.current_material]
        sections = [Section(thickness=self._thickness_spin.value())]
        case = build_cantilever_case(mesh)
        return mesh, materials, sections, case

    def _on_model_changed(self, index: int) -> None:
        """切换示例模型：更新信息标签、参数行可见性并重绘初始线框."""
        if index == 1:
            self._model_info.setText(f"BEAM2 梁 · {_COLUMN_N_ELEM + 1} 节点 · {_COLUMN_N_ELEM} 单元 · 顶部压缩")
        elif index == 2:
            self._model_info.setText("TRUSS2 两杆浅桁架 · 3 节点 · 2 单元 · 顶点集中力")
        else:
            self._model_info.setText("40 x 8 网格 · 369 节点 · 320 单元")
        self._update_param_visibility()
        self._render_initial()

    def _update_param_visibility(self) -> None:
        """按模型/分析类型关联显示参数行（无关参数隐藏，压缩面板高度）."""
        analysis = self._analysis_combo.currentData()
        continuum = self._model_combo.currentIndex() == 0
        dynamic = analysis in ("modal", "harmonic")
        rules = {
            self._poisson_spin: continuum,  # 泊松比仅连续体用到
            self._thickness_spin: continuum,  # 厚度仅 Q4 平面应力
            self._density_spin: dynamic,  # 密度仅动力学分析
            self._n_modes_spin: analysis in ("modal", "buckling"),
            self._fmax_spin: analysis == "harmonic",
            self._n_freq_spin: analysis == "harmonic",
            self._alpha_spin: analysis == "harmonic",
            self._beta_spin: analysis == "harmonic",
            self._increments_spin: analysis == "nonlinear",
        }
        for field, visible in rules.items():
            label = self._params_form.labelForField(field)
            if label is not None:
                label.setVisible(visible)
            field.setVisible(visible)

    def _on_solve(self) -> None:
        """按分析类型提交求解任务到进程执行器."""
        mesh, materials, sections, case = self._current_model()
        if self._executor is None:
            self._executor = ProcessExecutor()
        analysis = self._analysis_combo.currentData()
        if analysis == "modal":
            spec = TaskSpec(
                target="zylab.fea.modal:solve_modal",
                args=(mesh, materials, sections, case.constraints),
                kwargs={"n_modes": self._n_modes_spin.value()},
            )
        elif analysis == "harmonic":
            spec = TaskSpec(
                target="zylab.fea.harmonic:solve_harmonic",
                args=(mesh, materials, sections, case, self._sweep_frequencies()),
                kwargs={"alpha": self._alpha_spin.value(), "beta": self._beta_spin.value()},
            )
        elif analysis == "buckling":
            spec = TaskSpec(
                target="zylab.fea.buckling:solve_buckling",
                args=(mesh, materials, sections, case),
                kwargs={"n_modes": self._n_modes_spin.value()},
            )
        elif analysis == "nonlinear":
            spec = TaskSpec(
                target="zylab.fea.nonlinear:solve_nonlinear_static",
                args=(mesh, materials, sections, case),
                kwargs={"n_increments": self._increments_spin.value()},
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

    def _sweep_frequencies(self) -> np.ndarray:
        """构造谐响应频率扫描序列（0 到扫频上限，等间距）."""
        n_points = self._n_freq_spin.value()
        return np.linspace(0.0, self._fmax_spin.value(), n_points)

    def _on_progress(self, progress: float, message: str) -> None:
        """更新进度条与状态."""
        self._progress.setValue(int(progress * 100))
        self._status_label.setText(message)

    def _on_finished(self, solution: object) -> None:
        """按结果类型分发渲染（静力/模态/谐响应/屈曲/非线性）."""
        self._solve_button.setEnabled(True)
        self._status_label.setText("求解完成")
        self._set_result_error(False)
        self._nonlinear_solution = None
        if isinstance(solution, ModalSolution):
            self._solution = None
            self._modal_solution = solution
            self._harmonic_solution = None
            self._buckling_solution = None
            self._render_modal(solution)
        elif isinstance(solution, HarmonicResponse):
            self._solution = None
            self._modal_solution = None
            self._harmonic_solution = solution
            self._buckling_solution = None
            self._freq_table.setVisible(False)
            self._mode_spin.setVisible(False)
            self._render_harmonic(solution)
        elif isinstance(solution, BucklingSolution):
            self._solution = None
            self._modal_solution = None
            self._harmonic_solution = None
            self._buckling_solution = solution
            self._render_buckling(solution)
        elif isinstance(solution, NonlinearSolution):
            self._solution = None
            self._modal_solution = None
            self._harmonic_solution = None
            self._buckling_solution = None
            self._nonlinear_solution = solution
            self._freq_table.setVisible(False)
            self._mode_spin.setVisible(False)
            self._render_nonlinear(solution)
        else:
            self._solution = solution
            self._modal_solution = None
            self._harmonic_solution = None
            self._buckling_solution = None
            self._freq_table.setVisible(False)
            self._mode_spin.setVisible(False)
            self._render_solution(solution)
            max_u = float(np.max(np.linalg.norm(solution.displacements, axis=1)))
            self._result_label.setText(f"最大位移 |u| = {max_u:.6g}\n应变能 = {solution.strain_energy:.6g}")

    def _render_nonlinear(self, solution: NonlinearSolution) -> None:
        """渲染非线性收敛态变形云图与迭代历程摘要."""
        self._render_deformation(solution.mesh, solution.displacements)
        max_u = float(np.max(np.linalg.norm(solution.displacements, axis=1)))
        steps = len(solution.iterations)
        self._result_label.setText(
            f"最大位移 |u| = {max_u:.6g}\n收敛：{steps} 增量步 · {solution.total_iterations} 次 Newton 迭代"
        )

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
