"""结果视图：摘要 + 控制条（频率表/振型序号/非线性与瞬态视图切换）+ 绘图区.

按结果类型分发渲染（静力/非线性变形云图、模态/屈曲振型、谐响应频响曲线、
瞬态末帧云图与位移时程、模型网格预览），从原 FeaPage 移植并组件化，
供工作台页复用。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg

from zylab.fea import (
    BucklingSolution,
    HarmonicResponse,
    Mesh,
    ModalSolution,
    NonlinearSolution,
    StaticSolution,
    TransientSolution,
    export_csv,
)
from zylab.fea.viewdata import deformed_coords, edge_segments, mesh_edges, scalar_colors
from zylab.studio import ModelBundle
from zylab.studio.nodes import tip_node

from .. import theme
from ..qt_compat import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

__all__ = ["ResultView"]


class ResultView(QWidget):
    """结果视图控件（pyqtgraph 绘图 + 类型关联的控制条）."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化结果视图."""
        super().__init__(parent)
        self._modal: ModalSolution | None = None
        self._buckling: BucklingSolution | None = None
        self._nonlinear: NonlinearSolution | None = None
        self._transient: TransientSolution | None = None
        self._reference_load = 1.0
        self._solution: object | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACING_SM)

        self._summary = QLabel("尚未求解", objectName="resultText")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._export_row = QWidget()
        export_layout = QHBoxLayout(self._export_row)
        export_layout.setContentsMargins(0, 0, 0, 0)
        self._export_csv_btn = QPushButton("导出 CSV")
        self._export_csv_btn.clicked.connect(self._on_export_csv)
        export_layout.addWidget(self._export_csv_btn)
        self._export_png_btn = QPushButton("导出 PNG")
        self._export_png_btn.clicked.connect(self._on_export_png)
        export_layout.addWidget(self._export_png_btn)
        export_layout.addStretch(1)
        self._export_row.setVisible(False)
        layout.addWidget(self._export_row)

        self._freq_table = QTableWidget(objectName="freqTable")
        self._freq_table.setColumnCount(3)
        self._freq_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._freq_table.verticalHeader().setVisible(False)
        self._freq_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._freq_table.setVisible(False)
        self._freq_table.setMaximumHeight(160)
        layout.addWidget(self._freq_table)

        self._mode_spin = QSpinBox()
        self._mode_spin.setRange(1, 1)
        self._mode_spin.setPrefix("振型 ")
        self._mode_spin.setVisible(False)
        self._mode_spin.valueChanged.connect(self._render_mode_shape)
        layout.addWidget(self._mode_spin)

        self._view_combo = QComboBox(objectName="nonlinearView")
        self._view_combo.addItem("变形云图", "deform")
        self._view_combo.addItem("载荷-位移曲线", "curve")
        self._view_combo.setVisible(False)
        self._view_combo.currentIndexChanged.connect(self._on_view_changed)
        layout.addWidget(self._view_combo)

        self._plot = pg.PlotWidget(background=theme.current_palette().bg_app)
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setAspectLocked(True)
        self._plot.addLegend(offset=(12, 12))
        layout.addWidget(self._plot, stretch=1)

    # ------------------------------------------------------------------ 公共接口

    def show_mesh(self, bundle: ModelBundle) -> None:
        """渲染模型网格预览（未变形线框 + 规模摘要）."""
        self._reset_controls()
        mesh = bundle.mesh
        self._restore_mesh_view()
        edges = mesh_edges(mesh)
        segments = edge_segments(mesh.coords, edges)
        self._plot.clear()
        self._plot.plot(
            segments[:, :, 0].ravel(),
            segments[:, :, 1].ravel(),
            connect="pairs",
            pen=pg.mkPen(theme.current_palette().border_strong, width=2),
            name="未变形",
        )
        self._summary.setText(f"模型预览 · {mesh.n_nodes} 节点 · {mesh.n_elements} 单元")
        self._set_error(False)

    def show_solution(self, solution: object, reference_load: float = 1.0) -> None:
        """按解类型分发渲染（静力/模态/谐响应/瞬态/屈曲/几何非线性）."""
        self._reset_controls()
        self._set_error(False)
        if isinstance(solution, ModalSolution):
            self._modal = solution
            self._render_modal(solution)
        elif isinstance(solution, HarmonicResponse):
            self._render_harmonic(solution)
        elif isinstance(solution, TransientSolution):
            self._transient = solution
            self._render_transient(solution)
        elif isinstance(solution, BucklingSolution):
            self._buckling = solution
            self._reference_load = reference_load
            self._render_buckling(solution)
        elif isinstance(solution, NonlinearSolution):
            self._nonlinear = solution
            self._render_nonlinear(solution)
        elif isinstance(solution, StaticSolution):
            self._render_deformation(solution.mesh, solution.displacements)
            max_u = float(np.max(np.linalg.norm(solution.displacements, axis=1)))
            self._summary.setText(f"最大位移 |u| = {max_u:.6g}\n应变能 = {solution.strain_energy:.6g}")
        else:
            self._summary.setText(f"未知结果类型: {type(solution).__name__}")
            return
        self._solution = solution
        self._export_row.setVisible(True)

    def show_error(self, message: str) -> None:
        """显示失败信息."""
        self._reset_controls()
        self._summary.setText(message)
        self._set_error(True)

    def clear(self, message: str = "尚未求解") -> None:
        """清空结果（模板切换时）."""
        self._reset_controls()
        self._plot.clear()
        self._summary.setText(message)
        self._set_error(False)

    def refresh_theme(self) -> None:
        """主题切换后重刷绘图背景."""
        self._plot.setBackground(theme.current_palette().bg_app)

    # ------------------------------------------------------------------ 内部

    def _reset_controls(self) -> None:
        """隐藏类型关联控件并清空类型化解引用."""
        self._modal = None
        self._buckling = None
        self._nonlinear = None
        self._transient = None
        self._solution = None
        self._freq_table.setVisible(False)
        self._mode_spin.setVisible(False)
        self._view_combo.setVisible(False)
        self._export_row.setVisible(False)

    def _on_export_csv(self) -> None:
        """导出当前结果为 CSV（对话框选路径；失败信息显示在摘要）."""
        if self._solution is None:
            return
        path_str, _ = QFileDialog.getSaveFileName(self, "导出 CSV", "results.csv", "CSV 文件 (*.csv)")
        if not path_str:
            return
        try:
            export_csv(self._solution, Path(path_str))
        except (ValueError, OSError) as exc:
            self._summary.setText(f"导出失败: {exc}")
            return
        self._summary.setText(f"结果已导出: {Path(path_str).name}")

    def _on_export_png(self) -> None:
        """导出当前绘图区为 PNG 截图."""
        path_str, _ = QFileDialog.getSaveFileName(self, "导出 PNG", "result.png", "PNG 图片 (*.png)")
        if not path_str:
            return
        pixmap = self._plot.grab()
        if not pixmap.save(path_str):
            self._summary.setText("导出失败: 无法写入图片文件")
            return
        self._summary.setText(f"图片已导出: {Path(path_str).name}")

    def _set_error(self, error: bool) -> None:
        """切换摘要的错误着色（objectName 切换 + 重刷样式）."""
        self._summary.setObjectName("errorText" if error else "resultText")
        style = self._summary.style()
        style.unpolish(self._summary)
        style.polish(self._summary)

    def _restore_mesh_view(self) -> None:
        """恢复云图视图状态（锁纵横比 + 线性轴，清除曲线的视图残留）."""
        self._plot.setAspectLocked(True)
        self._plot.setLogMode(y=False)

    @staticmethod
    def _deform_scale(mesh: Mesh, field: np.ndarray) -> float:
        """变形放大系数：最大位移放大至模型尺度的 5%."""
        max_u = float(np.max(np.abs(field))) if field.size else 0.0
        if max_u < 1e-15:
            return 1.0
        span = float(np.ptp(mesh.coords, axis=0).max())
        return 0.05 * span / max_u

    def _render_deformation(self, mesh: Mesh, displacements: np.ndarray) -> None:
        """渲染变形线框与节点位移模云图（静力/几何非线性共用）."""
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
            pen=pg.mkPen(theme.current_palette().primary, width=3),
            name=f"变形 (x{scale:.0f})",
        )
        colors = [pg.mkColor(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in scalar_colors(field)]
        self._plot.addItem(pg.ScatterPlotItem(x=deformed[:, 0], y=deformed[:, 1], size=6, brush=colors, pen=None))

    def _render_modal(self, solution: ModalSolution) -> None:
        """填充频率表并渲染首阶振型云图."""
        self._fill_freq_table(solution)
        self._summary.setText(
            f"前 {solution.n_modes} 阶 · 基频 ω₁ = {solution.frequencies[0]:.6g} rad/s"
            f"（{solution.frequencies_hz[0]:.6g} Hz）"
        )
        self._render_mode_shape(1)

    def _fill_freq_table(self, solution: ModalSolution) -> None:
        """频率表 + 振型序号控件（模态专用）."""
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

    def _render_buckling(self, solution: BucklingSolution) -> None:
        """填充载荷因子表并渲染首阶屈曲模态."""
        self._freq_table.setHorizontalHeaderLabels(("阶", "载荷因子 λ", "临界载荷"))
        self._freq_table.setRowCount(solution.n_modes)
        for i in range(solution.n_modes):
            factor = float(solution.load_factors[i])
            self._freq_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self._freq_table.setItem(i, 1, QTableWidgetItem(f"{factor:.6g}"))
            self._freq_table.setItem(i, 2, QTableWidgetItem(f"{factor * self._reference_load:.6g}"))
        self._freq_table.setVisible(True)
        self._mode_spin.blockSignals(True)
        self._mode_spin.setRange(1, solution.n_modes)
        self._mode_spin.setValue(1)
        self._mode_spin.blockSignals(False)
        self._mode_spin.setVisible(True)
        self._summary.setText(
            f"前 {solution.n_modes} 阶 · 一阶载荷因子 λ₁ = {solution.load_factors[0]:.6g}"
            f"（临界载荷 = λ × 参考载荷 {self._reference_load:.6g}）"
        )
        self._render_mode_shape(1)

    def _render_mode_shape(self, index: int) -> None:
        """渲染第 index 阶（1 基）振型/屈曲模态云图."""
        shape: np.ndarray | None = None
        label = ""
        mesh: Mesh | None = None
        if self._modal is not None and 1 <= index <= self._modal.n_modes:
            shape = self._modal.mode_shape(index - 1)
            label = f"第 {index} 阶振型"
            mesh = self._modal.mesh
        elif self._buckling is not None and 1 <= index <= self._buckling.n_modes:
            shape = self._buckling.mode_shape(index - 1)
            label = f"第 {index} 阶屈曲"
            mesh = self._buckling.mesh
        if shape is None or mesh is None:
            return
        self._restore_mesh_view()
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
            pen=pg.mkPen(theme.current_palette().primary, width=3),
            name=f"{label} (x{scale:.0f})",
        )
        colors = [pg.mkColor(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in scalar_colors(field)]
        self._plot.addItem(pg.ScatterPlotItem(x=deformed[:, 0], y=deformed[:, 1], size=6, brush=colors, pen=None))

    def _render_harmonic(self, solution: HarmonicResponse) -> None:
        """渲染频响曲线（末端观察点 |uy| 随 ω 变化，对数幅值轴）并标注峰值."""
        mesh = solution.mesh
        node = tip_node(mesh)
        dof_y = node * mesh.dofs_per_node + 1  # 观察点竖向分量
        amplitude = np.abs(solution.displacements[dof_y, :])
        omegas = solution.frequencies

        self._plot.clear()
        self._plot.setAspectLocked(False)
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setLogMode(y=True)
        self._plot.setLabel("bottom", "ω", units="rad/s")
        self._plot.setLabel("left", "|uy|（对数）")
        self._plot.plot(
            omegas,
            amplitude,
            pen=pg.mkPen(theme.current_palette().primary, width=3),
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
        self._summary.setText(
            f"峰值 |uy| = {amplitude[peak_index]:.6g} @ ω = {omegas[peak_index]:.6g} rad/s"
            f"（{solution.n_frequencies} 个频率点）"
        )

    def _render_nonlinear(self, solution: NonlinearSolution) -> None:
        """渲染非线性结果：显示视图切换并默认变形云图."""
        self._view_combo.setItemText(0, "变形云图")
        self._view_combo.setItemText(1, "载荷-位移曲线")
        self._view_combo.blockSignals(True)
        self._view_combo.setCurrentIndex(0)
        self._view_combo.blockSignals(False)
        self._view_combo.setVisible(True)
        self._render_nonlinear_deform(solution)

    def _render_nonlinear_deform(self, solution: NonlinearSolution) -> None:
        """渲染非线性收敛态变形云图与迭代历程摘要."""
        self._render_deformation(solution.mesh, solution.displacements)
        max_u = float(np.max(np.linalg.norm(solution.displacements, axis=1)))
        self._summary.setText(
            f"最大位移 |u| = {max_u:.6g}\n收敛：{len(solution.iterations)} 增量步 · {solution.total_iterations} 次 Newton 迭代"
        )

    def _render_nonlinear_curve(self, solution: NonlinearSolution) -> None:
        """渲染载荷-位移曲线（最大位移节点 uy vs 载荷因子，含线性参照虚线）."""
        node = int(np.argmax(np.linalg.norm(solution.displacements, axis=1)))
        uy = solution.history_dof(node, 1)
        factors = solution.history_factors

        self._plot.clear()
        self._plot.setAspectLocked(False)
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setLogMode(y=False)
        self._plot.setLabel("bottom", "载荷因子 λ")
        self._plot.setLabel("left", f"节点 {node} uy")
        slope = uy[1] / factors[1]
        self._plot.plot(
            factors,
            slope * factors,
            pen=pg.mkPen(theme.current_palette().text_secondary, width=2, style=pg.QtCore.Qt.DashLine),
            name="线性参照",
        )
        self._plot.plot(factors, uy, pen=pg.mkPen(theme.current_palette().primary, width=3), name="非线性")
        self._summary.setText(
            f"载荷-位移曲线：{len(factors) - 1} 增量步\n"
            f"观察点节点 {node} uy = {uy[-1]:.6g}（线性 {slope * factors[-1]:.6g}）"
        )

    def _on_view_changed(self, index: int) -> None:
        """切换结果视图（非线性：变形云图/载荷-位移曲线；瞬态：末帧云图/位移时程）."""
        curve = self._view_combo.itemData(index) == "curve"
        if self._nonlinear is not None:
            if curve:
                self._render_nonlinear_curve(self._nonlinear)
            else:
                self._render_nonlinear_deform(self._nonlinear)
        elif self._transient is not None:
            if curve:
                self._render_transient_curve(self._transient)
            else:
                self._render_transient_deform(self._transient)

    def _render_transient(self, solution: TransientSolution) -> None:
        """渲染瞬态结果：显示视图切换并默认末帧变形云图."""
        self._view_combo.setItemText(0, "变形云图（末帧）")
        self._view_combo.setItemText(1, "位移时程曲线")
        self._view_combo.blockSignals(True)
        self._view_combo.setCurrentIndex(0)
        self._view_combo.blockSignals(False)
        self._view_combo.setVisible(True)
        self._render_transient_deform(solution)

    def _render_transient_deform(self, solution: TransientSolution) -> None:
        """渲染瞬态末帧变形云图与时程摘要."""
        mesh = solution.mesh
        final = solution.displacements[:, -1].reshape(mesh.n_nodes, mesh.dofs_per_node)
        self._render_deformation(mesh, final)
        max_u = float(np.max(np.abs(final)))
        self._summary.setText(
            f"末帧最大位移 |u| = {max_u:.6g}\n时程：{solution.n_steps} 步 · dt = {solution.dt:.4g}"
            f" · 总时长 = {solution.times[-1]:.6g}"
        )

    def _render_transient_curve(self, solution: TransientSolution) -> None:
        """渲染位移时程曲线（末端观察点 uy 随时间变化）并标注峰值."""
        mesh = solution.mesh
        node = tip_node(mesh)
        uy = solution.node_history(node, 1)
        times = solution.times

        self._plot.clear()
        self._plot.setAspectLocked(False)
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setLogMode(y=False)
        self._plot.setLabel("bottom", "时间 t")
        self._plot.setLabel("left", f"节点 {node} uy")
        self._plot.plot(
            times,
            uy,
            pen=pg.mkPen(theme.current_palette().primary, width=3),
            name=f"节点 {node} uy",
        )
        peak_index = int(np.argmax(np.abs(uy)))
        self._plot.addItem(
            pg.ScatterPlotItem(
                x=[times[peak_index]],
                y=[uy[peak_index]],
                size=10,
                brush=pg.mkBrush(theme.current_palette().error_text),
                pen=None,
                name="峰值",
            )
        )
        self._summary.setText(
            f"峰值 |uy| = {abs(uy[peak_index]):.6g} @ t = {times[peak_index]:.6g}"
            f"（{solution.n_steps} 步 · dt = {solution.dt:.4g}）"
        )
