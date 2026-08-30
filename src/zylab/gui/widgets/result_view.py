"""结果视图：多 TAB 容器 + 摘要/控制条/绘图区（Workbench 风格）.

- :class:`ResultTabs`：每个环节节点的结果独立一页（页名 = 节点名），
  可单独关闭，全部关闭/模板切换回到占位页（紧凑多结果同屏对比）；
- :class:`ResultView`：单页结果视图，按结果类型分发渲染（静力/非线性
  变形云图、模态/屈曲振型、谐响应频响曲线、瞬态末帧云图与位移时程、
  模型网格预览），从原 FeaPage 移植并组件化。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg

from zylab.fea import (
    BucklingSolution,
    ElectroThermalSolution,
    HarmonicResponse,
    Mesh,
    ModalSolution,
    NonlinearSolution,
    StaticSolution,
    TransientSolution,
    export_csv,
)
from zylab.fea.viewdata import (
    cmap_keys,
    cmap_label,
    cmap_lut,
    deformed_coords,
    edge_segments,
    mesh_edges,
    scalar_colors,
)
from zylab.studio import ConductionBundle, ModelBundle
from zylab.studio.nodes import tip_node

from .. import theme
from ..qt_compat import (
    QColor,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPainter,
    QPushButton,
    QSpinBox,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

__all__ = ["ColorBarWidget", "ResultTabs", "ResultView"]


class ColorBarWidget(QWidget):
    """云图标尺（自绘）：竖向色带 + 最大/中值/最小刻度.

    - 顶部为最大值色（与云图一致：标量归一化后经色带采样）；
    - 无场量时隐藏（clear），布局自动收回空间；
    - 刻度文字颜色随主题（refresh_theme 触发重绘）。
    """

    _BAR_W = 14  # 色带条宽度（像素）
    _PAD = 4  # 内边距

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化标尺（初始无场量，隐藏）."""
        super().__init__(parent)
        self._vmin = 0.0
        self._vmax = 1.0
        self._lut = cmap_lut("jet")
        self.setFixedWidth(self._BAR_W + 52)
        self.setVisible(False)

    def set_field(self, values: np.ndarray, cmap: str) -> None:
        """绑定标量场与色带（云图渲染时同步调用）."""
        data = np.asarray(values, dtype=float)
        if data.size == 0:
            self.clear()
            return
        self._vmin = float(np.min(data))
        self._vmax = float(np.max(data))
        self._lut = cmap_lut(cmap)
        self.setVisible(True)
        self.update()

    def clear(self) -> None:
        """清除场量（隐藏标尺）."""
        self.setVisible(False)

    def refresh_theme(self) -> None:
        """主题切换后重绘刻度文字."""
        self.update()

    def paintEvent(self, event) -> None:  # Qt 命名约定
        """绘制竖向色带（64 段）与三档刻度值."""
        del event
        pal = theme.current_palette()
        painter = QPainter(self)
        painter.setPen(QColor(pal.border))
        bar_x = self._PAD
        bar_h = self.height() - 2 * self._PAD
        if bar_h <= 0:
            return
        # 逐段填色：第 0 段在底部（最小值）
        n = self._lut.shape[0]
        seg_h = bar_h / n
        for i in range(n):
            r, g, b = self._lut[i]
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(int(r * 255), int(g * 255), int(b * 255)))
            y_top = self._PAD + bar_h - (i + 1) * seg_h
            painter.drawRect(int(bar_x), round(y_top), self._BAR_W, int(seg_h) + 1)
        painter.setPen(QColor(pal.border))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(int(bar_x), self._PAD, self._BAR_W, int(bar_h))
        # 刻度：最大（顶）/中/最小（底）
        painter.setPen(QColor(pal.text_secondary))
        painter.drawText(
            bar_x + self._BAR_W + self._PAD,
            self._PAD,
            self.width() - bar_x - self._BAR_W - self._PAD,
            14,
            0,
            f"{self._vmax:.4g}",
        )
        painter.drawText(
            bar_x + self._BAR_W + self._PAD,
            int(self.height() / 2 - 7),
            self.width() - bar_x - self._BAR_W - self._PAD,
            14,
            0,
            f"{(self._vmin + self._vmax) / 2.0:.4g}",
        )
        painter.drawText(
            bar_x + self._BAR_W + self._PAD,
            self.height() - self._PAD - 14,
            self.width() - bar_x - self._BAR_W - self._PAD,
            14,
            0,
            f"{self._vmin:.4g}",
        )


class ResultView(QWidget):
    """结果视图控件（pyqtgraph 绘图 + 类型关联的控制条）."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化结果视图."""
        super().__init__(parent)
        self._modal: ModalSolution | None = None
        self._buckling: BucklingSolution | None = None
        self._nonlinear: NonlinearSolution | None = None
        self._transient: TransientSolution | None = None
        self._electrothermal: ElectroThermalSolution | None = None
        self._reference_load = 1.0
        self._solution: object | None = None
        self._cmap = "jet"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACING_SM)

        self._summary = QLabel("尚未求解", objectName="resultText")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._export_row = QWidget()
        export_layout = QHBoxLayout(self._export_row)
        export_layout.setContentsMargins(0, 0, 0, 0)
        export_layout.addWidget(QLabel("色带"))
        self._cmap_combo = QComboBox(objectName="cmapCombo")
        for key in cmap_keys():
            self._cmap_combo.addItem(cmap_label(key), key)
        self._cmap_combo.currentIndexChanged.connect(self._on_cmap_changed)
        export_layout.addWidget(self._cmap_combo)
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
        plot_row = QWidget()
        plot_layout = QHBoxLayout(plot_row)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(theme.SPACING_SM)
        plot_layout.addWidget(self._plot, stretch=1)
        self._colorbar = ColorBarWidget()
        plot_layout.addWidget(self._colorbar)
        layout.addWidget(plot_row, stretch=1)

    # ------------------------------------------------------------------ 公共接口

    def show_mesh(self, bundle: ModelBundle | ConductionBundle) -> None:
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
        """按解类型分发渲染（静力/模态/谐响应/瞬态/屈曲/几何非线性/电-热耦合）."""
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
        elif isinstance(solution, ElectroThermalSolution):
            self._electrothermal = solution
            self._render_electrothermal(solution)
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
        """主题切换后重刷绘图背景与标尺刻度."""
        self._plot.setBackground(theme.current_palette().bg_app)
        self._colorbar.refresh_theme()

    # ------------------------------------------------------------------ 内部

    def _reset_controls(self) -> None:
        """隐藏类型关联控件并清空类型化解引用."""
        self._modal = None
        self._buckling = None
        self._nonlinear = None
        self._transient = None
        self._electrothermal = None
        self._solution = None
        self._freq_table.setVisible(False)
        self._mode_spin.setVisible(False)
        self._view_combo.setVisible(False)
        self._export_row.setVisible(False)
        self._colorbar.clear()

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
        colors = [pg.mkColor(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in scalar_colors(field, self._cmap)]
        self._plot.addItem(pg.ScatterPlotItem(x=deformed[:, 0], y=deformed[:, 1], size=6, brush=colors, pen=None))
        self._colorbar.set_field(field, self._cmap)

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
        colors = [pg.mkColor(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in scalar_colors(field, self._cmap)]
        self._plot.addItem(pg.ScatterPlotItem(x=deformed[:, 0], y=deformed[:, 1], size=6, brush=colors, pen=None))
        self._colorbar.set_field(field, self._cmap)

    def _render_harmonic(self, solution: HarmonicResponse) -> None:
        """渲染频响曲线（末端观察点 |uy| 随 ω 变化，对数幅值轴）并标注峰值."""
        mesh = solution.mesh
        node = tip_node(mesh)
        dof_y = node * mesh.dofs_per_node + 1  # 观察点竖向分量
        amplitude = np.abs(solution.displacements[dof_y, :])
        omegas = solution.frequencies
        self._colorbar.clear()  # 曲线视图无场量

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
        self._colorbar.clear()  # 曲线视图无场量

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

    def _on_cmap_changed(self, index: int) -> None:
        """切换色带：以当前解重渲染云图与标尺（保留类型内视图选择）."""
        self._cmap = self._cmap_combo.itemData(index)
        if self._solution is None:
            return  # 模型预览态无云图，下张云图生效
        saved = self._view_combo.currentIndex()
        self.show_solution(self._solution, self._reference_load)
        if saved > 0 and self._view_combo.count() > saved:
            self._view_combo.setCurrentIndex(saved)  # 触发 _on_view_changed 恢复原视图

    def _on_view_changed(self, index: int) -> None:
        """切换结果视图（非线性/瞬态/电-热耦合的类型关联双视图）."""
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
        elif self._electrothermal is not None:
            # 电-热耦合：0 = 温度云图，1 = 电压云图
            field = self._electrothermal.temperatures if index == 0 else self._electrothermal.voltages
            label = "温度 T" if index == 0 else "电压 V"
            self._render_scalar_field(self._electrothermal.mesh, field, label)

    def _render_scalar_field(self, mesh: Mesh, field: np.ndarray, label: str) -> None:
        """渲染标量场云图（未变形线框 + 节点场值着色）."""
        self._restore_mesh_view()
        edges = mesh_edges(mesh)
        segments = edge_segments(mesh.coords, edges)
        self._plot.clear()
        self._plot.plot(
            segments[:, :, 0].ravel(),
            segments[:, :, 1].ravel(),
            connect="pairs",
            pen=pg.mkPen(theme.current_palette().border_strong, width=2),
            name="网格",
        )
        colors = [pg.mkColor(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in scalar_colors(field, self._cmap)]
        self._plot.addItem(
            pg.ScatterPlotItem(x=mesh.coords[:, 0], y=mesh.coords[:, 1], size=6, brush=colors, pen=None, name=label)
        )
        self._colorbar.set_field(field, self._cmap)

    def _render_electrothermal(self, solution: ElectroThermalSolution) -> None:
        """渲染电-热耦合结果：显示视图切换并默认温度云图."""
        self._view_combo.setItemText(0, "温度云图")
        self._view_combo.setItemText(1, "电压云图")
        self._view_combo.blockSignals(True)
        self._view_combo.setCurrentIndex(0)
        self._view_combo.blockSignals(False)
        self._view_combo.setVisible(True)
        self._render_scalar_field(solution.mesh, solution.temperatures, "温度 T")
        self._summary.setText(
            f"峰值温度 T_max = {solution.t_max:.6g}（最低 {solution.t_min:.6g}）\n"
            f"总电功率 P = {solution.total_power:.6g} W（Joule 热源）"
        )

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
        self._colorbar.clear()  # 曲线视图无场量

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


class ResultTabs(QWidget):
    """多 TAB 结果容器（Workbench 风格）：每个环节节点的结果独立一页.

    - 页名 = 节点名；结果到达即建页并激活，节点重跑刷新原页（node_id 索引）；
    - 页可单独关闭；全部关闭或模板切换（clear）回到占位页；
    - 占位页不可关闭（提示运行入口）。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化容器：紧凑 TAB 条 + 占位页."""
        super().__init__(parent)
        self._views: dict[str, ResultView] = {}
        self._placeholder: ResultView | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._tab = QTabWidget(objectName="resultTabs")
        self._tab.setTabsClosable(True)
        self._tab.tabCloseRequested.connect(self._on_close_requested)
        layout.addWidget(self._tab)
        self._show_placeholder("尚未求解 —— 右键画布选择「运行全部」开始")

    # ------------------------------------------------------------------ 公共接口

    def view_for(self, node_id: str, title: str, activate: bool = True) -> ResultView:
        """取节点结果页（不存在则建页）；``activate=False`` 供预览批量建页不抢焦点."""
        view = self._views.get(node_id)
        if view is None:
            view = ResultView()
            self._views[node_id] = view
            self._tab.addTab(view, title)
            if self._placeholder is not None:  # 首个结果页顶替占位页
                self._tab.removeTab(self._tab.indexOf(self._placeholder))
                self._placeholder.deleteLater()
                self._placeholder = None
        if activate:
            self._tab.setCurrentWidget(view)
        return view

    def current_view(self) -> ResultView | None:
        """当前激活的结果页（占位页或无页时返回 None）."""
        widget = self._tab.currentWidget()
        return next((v for v in self._views.values() if v is widget), None)

    def show_error(self, node_id: str, title: str, message: str) -> None:
        """在节点页显示失败信息（无页则建页）."""
        self.view_for(node_id, title).show_error(message)

    def clear(self, message: str = "尚未求解") -> None:
        """清空全部结果页并回到占位页（模板切换时）."""
        self._views.clear()
        while self._tab.count():
            widget = self._tab.widget(0)
            self._tab.removeTab(0)
            widget.deleteLater()
        self._show_placeholder(message)

    def refresh_theme(self) -> None:
        """主题切换后重刷全部结果页绘图背景."""
        for index in range(self._tab.count()):
            widget = self._tab.widget(index)
            if isinstance(widget, ResultView):
                widget.refresh_theme()

    # ------------------------------------------------------------------ 内部

    def _show_placeholder(self, message: str) -> None:
        """显示占位页（仅提示文案，不可关闭）."""
        self._placeholder = ResultView()
        self._placeholder.clear(message)
        self._tab.addTab(self._placeholder, "结果")

    def _on_close_requested(self, index: int) -> None:
        """关闭单个结果页；占位页忽略，全部关闭后回到占位页."""
        widget = self._tab.widget(index)
        if widget is self._placeholder:
            return
        self._tab.removeTab(index)
        widget.deleteLater()
        for node_id, view in list(self._views.items()):
            if view is widget:
                del self._views[node_id]
        if self._tab.count() == 0:
            self._show_placeholder("尚未求解 —— 右键画布选择「运行全部」开始")
