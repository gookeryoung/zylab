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
    ElectroThermalTransientSolution,
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
    project3d,
    scalar_colors,
)
from zylab.studio import ConductionBundle, ModelBundle
from zylab.studio.nodes import tip_node

from .. import theme
from ..icons import nav_icon
from ..qt_compat import (
    QApplication,
    QColor,
    QComboBox,
    QEvent,
    QFileDialog,
    QFontMetrics,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QPainter,
    QPushButton,
    QSize,
    QSlider,
    QSpinBox,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTimer,
    QVBoxLayout,
    QWidget,
)

__all__ = ["ColorBarWidget", "ResultTabs", "ResultView"]


class ColorBarWidget(QWidget):
    """云图标尺（自绘）：竖向色带 + 最大/中值/最小刻度 + 单位.

    - 顶部为最大值色（与云图一致：标量归一化后经色带采样）；
    - 无场量时隐藏（clear），布局自动收回空间；
    - 刻度文字颜色随主题（refresh_theme 触发重绘）；
    - 单位后缀由场绑定方传入（如位移单位 m/mm）。
    """

    _BAR_W = 14  # 色带条宽度（像素）
    _PAD = 4  # 内边距
    _MAX_H = 460  # 最大高度（与绘图区同宽列内随高拉伸，超高截断）
    _MAX_LABEL_W = 120  # 刻度文字最大宽度（超大数值如 3.172e+04，超出省略截断）

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化标尺（初始无场量，隐藏）."""
        super().__init__(parent)
        self._vmin = 0.0
        self._vmax = 1.0
        self._unit = ""
        self._lut = cmap_lut("jet")
        self.setFixedWidth(self._BAR_W + 72)
        self.setMaximumHeight(self._MAX_H)
        self.setVisible(False)

    def set_field(self, values: np.ndarray, cmap: str, unit: str = "") -> None:
        """绑定标量场与色带（云图渲染时同步调用）."""
        data = np.asarray(values, dtype=float)
        if data.size == 0:
            self.clear()
            return
        self._vmin = float(np.min(data))
        self._vmax = float(np.max(data))
        self._unit = unit
        self._lut = cmap_lut(cmap)
        self._fit_width()
        self.setVisible(True)
        self.update()

    def clear(self) -> None:
        """清除场量（隐藏标尺）."""
        self.setVisible(False)

    def refresh_theme(self) -> None:
        """主题切换后重绘刻度文字."""
        self.update()

    def _labels(self) -> tuple[str, str, str]:
        """三档刻度文本（最大/中/最小，最大最小带单位后缀）."""
        unit = f" {self._unit}" if self._unit else ""
        mid = (self._vmin + self._vmax) / 2.0
        return f"{self._vmax:.4g}{unit}", f"{mid:.4g}", f"{self._vmin:.4g}{unit}"

    def _fit_width(self) -> None:
        """宽度按最宽刻度文本自适应（上限截断），避免文字被绘图区遮挡."""
        self._ensure_width(QFontMetrics(self.font()))

    def _ensure_width(self, metrics: QFontMetrics) -> None:
        """按给定字体度量确保宽度容纳刻度全文（超上限封顶省略）.

        字体度量晚于 :meth:`set_field` 生效时（如样式表字体在 polish 后
        才应用），绘制期以实际画笔字体再补偿一次，保证大数值刻度
        （如 ``3.172e+04``）全文显示而非被绘图区遮挡。
        """
        text_x = self._PAD + self._BAR_W + self._PAD
        widest = max(metrics.horizontalAdvance(t) for t in self._labels())
        needed = text_x + widest + self._PAD
        if needed > self.width():
            self.setFixedWidth(min(needed, text_x + self._MAX_LABEL_W + self._PAD))

    def paintEvent(self, event) -> None:  # Qt 命名约定
        """绘制竖向色带（64 段）、三档刻度值与单位后缀."""
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
        # 刻度：最大（顶）/中/最小（底）；文字行高按字体度量（避免下部裁剪）
        metrics = QFontMetrics(painter.font())
        text_h = metrics.height()
        text_x = bar_x + self._BAR_W + self._PAD
        # 字体度量晚于宽度自适应生效时按实际画笔字体补偿一次宽度
        self._ensure_width(metrics)
        text_w = self.width() - text_x - self._PAD
        labels = self._labels()
        painter.setPen(QColor(pal.text_secondary))
        ys = (self._PAD, (self.height() - text_h) // 2, self.height() - self._PAD - text_h)
        for y, text in zip(ys, labels):
            painter.drawText(text_x, y, text_w, text_h, 0, metrics.elidedText(text, Qt.ElideRight, text_w))


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
        self._et_transient: ElectroThermalTransientSolution | None = None
        self._reference_load = 1.0
        self._solution: object | None = None
        self._mesh_bundle: ModelBundle | ConductionBundle | None = None
        self._cmap = "jet"
        self._unit = "m"
        # 3D 视角状态：等轴测方位角/仰角（拖拽旋转更新，重绘时投影）
        self._azim = 30.0
        self._elev = 25.0
        self._is3d = False
        self._dragging = False
        self._drag_pos: tuple[int, int] | None = None
        # 动画状态：帧序列（位移数组, 标签）+ 统一变形放大系数
        self._frames: list[tuple[np.ndarray, str]] = []
        self._frame_index = 0
        self._anim_mesh: Mesh | None = None
        self._anim_scale = 1.0
        # 标量场动画状态（瞬态温度云图）：帧序列（场数组, 标签）
        self._scalar_frames: list[tuple[np.ndarray, str]] = []
        self._scalar_mesh: Mesh | None = None
        self._scalar_label = ""
        self._scalar_unit = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACING_SM)

        # 摘要单行化（超宽由结果类型控制文本长度，不再换行撑高）
        self._summary = QLabel("尚未求解", objectName="resultText")
        layout.addWidget(self._summary)

        # 单条工具条：阶数/视图/动画控制 + 色带/单位 + 导出图标按钮
        # （Workbench 布局习惯：功能键纯图标 + tooltip，控制条恒占一行）
        self._toolbar = QWidget(objectName="resultToolbar")
        bar = QHBoxLayout(self._toolbar)
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(theme.SPACING_SM)
        self._mode_spin = QSpinBox()
        self._mode_spin.setRange(1, 1)
        self._mode_spin.setPrefix("振型 ")
        self._mode_spin.setToolTip("模态阶数（模态/屈曲解显示）")
        self._mode_spin.setVisible(False)
        self._mode_spin.valueChanged.connect(self._render_mode_shape)
        bar.addWidget(self._mode_spin)
        self._view_combo = QComboBox(objectName="nonlinearView")
        self._view_combo.addItem("变形云图", "deform")
        self._view_combo.addItem("载荷-位移曲线", "curve")
        self._view_combo.setToolTip("结果视图切换")
        self._view_combo.setVisible(False)
        self._view_combo.currentIndexChanged.connect(self._on_view_changed)
        bar.addWidget(self._view_combo)
        self._play_btn = QPushButton(objectName="iconBtn")
        self._play_btn.setToolTip("播放动画")
        self._play_btn.setIconSize(QSize(14, 14))
        self._play_btn.clicked.connect(self._on_play_toggled)
        self._play_btn.setVisible(False)
        bar.addWidget(self._play_btn)
        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.valueChanged.connect(self._on_frame_slider)
        self._frame_slider.setVisible(False)
        bar.addWidget(self._frame_slider, stretch=1)
        self._frame_label = QLabel(objectName="secondaryText")
        self._frame_label.setVisible(False)
        bar.addWidget(self._frame_label)
        bar.addStretch(1)
        self._cmap_combo = QComboBox(objectName="cmapCombo")
        for key in cmap_keys():
            self._cmap_combo.addItem(cmap_label(key), key)
        self._cmap_combo.setToolTip("云图色带配色")
        self._cmap_combo.currentIndexChanged.connect(self._on_cmap_changed)
        self._cmap_combo.setVisible(False)
        bar.addWidget(self._cmap_combo)
        self._unit_combo = QComboBox(objectName="unitCombo")
        self._unit_combo.addItem("m (米)", "m")
        self._unit_combo.addItem("mm (毫米)", "mm")
        self._unit_combo.setToolTip("位移显示单位")
        self._unit_combo.currentIndexChanged.connect(self._on_unit_changed)
        self._unit_combo.setVisible(False)
        bar.addWidget(self._unit_combo)
        self._export_csv_btn = QPushButton(objectName="iconBtn")
        self._export_csv_btn.setToolTip("导出 CSV")
        self._export_csv_btn.setIconSize(QSize(14, 14))
        self._export_csv_btn.clicked.connect(self._on_export_csv)
        self._export_csv_btn.setVisible(False)
        bar.addWidget(self._export_csv_btn)
        self._export_png_btn = QPushButton(objectName="iconBtn")
        self._export_png_btn.setToolTip("导出 PNG")
        self._export_png_btn.setIconSize(QSize(14, 14))
        self._export_png_btn.clicked.connect(self._on_export_png)
        self._export_png_btn.setVisible(False)
        bar.addWidget(self._export_png_btn)
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(120)  # ms/帧
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self._refresh_button_icons()
        layout.addWidget(self._toolbar)

        # 数据页（模态/屈曲频率表，整页呈现；行点击联动阶数）
        self._freq_table = QTableWidget(objectName="freqTable")
        self._freq_table.setColumnCount(3)
        self._freq_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._freq_table.verticalHeader().setVisible(False)
        self._freq_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._freq_table.cellClicked.connect(self._on_table_row_clicked)
        self._data_page = QWidget()
        data_layout = QVBoxLayout(self._data_page)
        data_layout.setContentsMargins(theme.SPACING_SM, theme.SPACING_SM, theme.SPACING_SM, theme.SPACING_SM)
        data_layout.addWidget(self._freq_table)

        # 绘图区子 TAB：「云图」（色条 + 绘图）+「数据」（仅模态/屈曲）；
        # 单页时隐藏页签条（不浪费纵向空间）
        self._plot = pg.PlotWidget(background=theme.current_palette().bg_app)
        self._setup_plot_menu()
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setAspectLocked(True)
        self._plot.addLegend(offset=(12, 12))
        self._plot_row = QWidget()
        plot_layout = QHBoxLayout(self._plot_row)
        # 绘图行外边距：云图与周边控件/页边保留呼吸空间（原零边距贴边过近）
        margin = theme.SPACING_SM
        plot_layout.setContentsMargins(margin, margin, margin, margin)
        plot_layout.setSpacing(theme.SPACING_SM)
        self._colorbar = ColorBarWidget()
        plot_layout.addWidget(self._colorbar)  # 标尺居左（Workbench 布局习惯）
        plot_layout.addWidget(self._plot, stretch=1)
        self._tabs = QTabWidget(objectName="resultSubTabs")
        self._tabs.setTabPosition(QTabWidget.South)  # Workbench 数据页签居底
        self._tabs.addTab(self._plot_row, "云图")
        self._tabs.tabBar().setVisible(False)
        layout.addWidget(self._tabs, stretch=1)  # 唯一弹性项：占满剩余全部高度
        # 3D 模式下拦截绘图区拖拽做视角旋转（此时禁用平移）
        self._plot.viewport().installEventFilter(self)

    def _setup_plot_menu(self) -> None:
        """以精简中文右键菜单替换 pyqtgraph 默认菜单.

        默认菜单为英文且依赖内置功能：Average 项在本主题下渲染为白块遮挡、
        Downsampling 项对 ScatterPlotItem 触发 setDownsampling 崩溃，故整体
        关闭 PlotItem 菜单并替换 ViewBox 菜单（保留右键交互习惯）。
        """
        plot_item = self._plot.getPlotItem()
        # 仅关 PlotItem 菜单（enableViewBoxMenu=None 保留 ViewBox 弹出能力）
        plot_item.setMenuEnabled(False, enableViewBoxMenu=None)
        menu = QMenu(self._plot)
        menu.addAction("恢复默认视角", plot_item.autoRange)
        menu.addAction("复制图像", self._copy_plot)
        menu.addAction("导出图像 (PNG)", self._on_export_png)
        menu.addAction("导出数据 (CSV)", self._on_export_csv)
        plot_item.vb.menu = menu

    def _copy_plot(self) -> None:
        """当前绘图区截图复制到剪贴板（可直接粘贴到文档）."""
        QApplication.clipboard().setPixmap(self._plot.grab())
        self._summary.setText("图像已复制到剪贴板")

    # ------------------------------------------------------------------ 公共接口

    def show_mesh(self, bundle: ModelBundle | ConductionBundle) -> None:
        """渲染模型网格预览（未变形线框 + 节点散点 + 规模摘要；3D 等轴测投影）."""
        self._reset_controls()
        self._mesh_bundle = bundle
        mesh = bundle.mesh
        self._restore_mesh_view()
        xy, depth = self._project(mesh.coords)
        edges = mesh_edges(mesh)
        segments = edge_segments(xy, edges)
        self._plot.clear()
        self._plot.plot(
            segments[:, :, 0].ravel(),
            segments[:, :, 1].ravel(),
            connect="pairs",
            pen=pg.mkPen(theme.current_palette().border_strong, width=2),
            name="未变形",
        )
        # 节点散点（杆系模型一条线时节点分布也清晰可辨；3D 按深度排序遮挡）
        self._plot.addItem(
            pg.ScatterPlotItem(
                **self._scatter_kwargs(xy, depth, len(mesh.coords)),
                size=5,
                brush=pg.mkBrush(theme.current_palette().primary),
                pen=None,
            )
        )
        self._summary.setText(
            f"模型预览 · {mesh.n_nodes} 节点 · {mesh.n_elements} 单元" + ("（拖拽旋转视角）" if self._is3d else "")
        )
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
        elif isinstance(solution, ElectroThermalTransientSolution):
            self._et_transient = solution
            self._render_et_transient(solution)
        elif isinstance(solution, StaticSolution):
            self._render_deformation(solution.mesh, solution.displacements)
            max_u = float(np.max(np.linalg.norm(solution.displacements, axis=1)))
            self._summary.setText(
                f"最大位移 |u| = {max_u * self._disp_factor():.6g} {self._unit} · 应变能 = {solution.strain_energy:.6g}"
            )
        else:
            self._summary.setText(f"未知结果类型: {type(solution).__name__}")
            return
        self._solution = solution
        for btn in (self._cmap_combo, self._unit_combo, self._export_csv_btn, self._export_png_btn):
            btn.setVisible(True)

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
        """主题切换后重刷绘图背景、标尺刻度与工具条图标."""
        self._plot.setBackground(theme.current_palette().bg_app)
        self._colorbar.refresh_theme()
        self._refresh_button_icons()

    # ------------------------------------------------------------------ 内部

    def _reset_controls(self) -> None:
        """隐藏类型关联控件并清空类型化解引用."""
        self._modal = None
        self._buckling = None
        self._nonlinear = None
        self._transient = None
        self._electrothermal = None
        self._et_transient = None
        self._solution = None
        self._mesh_bundle = None
        self._stop_anim()
        self._frames = []
        self._anim_mesh = None
        self._scalar_frames = []
        self._scalar_mesh = None
        self._remove_data_tab()
        self._mode_spin.setVisible(False)
        self._view_combo.setVisible(False)
        for btn in (self._cmap_combo, self._unit_combo, self._export_csv_btn, self._export_png_btn):
            btn.setVisible(False)
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

    # ------------------------------------------------------------------ 3D 视角

    def _project(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """坐标投影到屏幕平面（2D 直通；3D 等轴测投影并返回观察深度）.

        3D 时切换交互模式（禁平移 + 标记可拖拽旋转），2D 时恢复。
        """
        is3d = coords.shape[1] >= 3
        if is3d != self._is3d:
            self._is3d = is3d
            self._plot.setMouseEnabled(x=not is3d, y=not is3d)
        if not is3d:
            return coords[:, :2], None
        xy, depth = project3d(coords[:, :3], self._azim, self._elev)
        return xy, depth

    @staticmethod
    def _scatter_kwargs(xy: np.ndarray, depth: np.ndarray | None, n: int) -> dict[str, np.ndarray]:
        """散点绘制参数（3D 按深度排序：远者先画近者覆盖，形成遮挡）."""
        if depth is None or n == 0:
            return {"x": xy[:, 0], "y": xy[:, 1]}
        order = np.argsort(depth, kind="stable")
        return {"x": xy[order, 0], "y": xy[order, 1]}

    def eventFilter(self, obj, event) -> bool:  # Qt 命名约定
        """3D 模式下拦截绘图区拖拽旋转视角（方位角/仰角），其余事件放行."""
        if not self._is3d:
            return super().eventFilter(obj, event)
        if event.type() == QEvent.MouseButtonPress and event.buttons() & Qt.LeftButton:
            self._dragging = True
            self._drag_pos = (event.pos().x(), event.pos().y())
            return True
        if event.type() == QEvent.MouseMove and self._dragging and self._drag_pos is not None:
            dx = event.pos().x() - self._drag_pos[0]
            dy = event.pos().y() - self._drag_pos[1]
            self._drag_pos = (event.pos().x(), event.pos().y())
            self._azim = (self._azim + 0.5 * dx) % 360.0
            self._elev = float(np.clip(self._elev - 0.5 * dy, -89.0, 89.0))
            self._rerender_3d()
            return True
        if event.type() in (QEvent.MouseButtonRelease, QEvent.Leave):
            self._dragging = False
            self._drag_pos = None
        return super().eventFilter(obj, event)

    def _rerender_3d(self) -> None:
        """视角变化后按当前内容重投影（解态重渲染，预览态重画网格）."""
        if self._solution is not None:
            self._rerender_current()
        elif self._mesh_bundle is not None:
            self.show_mesh(self._mesh_bundle)

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

    def _disp_factor(self) -> float:
        """位移显示换算系数（m -> mm 时 1000）."""
        return 1000.0 if self._unit == "mm" else 1.0

    def _draw_deformed(self, mesh: Mesh, displacements: np.ndarray, scale: float, label: str) -> None:
        """绘制变形线框与节点位移模着色散点（不动标尺/摘要，供单帧与动画复用）."""
        self._restore_mesh_view()
        edges = mesh_edges(mesh)
        field = np.linalg.norm(displacements, axis=1)
        deformed = deformed_coords(mesh, displacements, scale)
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
        self._colorbar.set_field(field * self._disp_factor(), self._cmap, self._unit)

    def _render_deformation(self, mesh: Mesh, displacements: np.ndarray) -> None:
        """渲染变形线框与节点位移模云图（静力/几何非线性共用）."""
        field = np.linalg.norm(displacements, axis=1)
        scale = self._deform_scale(mesh, field)
        self._draw_deformed(mesh, displacements, scale, "变形")

    # ------------------------------------------------------------------ 动画

    def _start_anim(self, mesh: Mesh, frames: list[tuple[np.ndarray, str]], *, first_index: int = 0) -> None:
        """绑定动画帧序列（帧数 <= 1 时隐藏控制行）.

        Args:
            mesh: 云图网格（帧位移的形状上下文）。
            frames: ``[(位移数组, 帧标签), ...]``。
            first_index: 初始显示帧（结果态默认末帧，振型默认当前阶）。
        """
        self._scalar_frames = []  # 位移帧与标量帧互斥（同一时刻仅一类动画）
        self._scalar_mesh = None
        self._frames = frames
        self._anim_mesh = mesh
        if len(frames) <= 1:
            self._stop_anim()
            return
        max_u = max(float(np.max(np.abs(disp))) for disp, _ in frames)
        self._anim_scale = self._deform_scale(mesh, np.array([max_u]))
        self._frame_slider.blockSignals(True)
        self._frame_slider.setRange(0, len(frames) - 1)
        self._frame_slider.setValue(first_index)
        self._frame_slider.blockSignals(False)
        for widget in (self._play_btn, self._frame_slider, self._frame_label):
            widget.setVisible(True)
        self._frame_index = first_index
        self._show_frame(first_index)

    def _show_frame(self, index: int) -> None:
        """显示指定帧（位移/标量帧分发重绘 + 滑块/标签联动）."""
        if self._scalar_frames:  # 标量帧（瞬态温度云图）
            if not (0 <= index < len(self._scalar_frames)) or self._scalar_mesh is None:
                return
            self._frame_index = index
            field, label = self._scalar_frames[index]
            self._render_scalar_field(self._scalar_mesh, field, self._scalar_label, self._scalar_unit)
            self._frame_label.setText(label)
        elif self._frames and self._anim_mesh is not None and 0 <= index < len(self._frames):
            self._frame_index = index
            disp, label = self._frames[index]
            self._draw_deformed(self._anim_mesh, disp, self._anim_scale, label)
            self._frame_label.setText(label)
        else:
            return
        if self._frame_slider.value() != index:
            self._frame_slider.blockSignals(True)
            self._frame_slider.setValue(index)
            self._frame_slider.blockSignals(False)

    def _start_scalar_anim(
        self,
        mesh: Mesh,
        frames: list[tuple[np.ndarray, str]],
        label: str,
        unit: str,
        *,
        first_index: int = 0,
    ) -> None:
        """绑定标量场帧序列（复用播放/滑块控件；帧数 <= 1 时退化为静态云图）.

        Args:
            mesh: 云图网格（场数组的形状上下文）。
            frames: ``[(场数组, 帧标签), ...]``（逐帧节点标量，如各时刻温度）。
            label: 场名（图例与标尺标题，如「温度 T」）。
            unit: 场单位（标尺刻度后缀，如 K）。
            first_index: 初始显示帧（结果态默认末帧）。
        """
        self._frames = []  # 位移帧与标量帧互斥（同一时刻仅一类动画）
        self._anim_mesh = None
        self._scalar_mesh = mesh
        self._scalar_label = label
        self._scalar_unit = unit
        self._scalar_frames = frames
        if len(frames) <= 1:
            self._stop_anim()
            if frames:  # 单帧：仍渲染静态云图并显示帧标签
                field, text = frames[0]
                self._render_scalar_field(mesh, field, label, unit)
                self._frame_label.setText(text)
            return
        self._frame_slider.blockSignals(True)
        self._frame_slider.setRange(0, len(self._scalar_frames) - 1)
        self._frame_slider.setValue(first_index)
        self._frame_slider.blockSignals(False)
        for widget in (self._play_btn, self._frame_slider, self._frame_label):
            widget.setVisible(True)
        self._frame_index = first_index
        self._show_frame(first_index)

    def _on_play_toggled(self) -> None:
        """播放/暂停切换（播放到末帧自动停在末帧）."""
        if self._anim_timer.isActive():
            self._anim_timer.stop()
            self._set_play_icon(False)
        else:
            if not self._frames and not self._scalar_frames:
                return
            self._frame_index = 0 if self._frame_index >= self._active_frame_count() - 1 else self._frame_index
            self._show_frame(self._frame_index)
            self._anim_timer.start()
            self._set_play_icon(True)

    def _active_frame_count(self) -> int:
        """当前活跃动画帧数（位移帧或标量帧，取在绑者）."""
        if self._scalar_frames:
            return len(self._scalar_frames)
        return len(self._frames)

    def _on_anim_tick(self) -> None:
        """定时器驱动逐帧推进（末帧回绕重新播放）."""
        if not self._frames and not self._scalar_frames:
            self._stop_anim()
            return
        self._show_frame((self._frame_index + 1) % self._active_frame_count())

    def _on_frame_slider(self, value: int) -> None:
        """拖动帧滑块：暂停自动播放并显示指定帧."""
        if self._anim_timer.isActive():
            self._on_play_toggled()
        self._show_frame(value)

    def _stop_anim(self) -> None:
        """停止播放并隐藏动画控件."""
        self._anim_timer.stop()
        self._set_play_icon(False)
        for widget in (self._play_btn, self._frame_slider, self._frame_label):
            widget.setVisible(False)

    def _render_modal(self, solution: ModalSolution) -> None:
        """填充频率表并渲染首阶振型云图."""
        self._fill_freq_table(solution)
        self._summary.setText(
            f"前 {solution.n_modes} 阶 · 基频 ω₁ = {solution.frequencies[0]:.6g} rad/s"
            f"（{solution.frequencies_hz[0]:.6g} Hz）"
        )
        self._render_mode_shape(1)

    def _fill_freq_table(self, solution: ModalSolution) -> None:
        """频率表（数据页）+ 振型序号控件（模态专用）."""
        self._freq_table.setHorizontalHeaderLabels(("阶", "ω (rad/s)", "f (Hz)"))
        self._freq_table.setRowCount(solution.n_modes)
        for i in range(solution.n_modes):
            self._freq_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self._freq_table.setItem(i, 1, QTableWidgetItem(f"{solution.frequencies[i]:.6g}"))
            self._freq_table.setItem(i, 2, QTableWidgetItem(f"{solution.frequencies_hz[i]:.6g}"))
        self._show_data_tab()
        self._mode_spin.blockSignals(True)
        self._mode_spin.setRange(1, solution.n_modes)
        self._mode_spin.setValue(1)
        self._mode_spin.blockSignals(False)
        self._mode_spin.setVisible(True)

    def _render_buckling(self, solution: BucklingSolution) -> None:
        """填充载荷因子表（数据页）并渲染首阶屈曲模态."""
        self._freq_table.setHorizontalHeaderLabels(("阶", "载荷因子 λ", "临界载荷"))
        self._freq_table.setRowCount(solution.n_modes)
        for i in range(solution.n_modes):
            factor = float(solution.load_factors[i])
            self._freq_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self._freq_table.setItem(i, 1, QTableWidgetItem(f"{factor:.6g}"))
            self._freq_table.setItem(i, 2, QTableWidgetItem(f"{factor * self._reference_load:.6g}"))
        self._show_data_tab()
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
        """渲染第 index 阶（1 基）振型/屈曲模态云图（各阶可动画循环）."""
        solution = self._modal if self._modal is not None else self._buckling
        if solution is None:
            return
        mesh = solution.mesh
        kind = "振型" if self._modal is not None else "屈曲"
        frames = [
            (solution.mode_shape(i)[:, :2], f"第 {i + 1} 阶{kind}")  # 梁含转角分量，仅取平动
            for i in range(solution.n_modes)
        ]
        self._start_anim(mesh, frames, first_index=index - 1)
        # 数据页同步高亮当前阶（cellClicked 不因程序化选择触发，无回环）
        self._freq_table.selectRow(index - 1)

    def _show_data_tab(self) -> None:
        """加入「数据」子页（频率/载荷因子表整页呈现；页签条随之显示）."""
        if self._tabs.indexOf(self._data_page) < 0:
            self._tabs.addTab(self._data_page, "数据")
            self._tabs.tabBar().setVisible(True)

    def _remove_data_tab(self) -> None:
        """移除「数据」子页（非模态/屈曲解；单页时隐藏页签条）."""
        index = self._tabs.indexOf(self._data_page)
        if index >= 0:
            self._tabs.removeTab(index)
        self._tabs.tabBar().setVisible(self._tabs.count() > 1)

    def _on_table_row_clicked(self, row: int, _column: int) -> None:
        """数据页表格行点击：切换到对应阶模态云图."""
        if self._mode_spin.isVisibleTo(self) and 0 <= row < self._mode_spin.maximum():
            self._mode_spin.setValue(row + 1)  # 触发 _render_mode_shape 重渲染

    def _set_play_icon(self, playing: bool) -> None:
        """播放按钮图标态切换（播放中显示暂停图标）."""
        pal = theme.current_palette()
        self._play_btn.setIcon(nav_icon("pause" if playing else "play", pal.text_primary))
        self._play_btn.setToolTip("暂停" if playing else "播放动画")

    def _refresh_button_icons(self) -> None:
        """按当前主题刷新工具条图标按钮（导出/播放）."""
        pal = theme.current_palette()
        self._export_csv_btn.setIcon(nav_icon("export_csv", pal.text_secondary))
        self._export_png_btn.setIcon(nav_icon("export_png", pal.text_secondary))
        self._set_play_icon(self._anim_timer.isActive())

    def _render_harmonic(self, solution: HarmonicResponse) -> None:
        """渲染频响曲线（末端观察点 |uy| 随 ω 变化，对数幅值轴）并标注峰值."""
        mesh = solution.mesh
        node = tip_node(mesh)
        dof_y = node * mesh.dofs_per_node + 1  # 观察点竖向分量
        amplitude = np.abs(solution.displacements[dof_y, :]) * self._disp_factor()
        omegas = solution.frequencies
        self._stop_anim()  # 曲线视图无动画帧
        self._colorbar.clear()  # 曲线视图无场量

        self._plot.clear()
        self._plot.setAspectLocked(False)
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setLogMode(y=True)
        self._plot.setLabel("bottom", "ω", units="rad/s")
        self._plot.setLabel("left", f"|uy|（对数，{self._unit}）")
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
            f"峰值 |uy| = {amplitude[peak_index]:.6g} {self._unit} @ ω = {omegas[peak_index]:.6g} rad/s"
            f"（{solution.n_frequencies} 个频率点）"
        )

    def _render_nonlinear(self, solution: NonlinearSolution) -> None:
        """渲染非线性结果：显示视图切换并默认变形云图."""
        self._set_view_items((("变形云图", "field"), ("载荷-位移曲线", "curve")))
        self._render_nonlinear_deform(solution)

    def _render_nonlinear_deform(self, solution: NonlinearSolution) -> None:
        """渲染非线性变形云图（增量步动画）与迭代历程摘要."""
        mesh = solution.mesh
        frames = [
            (snapshot.reshape(mesh.n_nodes, mesh.dofs_per_node), f"λ = {factor:.3g}")
            for snapshot, factor in zip(solution.history_displacements, solution.history_factors)
        ]
        if len(frames) > 1:
            self._start_anim(mesh, frames, first_index=len(frames) - 1)  # 结果态默认末帧（收敛态）
        else:  # 无历时快照（兼容旧结果）：退化为单帧收敛态云图
            self._render_deformation(mesh, solution.displacements)
        max_u = float(np.max(np.linalg.norm(solution.displacements, axis=1)))
        self._summary.setText(
            f"最大位移 |u| = {max_u * self._disp_factor():.6g} {self._unit} · "
            f"收敛：{len(solution.iterations)} 增量步 · {solution.total_iterations} 次 Newton 迭代"
        )

    def _render_nonlinear_curve(self, solution: NonlinearSolution) -> None:
        """渲染载荷-位移曲线（最大位移节点 uy vs 载荷因子，含线性参照虚线）."""
        node = int(np.argmax(np.linalg.norm(solution.displacements, axis=1)))
        uy = solution.history_dof(node, 1) * self._disp_factor()
        factors = solution.history_factors
        self._stop_anim()  # 曲线视图无动画帧
        self._colorbar.clear()  # 曲线视图无场量

        self._plot.clear()
        self._plot.setAspectLocked(False)
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setLogMode(y=False)
        self._plot.setLabel("bottom", "载荷因子 λ")
        self._plot.setLabel("left", f"节点 {node} uy ({self._unit})")
        slope = uy[1] / factors[1]
        self._plot.plot(
            factors,
            slope * factors,
            pen=pg.mkPen(theme.current_palette().text_secondary, width=2, style=pg.QtCore.Qt.DashLine),
            name="线性参照",
        )
        self._plot.plot(factors, uy, pen=pg.mkPen(theme.current_palette().primary, width=3), name="非线性")
        self._summary.setText(
            f"载荷-位移曲线：{len(factors) - 1} 增量步 · "
            f"观察点节点 {node} uy = {uy[-1]:.6g} {self._unit}（线性 {slope * factors[-1]:.6g}）"
        )

    def _on_cmap_changed(self, index: int) -> None:
        """切换色带：以当前解重渲染云图与标尺（保留类型内视图选择）."""
        self._cmap = self._cmap_combo.itemData(index)
        self._rerender_current()

    def _on_unit_changed(self, index: int) -> None:
        """切换位移单位（m/mm）：重渲染当前解的标尺刻度与文本."""
        self._unit = self._unit_combo.itemData(index)
        self._rerender_current()

    def _rerender_current(self) -> None:
        """按当前解重渲染（保留类型内视图选择、模态阶数与动画帧位置）."""
        if self._solution is None:
            return  # 模型预览态无云图，下次渲染生效
        saved_view = self._view_combo.currentIndex()
        saved_frame = self._frame_index
        saved_mode = self._mode_spin.value() if self._mode_spin.isVisibleTo(self) else 0
        self.show_solution(self._solution, self._reference_load)
        if saved_view > 0 and self._view_combo.count() > saved_view:
            self._view_combo.setCurrentIndex(saved_view)  # 触发 _on_view_changed 恢复原视图
        elif saved_mode > 1 and self._mode_spin.isVisibleTo(self):
            self._mode_spin.setValue(saved_mode)  # 触发 _render_mode_shape 恢复原阶
        elif saved_frame > 0 and self._frames:
            self._show_frame(min(saved_frame, len(self._frames) - 1))  # 恢复动画帧位置
        elif saved_frame > 0 and self._scalar_frames:
            self._show_frame(min(saved_frame, len(self._scalar_frames) - 1))  # 恢复标量帧位置

    def _set_view_items(self, items: tuple[tuple[str, str], ...], current: int = 0) -> None:
        """重设视图下拉项 ``(text, data)``（清空其他类型遗留项）并选默认项."""
        self._view_combo.blockSignals(True)
        while self._view_combo.count():
            self._view_combo.removeItem(0)
        for text, data in items:
            self._view_combo.addItem(text, data)
        self._view_combo.setCurrentIndex(current)
        self._view_combo.blockSignals(False)
        self._view_combo.setVisible(True)

    def _on_view_changed(self, index: int) -> None:
        """切换结果视图（非线性/瞬态/电-热耦合的类型关联多视图）."""
        curve = self._view_combo.itemData(index) == "curve" if self._view_combo.count() else False
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
        elif self._et_transient is not None:
            if index == 1:  # 电压云图（常值场）
                self._stop_anim()
                self._render_scalar_field(self._et_transient.mesh, self._et_transient.voltages, "电压 V", "V")
            elif index == 2:  # 温度时程曲线
                self._render_et_transient_curve(self._et_transient)
            else:  # 温度云图（时程动画）
                self._render_et_transient_field(self._et_transient)
        elif self._electrothermal is not None:
            # 电-热耦合：0 = 温度云图，1 = 电压云图
            field = self._electrothermal.temperatures if index == 0 else self._electrothermal.voltages
            label = "温度 T" if index == 0 else "电压 V"
            unit = "K" if index == 0 else "V"
            self._render_scalar_field(self._electrothermal.mesh, field, label, unit)

    def _render_scalar_field(self, mesh: Mesh, field: np.ndarray, label: str, unit: str = "") -> None:
        """渲染标量场云图（未变形线框 + 节点场值着色；3D 等轴测投影）."""
        self._restore_mesh_view()
        xy, depth = self._project(mesh.coords)
        edges = mesh_edges(mesh)
        segments = edge_segments(xy, edges)
        self._plot.clear()
        self._plot.plot(
            segments[:, :, 0].ravel(),
            segments[:, :, 1].ravel(),
            connect="pairs",
            pen=pg.mkPen(theme.current_palette().border_strong, width=2),
            name="网格",
        )
        colors = [pg.mkColor(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in scalar_colors(field, self._cmap)]
        if depth is not None:  # 3D：深度排序使近侧散点覆盖远侧（遮挡层次）
            order = np.argsort(depth, kind="stable")
            spots = [{"pos": (float(xy[i, 0]), float(xy[i, 1])), "brush": colors[i], "size": 6} for i in order]
            self._plot.addItem(pg.ScatterPlotItem(spots=spots, pen=None, name=label))
        else:
            self._plot.addItem(pg.ScatterPlotItem(x=xy[:, 0], y=xy[:, 1], size=6, brush=colors, pen=None, name=label))
        self._colorbar.set_field(field, self._cmap, unit)

    def _render_electrothermal(self, solution: ElectroThermalSolution) -> None:
        """渲染电-热耦合结果：显示视图切换并默认温度云图."""
        self._set_view_items((("温度云图", "field"), ("电压云图", "field")))
        self._render_scalar_field(solution.mesh, solution.temperatures, "温度 T")
        self._summary.setText(
            f"峰值温度 T_max = {solution.t_max:.6g}（最低 {solution.t_min:.6g}） · "
            f"总电功率 P = {solution.total_power:.6g} W（Joule 热源）"
        )

    # ------------------------------------------------------------------ 瞬态电-热

    def _render_et_transient(self, solution: ElectroThermalTransientSolution) -> None:
        """渲染瞬态电-热耦合结果：三视图（温度云图动画/电压云图/温度时程）."""
        self._set_view_items((("温度云图", "field"), ("电压云图", "field"), ("温度时程曲线", "curve")))
        self._render_et_transient_field(solution)
        thermal = solution.thermal
        self._summary.setText(
            f"末帧峰值温度 T_max = {solution.t_max:.6g}（最低 {solution.t_min:.6g}） · "
            f"总电功率 P = {solution.total_power:.6g} W · 时程：{thermal.times.size - 1} 步"
            f" · 总时长 = {thermal.total_time:.6g} s"
        )

    def _render_et_transient_field(self, solution: ElectroThermalTransientSolution) -> None:
        """渲染瞬态温度云图时程动画（结果态默认末帧，可播放/逐帧回放）."""
        thermal = solution.thermal
        frames = [(thermal.temperatures[i], f"t = {t:.4g} s") for i, t in enumerate(thermal.times)]
        self._start_scalar_anim(solution.mesh, frames, "温度 T", "K", first_index=len(frames) - 1)

    def _render_et_transient_curve(self, solution: ElectroThermalTransientSolution) -> None:
        """渲染温度时程曲线（末帧热点节点的 T(t)）并标注峰值."""
        thermal = solution.thermal
        hotspot = int(np.argmax(thermal.temperatures[-1]))
        temps = thermal.temperatures[:, hotspot]
        times = thermal.times
        self._stop_anim()  # 曲线视图无动画帧
        self._colorbar.clear()  # 曲线视图无场量

        self._plot.clear()
        self._plot.setAspectLocked(False)
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setLogMode(y=False)
        self._plot.setLabel("bottom", "时间 t (s)")
        self._plot.setLabel("left", f"热点节点 {hotspot} T (K)")
        self._plot.plot(
            times,
            temps,
            pen=pg.mkPen(theme.current_palette().primary, width=3),
            name=f"节点 {hotspot} T",
        )
        peak_index = int(np.argmax(temps))
        self._plot.addItem(
            pg.ScatterPlotItem(
                x=[times[peak_index]],
                y=[temps[peak_index]],
                size=10,
                brush=pg.mkBrush(theme.current_palette().error_text),
                pen=None,
                name="峰值",
            )
        )
        self._summary.setText(
            f"热点峰值 T = {temps[peak_index]:.6g} K @ t = {times[peak_index]:.6g} s（观察节点 {hotspot}）"
        )

    def _render_transient(self, solution: TransientSolution) -> None:
        """渲染瞬态结果：显示视图切换并默认末帧变形云图."""
        self._set_view_items((("变形云图（末帧）", "field"), ("位移时程曲线", "curve")))
        self._render_transient_deform(solution)

    def _render_transient_deform(self, solution: TransientSolution) -> None:
        """渲染瞬态变形云图（时程动画）与时程摘要."""
        mesh = solution.mesh
        shape = (mesh.n_nodes, mesh.dofs_per_node)
        frames = [
            (solution.displacements[:, i].reshape(shape), f"t = {solution.times[i]:.4g} s")
            for i in range(solution.times.size)
        ]
        if len(frames) > 1:
            self._start_anim(mesh, frames, first_index=len(frames) - 1)  # 结果态默认末帧
        else:
            self._render_deformation(mesh, solution.displacements[:, -1].reshape(shape))
        max_u = float(np.max(np.abs(solution.displacements[:, -1])))
        self._summary.setText(
            f"末帧最大位移 |u| = {max_u * self._disp_factor():.6g} {self._unit} · "
            f"时程：{solution.n_steps} 步 · dt = {solution.dt:.4g}"
            f" · 总时长 = {solution.times[-1]:.6g} s"
        )

    def _render_transient_curve(self, solution: TransientSolution) -> None:
        """渲染位移时程曲线（末端观察点 uy 随时间变化）并标注峰值."""
        mesh = solution.mesh
        node = tip_node(mesh)
        uy = solution.node_history(node, 1) * self._disp_factor()
        times = solution.times
        self._stop_anim()  # 曲线视图无动画帧
        self._colorbar.clear()  # 曲线视图无场量

        self._plot.clear()
        self._plot.setAspectLocked(False)
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setLogMode(y=False)
        self._plot.setLabel("bottom", "时间 t (s)")
        self._plot.setLabel("left", f"节点 {node} uy ({self._unit})")
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
            f"峰值 |uy| = {abs(uy[peak_index]):.6g} {self._unit} @ t = {times[peak_index]:.6g} s"
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
