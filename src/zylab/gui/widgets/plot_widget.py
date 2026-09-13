"""zylab 统一绘图组件：预配置 pyqtgraph PlotWidget + 可组合中文右键菜单.

解决三处独立创建 PlotWidget（笔记本 / DSL 参数化计算 / FEA 结果视图）
的重复建设问题，并修复 DSL 曲线完全缺失右键菜单（暴露 pyqtgraph 默认
英文菜单）的用户体验缺陷。

设计原则：

- **PlotWidget 继承**：封装背景/网格/轴色/图例的统一初始化逻辑，
  三处调用方不再各自重复写约 15 行相同代码；
- **菜单配置化**：通过 :class:`PlotMenuConfig` 数据类按场景灵活裁剪
  菜单项（恢复视角 / 复制图像 / 导出 PNG / 导出 CSV / 笔记本专属回调
  / 轴自适应子菜单），避免条件分支散落各处；
- **回调注入**：导出 CSV 等场景依赖调用方特定逻辑（ResultView 用
  ``fea.export_csv``，DSL 用 ``CurveData.series`` 转 CSV），通过
  Callable 注入而非硬依赖，保持统一组件轻量；
- **主题联动**：提供 :meth:`refresh_theme` 接口，背景/轴色/图例文字
  随全局主题切换重刷。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pyqtgraph as pg

from zylab.sci.palettes import PG_CURVE_DEFAULTS

from .. import theme
from ..qt_compat import QApplication, QFileDialog, QMenu, QWidget

__all__ = ["PlotMenuConfig", "ZyPlotWidget", "apply_plot_context_menu"]


# ------------------------------------------------------------------ 模块级辅助函数（供普通 pg.PlotWidget 复用）


def _copy_widget_plot(plot: QWidget) -> None:
    """任意 PlotWidget 截图复制到剪贴板（非 ZyPlotWidget 场景的通用实现）."""
    QApplication.clipboard().setPixmap(plot.grab())


def _export_widget_png(plot: QWidget) -> None:
    """任意 PlotWidget 导出 PNG（非 ZyPlotWidget 场景的通用实现）."""
    path_str, _ = QFileDialog.getSaveFileName(plot, "导出图像 (PNG)", "plot.png", "PNG 图片 (*.png)")
    if not path_str:
        return
    plot.grab().save(path_str)


def _export_widget_csv(plot: QWidget, export_fn: Callable[[str], None]) -> None:
    """任意 PlotWidget 导出 CSV（通用壳 + 调用方注入回调）."""
    path_str, _ = QFileDialog.getSaveFileName(plot, "导出数据 (CSV)", "data.csv", "CSV 文件 (*.csv)")
    if not path_str:
        return
    export_fn(path_str)


def apply_plot_context_menu(plot: pg.PlotWidget, config: PlotMenuConfig) -> QMenu:
    """按配置构建中文右键菜单并绑定到任意 PlotWidget（不限于 ZyPlotWidget）.

    供普通 ``pg.PlotWidget``（如 ResultView 的云图视图）复用，统一菜单项与
    pyqtgraph 默认英文菜单的替换逻辑。

    :param plot: 目标 PlotWidget（必须有 getPlotItem() 和 scene() 方法）。
    :param config: 菜单项开关 + 回调注入。
    :returns: 构建好的 QMenu（已绑定到 ViewBox.vb.menu）。
    """
    plot_item = plot.getPlotItem()
    # 1. 关闭 pyqtgraph 默认菜单（Plot Options / Average / Downsampling 等英文项）
    plot_item.setMenuEnabled(False, enableViewBoxMenu=None)
    # 2. 场景级菜单（GraphicsScene 内置 "Export..." 英文项）整体置空，杜绝漏网英文
    plot.scene().contextMenu = []

    menu = QMenu(plot)

    # -- 恢复默认视角 --
    if config.show_reset:
        menu.addAction("恢复默认视角", plot_item.autoRange)
        menu.addSeparator()

    # -- 复制图像 + 导出 PNG（通用，调用 plot.grab()） --
    if config.show_copy:
        menu.addAction("复制图像", lambda: _copy_widget_plot(plot))
    if config.show_export_png:
        menu.addAction("导出图像 (PNG)...", lambda: _export_widget_png(plot))

    # -- 导出 CSV（回调注入，仅当提供时启用） --
    if config.export_csv_fn is not None:
        menu.addAction("导出数据 (CSV)...", lambda: _export_widget_csv(plot, config.export_csv_fn))

    # -- 分隔线（导出项与轴自适应分隔） --
    if config.show_copy or config.show_export_png or config.export_csv_fn is not None:
        menu.addSeparator()

    # -- 轴自适应子菜单（精细范围控制） --
    if config.show_range_submenu:
        range_menu = menu.addMenu("轴自适应")
        range_menu.addAction("X 轴自适应", lambda: plot_item.vb.enableAutoRange(x=True, y=False))
        range_menu.addAction("Y 轴自适应", lambda: plot_item.vb.enableAutoRange(x=False, y=True))
        range_menu.addAction("XY 轴同时自适应", lambda: plot_item.vb.enableAutoRange(x=True, y=True))
        menu.addSeparator()

    # -- 笔记本专属 --
    if config.notebook_clear_output_fn is not None:
        menu.addAction("清除本格输出", config.notebook_clear_output_fn)

    # 绑定到 ViewBox（右键交互入口）
    plot_item.vb.menu = menu
    return menu


#: 笔记本右键菜单（清除本格输出）的特殊操作类型标记.
_NOTEBOOK_CLEAR = "__notebook_clear_output__"


@dataclass
class PlotMenuConfig:
    """右键菜单可配置项（按场景组合裁剪，布尔开关 + 回调注入）.

    默认值覆盖最通用场景（恢复视角 / 复制 / 导出PNG / 轴自适应）；
    ResultView 额外提供 ``export_csv_fn``；DSL 提供 ``export_csv_from_series``；
    笔记本注入 ``notebook_clear_output_fn``。

    :param show_reset: 显示"恢复默认视角"。
    :param show_copy: 显示"复制图像"（到剪贴板）。
    :param show_export_png: 显示"导出图像 (PNG)..."。
    :param show_range_submenu: 显示"轴自适应"子菜单（X/Y/XY 独立切换）。
    :param export_csv_fn: 导出 CSV 的回调 ``(path: str) -> None``。
        提供即启用"导出数据 (CSV)..."，为 ``None`` 时隐藏该项。
    :param notebook_clear_output_fn: 笔记本专属：清除本格输出的回调。
        提供即启用"清除本格输出"。
    """

    show_reset: bool = True
    show_copy: bool = True
    show_export_png: bool = True
    show_range_submenu: bool = True
    export_csv_fn: Callable[[str], None] | None = None
    notebook_clear_output_fn: Callable[[], None] | None = None


class ZyPlotWidget(pg.PlotWidget):
    """zylab 统一绘图组件：预配置背景/网格/轴色/图例 + 可组合右键菜单.

    初始化流程（幂等，主题变更后可重刷）：

    1. 背景 = ``theme.current_palette().bg_app``；
    2. 网格 = ``showGrid(x, y, PG_CURVE_DEFAULTS["grid_alpha"])``；
    3. 轴线 = ``border_strong`` + ``PG_CURVE_DEFAULTS["axis_width"]``；
    4. 禁用 pyqtgraph 默认菜单（PlotItem 菜单 + Scene contextMenu 全置空）；
    5. 等待 :meth:`setup_context_menu` 注入中文菜单配置。

    使用示例（笔记本内嵌绘图）::

        plot = ZyPlotWidget(parent=self, show_legend=True)
        plot.addLegend(offset=(8, 8))  # 调用方继续叠加业务逻辑
        plot.setup_context_menu(PlotMenuConfig(
            notebook_clear_output_fn=self._clear_cell_output,
        ))
        plot.plot(x, y, pen=...)
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        log_x: bool = False,
        log_y: bool = False,
    ) -> None:
        """初始化统一 PlotWidget（不调用 setup_context_menu，等待外部注入）.

        :param parent: 父控件。
        :param log_x: 是否启用 X 轴对数刻度。
        :param log_y: 是否启用 Y 轴对数刻度。
        """
        super().__init__(parent=parent, background=theme.current_palette().bg_app)
        self._log_x = log_x
        self._log_y = log_y
        self._setup_common()
        if log_x or log_y:
            self.plotItem.setLogMode(x=log_x, y=log_y)

    # ------------------------------------------------------------------ 公共接口

    def setup_context_menu(self, config: PlotMenuConfig) -> None:
        """按配置构建中文右键菜单（替换 pyqtgraph 默认英文菜单）.

        :param config: 菜单项开关 + 回调注入。
        """
        apply_plot_context_menu(self, config)

    def refresh_theme(self) -> None:
        """主题切换后重刷背景/轴色（网格透明度不变，颜色随主题轴色）.

        调用时机：全局 theme 变更（笔记本页、ResultView 等都有
        ``refresh_theme`` 入口统一触发）。
        """
        self.setBackground(theme.current_palette().bg_app)
        pal = theme.current_palette()
        defaults = PG_CURVE_DEFAULTS
        axis_pen = pg.mkPen(pal.border_strong, width=defaults["axis_width"])
        plot_item = self.getPlotItem()
        for axis_name in ("bottom", "left"):
            plot_item.getAxis(axis_name).setPen(axis_pen)
            plot_item.getAxis(axis_name).setTextPen(pg.mkPen(pal.text_primary, width=1))
        # 图例背景 + 文字色（若已创建）
        legend = plot_item.legend
        if legend is not None:
            legend.setLabelTextColor(pal.text_primary)

    # ------------------------------------------------------------------ 内部辅助

    def _setup_common(self) -> None:
        """统一初始化（背景/网格/轴色）.

        三处调用方原来各自重复约 15 行相同代码，收拢于此。
        """
        pal = theme.current_palette()
        defaults = PG_CURVE_DEFAULTS
        self.setBackground(pal.bg_app)
        self.showGrid(x=True, y=True, alpha=defaults["grid_alpha"])
        plot_item = self.getPlotItem()
        axis_pen = pg.mkPen(pal.border_strong, width=defaults["axis_width"])
        for axis_name in ("bottom", "left"):
            plot_item.getAxis(axis_name).setPen(axis_pen)
            plot_item.getAxis(axis_name).setTextPen(pg.mkPen(pal.text_primary, width=1))
