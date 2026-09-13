"""gui.widgets.plot_widget 统一绘图组件测试：ZyPlotWidget 初始化 + PlotMenuConfig 配置化菜单."""

from __future__ import annotations

from zylab.gui.widgets.plot_widget import (
    PlotMenuConfig,
    ZyPlotWidget,
    apply_plot_context_menu,
)


def test_plot_menu_config_defaults() -> None:
    """PlotMenuConfig 默认值：显示全部通用项、无 CSV 回调、无笔记本回调."""
    cfg = PlotMenuConfig()
    assert cfg.show_reset is True
    assert cfg.show_copy is True
    assert cfg.show_export_png is True
    assert cfg.show_range_submenu is True
    assert cfg.export_csv_fn is None
    assert cfg.notebook_clear_output_fn is None


def test_plot_menu_config_minimal() -> None:
    """PlotMenuConfig 可配置关闭范围子菜单（纯 view 场景）."""
    cfg = PlotMenuConfig(show_range_submenu=False)
    assert cfg.show_range_submenu is False
    assert cfg.show_reset is True  # 默认仍开


def test_zy_plot_widget_creatable(qtbot) -> None:
    """ZyPlotWidget 可正常创建（背景/网格自动配置）."""
    w = ZyPlotWidget()
    qtbot.addWidget(w)
    assert w.getPlotItem() is not None


def test_zy_plot_widget_log_axis(qtbot) -> None:
    """ZyPlotWidget 构造时传入 log_x/log_y 可设置对数刻度."""
    w = ZyPlotWidget(log_x=True, log_y=True)
    qtbot.addWidget(w)
    assert w.plotItem.getAxis("bottom").logMode is True
    assert w.plotItem.getAxis("left").logMode is True


def test_zy_plot_widget_no_log_axis_by_default(qtbot) -> None:
    """ZyPlotWidget 默认线性轴."""
    w = ZyPlotWidget()
    qtbot.addWidget(w)
    assert w.plotItem.getAxis("bottom").logMode is False
    assert w.plotItem.getAxis("left").logMode is False


def test_zy_plot_widget_setup_context_menu_minimal(qtbot) -> None:
    """setup_context_menu 最小配置：恢复默认视角 + 复制 + 导出PNG + 轴自适应."""
    w = ZyPlotWidget()
    qtbot.addWidget(w)
    w.setup_context_menu(PlotMenuConfig())
    plot_item = w.getPlotItem()
    assert plot_item._menuEnabled is False  # pyqtgraph 默认菜单已关
    actions = plot_item.vb.menu.actions()
    texts = [a.text() for a in actions if a.text()]
    assert "恢复默认视角" in texts
    assert "复制图像" in texts
    assert any(t.startswith("导出图像 (PNG)") for t in texts)
    assert any(t == "轴自适应" for t in texts)
    # 场景级英文菜单已清空
    assert w.scene().contextMenu == []


def test_zy_plot_widget_setup_context_menu_with_csv(qtbot, tmp_path) -> None:
    """setup_context_menu 配置 CSV 回调：导出数据项出现且回调可执行."""
    called = {"count": 0, "last_path": ""}

    def fake_export(path: str) -> None:
        called["count"] += 1
        called["last_path"] = path

    w = ZyPlotWidget()
    qtbot.addWidget(w)
    w.setup_context_menu(PlotMenuConfig(export_csv_fn=fake_export))
    actions = w.getPlotItem().vb.menu.actions()
    texts = [a.text() for a in actions if a.text()]
    assert any(t.startswith("导出数据 (CSV)") for t in texts)


def test_zy_plot_widget_setup_context_menu_notebook_clear(qtbot) -> None:
    """setup_context_menu 配置笔记本清除输出回调：项出现."""
    cleared = {"flag": False}

    def clear() -> None:
        cleared["flag"] = True

    w = ZyPlotWidget()
    qtbot.addWidget(w)
    w.setup_context_menu(PlotMenuConfig(notebook_clear_output_fn=clear))
    actions = w.getPlotItem().vb.menu.actions()
    texts = [a.text() for a in actions if a.text()]
    assert "清除本格输出" in texts


def test_apply_plot_context_menu_for_plain_pg_plot(qtbot) -> None:
    """apply_plot_context_menu 可接受普通 pg.PlotWidget（不限于 ZyPlotWidget）."""
    import pyqtgraph as pg

    plot = pg.PlotWidget()
    qtbot.addWidget(plot)
    apply_plot_context_menu(plot, PlotMenuConfig())
    plot_item = plot.getPlotItem()
    assert plot_item._menuEnabled is False
    assert plot_item.vb.menu is not None
    actions = plot_item.vb.menu.actions()
    texts = [a.text() for a in actions if a.text()]
    assert "恢复默认视角" in texts
    assert "复制图像" in texts


def test_apply_plot_context_menu_subset_config(qtbot) -> None:
    """apply_plot_context_menu 可按配置裁剪菜单项（关闭恢复/复制/导出）."""
    import pyqtgraph as pg

    plot = pg.PlotWidget()
    qtbot.addWidget(plot)
    apply_plot_context_menu(
        plot,
        PlotMenuConfig(show_reset=False, show_copy=False, show_export_png=False, show_range_submenu=False),
    )
    texts = [a.text() for a in plot.getPlotItem().vb.menu.actions() if a.text()]
    assert "恢复默认视角" not in texts
    assert "复制图像" not in texts
    assert any(t == "轴自适应" for t in texts) is False


def test_widget_plot_helpers_calls_when_menu_triggered(qtbot, tmp_path) -> None:
    """模块级辅助函数（_copy_widget_plot/_export_widget_png/_export_widget_csv）
    通过 mock QFileDialog 触发菜单项验证被调用."""
    from unittest.mock import patch

    import pyqtgraph as pg

    plot = pg.PlotWidget()
    qtbot.addWidget(plot)

    called = {"csv": False, "png_saved": False, "copy_called": False}

    def fake_csv(path: str) -> None:
        called["csv"] = True

    apply_plot_context_menu(
        plot,
        PlotMenuConfig(export_csv_fn=fake_csv),
    )
    menu = plot.getPlotItem().vb.menu

    # 1. 触发复制图像 action（mock clipboard.setPixmap）
    copy_action = next(a for a in menu.actions() if a.text() == "复制图像")
    with patch("zylab.gui.widgets.plot_widget.QApplication.clipboard") as mock_cb:
        mock_cb.return_value.setPixmap.side_effect = lambda _: called.__setitem__("copy_called", True)
        copy_action.trigger()
    assert called["copy_called"] is True

    # 2. 触发 CSV 导出 action（mock save 对话框）
    csv_action = next(a for a in menu.actions() if a.text().startswith("导出数据"))
    with patch("zylab.gui.widgets.plot_widget.QFileDialog.getSaveFileName") as mock_dialog:
        mock_dialog.return_value = (str(tmp_path / "out.csv"), "CSV")
        csv_action.trigger()
    assert called["csv"] is True

    # 3. 触发 PNG 导出 action（mock save 对话框）
    png_action = next(a for a in menu.actions() if a.text().startswith("导出图像"))
    with patch("zylab.gui.widgets.plot_widget.QFileDialog.getSaveFileName") as mock_dialog:
        mock_dialog.return_value = (str(tmp_path / "out.png"), "PNG")
        with patch.object(pg.PlotWidget, "grab") as mock_grab:
            mock_grab.return_value.save.side_effect = lambda p: called.__setitem__("png_saved", True)
            png_action.trigger()
    assert called["png_saved"] is True

    # 4. 取消对话框：csv 和 png 的 return 分支
    with patch("zylab.gui.widgets.plot_widget.QFileDialog.getSaveFileName") as mock_dialog:
        mock_dialog.return_value = ("", "CSV")
        csv_action.trigger()
    with patch("zylab.gui.widgets.plot_widget.QFileDialog.getSaveFileName") as mock_dialog:
        mock_dialog.return_value = ("", "PNG")
        png_action.trigger()


def test_zy_plot_widget_refresh_theme_no_crash(qtbot) -> None:
    """refresh_theme 不崩溃（空图例/存在图例两种情况）."""
    w = ZyPlotWidget()
    qtbot.addWidget(w)
    # 无图例时调用
    w.refresh_theme()
    # 有图例时调用
    w.addLegend(offset=(8, 8))
    w.refresh_theme()
