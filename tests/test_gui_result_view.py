"""gui.widgets.result_view 结果视图测试：各解类型渲染分发与控制条联动."""

from __future__ import annotations

import numpy as np
import pytest

from zylab.gui.widgets.result_view import ColorBarWidget, ResultView
from zylab.studio import nodes


def _beam_bundle():
    """小尺寸悬臂梁模型."""
    return nodes.build_cantilever({}, {"nx": 8, "ny": 2})


@pytest.mark.gui
def test_plot_context_menu_replaced(qtbot) -> None:
    """云图右键菜单替换为精简中文菜单（默认英文菜单整体关闭）."""
    view = ResultView()
    qtbot.addWidget(view)
    plot_item = view._plot.getPlotItem()
    assert plot_item._menuEnabled is False  # Plot Options/Average/Downsampling 等不再出现
    assert plot_item.getContextMenus(None) is None
    texts = [a.text() for a in plot_item.vb.menu.actions()]
    assert texts == ["恢复默认视角", "复制图像", "导出图像 (PNG)", "导出数据 (CSV)"]


@pytest.mark.gui
def test_show_mesh_preview(qtbot) -> None:
    """模型预览：线框 + 节点散点 + 规模摘要."""
    import pyqtgraph as pg

    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    view.show_mesh(_beam_bundle())
    assert "模型预览" in view._summary.text()
    assert "27 节点" in view._summary.text()
    assert view._plot.getPlotItem().listDataItems()
    # 节点散点：杆系模型一条线时节点分布仍清晰可辨
    scatters = [it for it in view._plot.getPlotItem().items if isinstance(it, pg.ScatterPlotItem)]
    assert len(scatters) == 1
    xs, ys = scatters[0].getData()
    assert len(xs) == 27 and len(ys) == 27


@pytest.mark.gui
def test_show_static_solution(qtbot) -> None:
    """静力解：变形云图 + 位移/应变能摘要."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    bundle = _beam_bundle()
    view.show_solution(nodes.run_static({"model": bundle}, {}))
    assert "最大位移" in view._summary.text()
    assert "应变能" in view._summary.text()
    assert view._tabs.indexOf(view._data_page) < 0  # 无数据子页


@pytest.mark.gui
def test_show_modal_solution_and_mode_switch(qtbot) -> None:
    """模态解：频率表 + 振型切换."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    solution = nodes.run_modal({"model": _beam_bundle()}, {"n_modes": 3})
    view.show_solution(solution)
    assert view._tabs.indexOf(view._data_page) >= 0  # 数据子页已建
    assert view._tabs.tabBar().isVisible()  # 多页时页签条可见
    assert view._freq_table.rowCount() == 3
    assert view._mode_spin.isVisible()
    assert "基频" in view._summary.text()
    view._mode_spin.setValue(2)  # 切换第二阶振型
    assert view._plot.getPlotItem().listDataItems()
    assert view._freq_table.currentRow() == 1  # 数据页同步高亮当前阶


@pytest.mark.gui
def test_show_buckling_solution(qtbot) -> None:
    """屈曲解：载荷因子表（临界载荷 = λ × 参考载荷）."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    bundle = nodes.build_column({}, {})
    solution = nodes.run_buckling({"model": bundle}, {"n_modes": 2})
    view.show_solution(solution, reference_load=2.0)
    assert view._freq_table.horizontalHeaderItem(1).text() == "载荷因子 λ"
    critical = float(view._freq_table.item(0, 2).text())
    assert critical == pytest.approx(float(solution.load_factors[0]) * 2.0, rel=1e-3)
    assert "λ₁" in view._summary.text()


@pytest.mark.gui
def test_show_harmonic_solution(qtbot) -> None:
    """谐响应解：频响曲线（对数幅值轴）+ 峰值摘要."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    solution = nodes.run_harmonic({"model": _beam_bundle()}, {"f_max": 2.0, "n_freq": 10})
    view.show_solution(solution)
    assert "峰值" in view._summary.text()
    assert view._plot.getPlotItem().getAxis("left").logMode


@pytest.mark.gui
def test_show_nonlinear_solution_and_view_switch(qtbot) -> None:
    """非线性解：视图切换（变形云图 <-> 载荷-位移曲线）."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    bundle = nodes.build_truss({}, {})
    solution = nodes.run_nonlinear({"model": bundle}, {"n_increments": 4})
    view.show_solution(solution)
    assert view._view_combo.isVisible()
    assert "Newton" in view._summary.text()
    view._view_combo.setCurrentIndex(1)  # 曲线视图
    assert "载荷-位移曲线" in view._summary.text()
    assert not view._plot.getPlotItem().getAxis("left").logMode
    view._view_combo.setCurrentIndex(0)  # 切回云图
    assert "Newton" in view._summary.text()


@pytest.mark.gui
def test_show_transient_solution_and_view_switch(qtbot) -> None:
    """瞬态解：视图切换（末帧变形云图 <-> 位移时程曲线）."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    solution = nodes.run_transient(
        {"model": _beam_bundle()},
        {"duration": 4.0, "n_steps": 40, "alpha": 0.5},
    )
    view.show_solution(solution)
    assert view._view_combo.isVisible()
    assert view._view_combo.itemText(1) == "位移时程曲线"
    assert "时程" in view._summary.text()
    view._view_combo.setCurrentIndex(1)  # 时程曲线视图
    assert "峰值" in view._summary.text()
    assert not view._plot.getPlotItem().getAxis("left").logMode
    view._view_combo.setCurrentIndex(0)  # 切回末帧云图
    assert "时程" in view._summary.text()


@pytest.mark.gui
def test_show_electrothermal_solution_and_view_switch(qtbot) -> None:
    """电-热耦合解：温度/电压云图切换 + 峰值温度与总电功率摘要."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    bundle = nodes.build_joule_plate({}, {"nx": 4, "ny": 2})
    view.show_mesh(bundle)
    assert "模型预览" in view._summary.text()
    solution = nodes.run_electrothermal({"model": bundle}, {})
    view.show_solution(solution)
    assert view._view_combo.isVisible()
    assert view._view_combo.itemText(0) == "温度云图"
    assert view._view_combo.itemText(1) == "电压云图"
    assert "T_max" in view._summary.text()
    assert "总电功率" in view._summary.text()
    view._view_combo.setCurrentIndex(1)  # 电压云图
    assert view._plot.getPlotItem().listDataItems()
    assert not view._plot.getPlotItem().getAxis("left").logMode
    view._view_combo.setCurrentIndex(0)  # 切回温度云图
    assert view._plot.getPlotItem().listDataItems()
    assert not view._plot.getPlotItem().getAxis("left").logMode


@pytest.mark.gui
def test_show_vfilm_3d_preview_and_rotation(qtbot) -> None:
    """V 形薄膜 3D 预览：等轴测投影模式 + 视角旋转重投影不崩溃."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    bundle = nodes.build_vfilm_resistor({}, {"n_lead": 2, "n_diag": 3, "n_width": 2, "n_sub": 1})
    view.show_mesh(bundle)
    assert view._is3d  # 3D 坐标进入投影模式
    assert "拖拽旋转视角" in view._summary.text()
    view._azim = 120.0  # 模拟拖拽旋转后的新视角
    view._elev = -30.0
    view._rerender_3d()  # 预览态重投影（走 show_mesh）
    assert view._is3d
    assert view._plot.getPlotItem().listDataItems()


@pytest.mark.gui
def test_show_et_transient_views_and_scalar_anim(qtbot) -> None:
    """瞬态电-热解：三视图 + 温度云图标量帧动画（3D 深度排序散点）."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    bundle = nodes.build_vfilm_resistor({}, {"n_lead": 2, "n_diag": 3, "n_width": 2, "n_sub": 1})
    solution = nodes.run_electrothermal_transient({"model": bundle}, {"n_steps": 3})
    view.show_solution(solution)
    assert view._view_combo.count() == 3
    assert view._view_combo.itemText(0) == "温度云图"
    assert view._view_combo.itemText(1) == "电压云图"
    assert view._view_combo.itemText(2) == "温度时程曲线"
    # 标量帧动画：位移帧互斥清空、默认末帧、控制行显示
    assert view._scalar_frames and not view._frames
    assert len(view._scalar_frames) == 4  # 步数 + 1（含 t=0 初始帧）
    assert view._frame_index == 3
    assert view._frame_slider.isVisible()
    assert view._frame_label.text().startswith("t = ")
    assert view._colorbar.isVisible()  # 温度场绑定标尺
    view._on_play_toggled()  # 末帧回绕，从首帧播放
    assert view._anim_timer.isActive()
    assert view._frame_index == 0
    view._on_anim_tick()  # 标量帧推进
    assert view._frame_index == 1
    view._on_play_toggled()  # 暂停
    assert not view._anim_timer.isActive()
    view._frame_slider.setValue(3)  # 拖回末帧
    assert view._frame_index == 3
    view._view_combo.setCurrentIndex(1)  # 电压云图（静态场）
    assert view._plot.getPlotItem().listDataItems()
    assert not view._frame_slider.isVisible()
    view._view_combo.setCurrentIndex(2)  # 温度时程曲线
    assert "热点峰值" in view._summary.text()
    assert not view._colorbar.isVisible()
    view._view_combo.setCurrentIndex(0)  # 切回温度云图动画
    assert view._frame_slider.isVisible()
    assert len(view._scalar_frames) == 4


@pytest.mark.gui
def test_show_error_and_clear(qtbot) -> None:
    """错误着色与清空."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    view.show_error("SolverError: 矩阵奇异")
    assert view._summary.objectName() == "errorText"
    view.clear()
    assert view._summary.objectName() == "resultText"
    assert "尚未求解" in view._summary.text()


@pytest.mark.gui
def test_show_unknown_result_type(qtbot) -> None:
    """未知结果类型兜底提示."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    view.show_solution(object())
    assert "未知结果类型" in view._summary.text()


@pytest.mark.gui
def test_refresh_theme_no_crash(qtbot) -> None:
    """主题刷新不抛异常."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    view.refresh_theme()


@pytest.mark.gui
def test_export_row_visibility(qtbot) -> None:
    """导出按钮：有解时可见，清空后隐藏."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    assert not view._export_csv_btn.isVisible()
    view.show_solution(nodes.run_static({"model": _beam_bundle()}, {}))
    assert view._export_csv_btn.isVisible()
    assert view._export_png_btn.isVisible()
    view.clear()
    assert not view._export_csv_btn.isVisible()


@pytest.mark.gui
def test_colorbar_visibility_and_cmap_switch(qtbot) -> None:
    """标尺：云图显示（随场量绑定），曲线隐藏；切换色带重渲染云图."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    assert not view._colorbar.isVisible()  # 模型预览态无标尺
    view.show_solution(nodes.run_static({"model": _beam_bundle()}, {}))
    assert view._colorbar.isVisible()
    view._cmap_combo.setCurrentIndex(1)  # 切换色带（Viridis）
    assert view._colorbar.isVisible()  # 云图重渲染后标尺仍在
    assert "最大位移" in view._summary.text()
    assert view._cmap == "viridis"
    view.clear()
    assert not view._colorbar.isVisible()


@pytest.mark.gui
def test_colorbar_hidden_on_curve_view(qtbot) -> None:
    """非线性解切到曲线视图时标尺隐藏，切回云图恢复."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    bundle = nodes.build_truss({}, {})
    view.show_solution(nodes.run_nonlinear({"model": bundle}, {"n_increments": 4}))
    assert view._colorbar.isVisible()
    view._view_combo.setCurrentIndex(1)  # 载荷-位移曲线
    assert not view._colorbar.isVisible()
    view._view_combo.setCurrentIndex(0)  # 切回变形云图
    assert view._colorbar.isVisible()


@pytest.mark.gui
def test_colorbar_field_range(qtbot) -> None:
    """标尺刻度绑定场值范围：顶=最大、底=最小，空场隐藏."""
    from zylab.gui.widgets.result_view import ColorBarWidget

    bar = ColorBarWidget()
    qtbot.addWidget(bar)
    assert bar.isHidden()
    bar.set_field(np.array([1.0, 2.0, 3.0]), "jet")
    assert not bar.isHidden()
    assert bar._vmin == pytest.approx(1.0)
    assert bar._vmax == pytest.approx(3.0)
    bar.set_field(np.zeros(0), "jet")  # 空场 -> 隐藏
    assert bar.isHidden()


@pytest.mark.gui
def test_colorbar_width_fits_large_values(qtbot) -> None:
    """大数值刻度（如 3.172e+04）宽度自适应容纳文本：自适应 + 绘制期补偿 + 封顶."""
    from zylab.gui.qt_compat import QFontMetrics
    from zylab.gui.widgets.result_view import ColorBarWidget

    bar = ColorBarWidget()
    qtbot.addWidget(bar)
    bar.set_field(np.array([1.0, 2.0]), "jet")  # 短刻度基线
    w_small = bar.width()
    bar.set_field(np.array([0.0, 31720.0]), "jet", "℃")  # 大数值长刻度
    assert bar.width() > w_small  # 自适应放大
    assert bar.width() <= ColorBarWidget._PAD * 3 + ColorBarWidget._BAR_W + ColorBarWidget._MAX_LABEL_W  # 封顶
    # 绘制期补偿（字体度量晚于 set_field 生效时）：以任意字体度量重算宽度，
    # 结果等于容纳全文所需宽度与上限封顶的较小者（与字体环境无关的确定性断言）
    metrics = QFontMetrics(bar.font())
    bar._ensure_width(metrics)
    text_x = ColorBarWidget._PAD + ColorBarWidget._BAR_W + ColorBarWidget._PAD
    widest = max(metrics.horizontalAdvance(t) for t in bar._labels())
    expected = min(text_x + widest + ColorBarWidget._PAD, text_x + ColorBarWidget._MAX_LABEL_W + ColorBarWidget._PAD)
    assert bar.width() == expected
    bar.show()
    assert not bar.grab().isNull()  # 绘制路径正常（含补偿触发的重绘）


@pytest.mark.gui
def test_unit_switch_rerenders(qtbot) -> None:
    """单位切换（m -> mm）：摘要与标尺刻度乘 1000 并带单位后缀."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    static = nodes.run_static({"model": _beam_bundle()}, {})
    view.show_solution(static)
    max_u = float(np.max(np.linalg.norm(static.displacements, axis=1)))
    assert f"{max_u:.6g} m" in view._summary.text()
    view._unit_combo.setCurrentIndex(1)  # mm
    assert view._unit == "mm"
    assert f"{max_u * 1000.0:.6g} mm" in view._summary.text()
    assert view._colorbar._vmax == pytest.approx(max_u * 1000.0, rel=1e-6)
    assert view._colorbar._unit == "mm"


@pytest.mark.gui
def test_anim_frames_nonlinear(qtbot) -> None:
    """非线性云图动画：增量步帧序列、控制行显示、默认末帧、播放推进."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    solution = nodes.run_nonlinear({"model": nodes.build_truss({}, {})}, {"n_increments": 4})
    view.show_solution(solution)
    assert view._frame_slider.isVisible()  # 帧数 = 步数 + 1 > 1（动画控件显示）
    assert view._play_btn.isVisible()
    assert len(view._frames) == 5
    assert view._frame_index == 4  # 结果态默认末帧（收敛态）
    assert view._frame_label.text().startswith("λ = ")
    view._on_play_toggled()  # 播放（末帧回绕，从首帧重新开始）
    assert view._anim_timer.isActive()
    assert view._frame_index == 0
    view._on_anim_tick()  # 手动推进一帧
    assert view._frame_index == 1
    view._on_play_toggled()  # 暂停
    assert not view._anim_timer.isActive()
    view._view_combo.setCurrentIndex(1)  # 曲线视图：动画控件隐藏
    assert not view._frame_slider.isVisible()


@pytest.mark.gui
def test_anim_frames_transient(qtbot) -> None:
    """瞬态云图动画：时程帧序列（标签含 t = ... s）."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    solution = nodes.run_transient(
        {"model": _beam_bundle()},
        {"duration": 4.0, "n_steps": 40, "alpha": 0.5},
    )
    view.show_solution(solution)
    assert view._frame_slider.isVisible()
    assert len(view._frames) == 41  # 步数 + 1
    assert "s" in view._frame_label.text()
    view._frame_slider.setValue(0)  # 拖回首帧
    assert view._frame_index == 0


@pytest.mark.gui
def test_anim_frames_modal_modes(qtbot) -> None:
    """模态振型动画：各阶帧序列，mode_spin 切换联动帧位."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    solution = nodes.run_modal({"model": _beam_bundle()}, {"n_modes": 3})
    view.show_solution(solution)
    assert view._frame_slider.isVisible()
    assert len(view._frames) == 3
    view._mode_spin.setValue(2)  # 切换第二阶振型
    assert view._frame_index == 1
    assert "第 2 阶" in view._frame_label.text()
    view._on_table_row_clicked(0, 0)  # 数据页点击第一行 -> 回到首阶
    assert view._mode_spin.value() == 1
    assert view._frame_index == 0


@pytest.mark.gui
def test_anim_hidden_for_static(qtbot) -> None:
    """静力解单帧：动画控制行不显示."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    view.show_solution(nodes.run_static({"model": _beam_bundle()}, {}))
    assert not view._frame_slider.isVisible()
    assert view._frames == []


@pytest.mark.gui
def test_colorbar_left_of_plot(qtbot) -> None:
    """标尺位于绘图区左侧（Workbench 布局）；限高只作用于标尺，绘图区不限高."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    assert view._plot_row.layout().indexOf(view._colorbar) == 0
    assert view._plot_row.layout().indexOf(view._plot) == 1
    assert view._colorbar.maximumHeight() == ColorBarWidget._MAX_H
    assert view._plot_row.maximumHeight() == 16777215  # Qt 默认无上限
    # 绘图行四周保留呼吸边距（非零边距）
    margins = view._plot_row.layout().contentsMargins()
    assert (margins.left(), margins.top(), margins.right(), margins.bottom()) == (8, 8, 8, 8)
    # 子 TAB 容器位于布局末尾（无尾部弹簧分抢高度），云图页在册
    layout = view.layout()
    assert layout.indexOf(view._tabs) == layout.count() - 1
    assert view._tabs.indexOf(view._plot_row) == 0


@pytest.mark.gui
def test_toolbar_single_row_compact(qtbot) -> None:
    """工具条单行：全部控制（阶数/视图/动画/色带/单位/导出）同住一行，摘要单行化."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    bar = view._toolbar.layout()
    for widget in (
        view._mode_spin,
        view._view_combo,
        view._play_btn,
        view._frame_slider,
        view._frame_label,
        view._cmap_combo,
        view._unit_combo,
        view._export_csv_btn,
        view._export_png_btn,
    ):
        assert bar.indexOf(widget) >= 0  # 均在工具条内（不再各自占行）
    assert not view._summary.wordWrap()  # 摘要单行
    assert view._export_csv_btn.toolTip() == "导出 CSV"  # 图标按钮语义由 tooltip 承载
    assert not view._export_csv_btn.text()  # 无文字


@pytest.mark.gui
def test_data_tab_added_and_removed(qtbot) -> None:
    """数据子页：模态解加入（South 位页签），切换解/清空后移除."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    solution = nodes.run_modal({"model": _beam_bundle()}, {"n_modes": 2})
    view.show_solution(solution)
    assert view._tabs.count() == 2
    assert view._tabs.tabText(1) == "数据"
    assert view._tabs.tabPosition() == view._tabs.South  # Workbench 数据页签居底
    view.show_solution(nodes.run_static({"model": _beam_bundle()}, {}))  # 换静力解
    assert view._tabs.count() == 1  # 数据页移除
    assert not view._tabs.tabBar().isVisible()  # 单页时页签条隐藏


@pytest.mark.gui
def test_export_csv_writes_file(qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """导出 CSV：对话框路径落盘，摘要提示文件名."""
    path = tmp_path / "out.csv"
    monkeypatch.setattr(
        "zylab.gui.widgets.result_view.QFileDialog.getSaveFileName",
        lambda *_a, **_kw: (str(path), ""),
    )
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    view.show_solution(nodes.run_static({"model": _beam_bundle()}, {}))
    view._on_export_csv()
    assert path.exists()
    assert "结果已导出" in view._summary.text()


@pytest.mark.gui
def test_export_csv_failure_shows_message(qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """导出失败：异常信息显示在摘要."""
    path = tmp_path / "bad.csv"

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise ValueError("不支持导出的结果类型")

    monkeypatch.setattr(
        "zylab.gui.widgets.result_view.QFileDialog.getSaveFileName",
        lambda *_a, **_kw: (str(path), ""),
    )
    monkeypatch.setattr("zylab.gui.widgets.result_view.export_csv", _raise)
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    view.show_solution(nodes.run_static({"model": _beam_bundle()}, {}))
    view._on_export_csv()
    assert "导出失败" in view._summary.text()
    assert not path.exists()


@pytest.mark.gui
def test_export_csv_dialog_cancelled(qtbot, monkeypatch: pytest.MonkeyPatch) -> None:
    """对话框取消（空路径）：不导出不改摘要."""
    monkeypatch.setattr(
        "zylab.gui.widgets.result_view.QFileDialog.getSaveFileName",
        lambda *_a, **_kw: ("", ""),
    )
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    view.show_solution(nodes.run_static({"model": _beam_bundle()}, {}))
    view._on_export_csv()
    assert "最大位移" in view._summary.text()  # 摘要未被覆盖


@pytest.mark.gui
def test_export_png_writes_file(qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """导出 PNG：绘图区截图落盘，摘要提示文件名."""
    path = tmp_path / "shot.png"
    monkeypatch.setattr(
        "zylab.gui.widgets.result_view.QFileDialog.getSaveFileName",
        lambda *_a, **_kw: (str(path), ""),
    )
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    view.show_solution(nodes.run_static({"model": _beam_bundle()}, {}))
    view._on_export_png()
    assert path.exists()
    assert path.stat().st_size > 0
    assert "图片已导出" in view._summary.text()
