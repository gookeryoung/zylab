"""gui.pages.fea_page 分析页测试."""

from __future__ import annotations

import numpy as np
import pytest

from zylab.fea import (
    Section,
    solve_buckling,
    solve_harmonic,
    solve_modal,
    solve_nonlinear_static,
    solve_static,
)
from zylab.gui.pages.fea_page import (
    FeaPage,
    build_cantilever_case,
    build_cantilever_mesh,
)


@pytest.mark.gui
def test_fea_page_builds(qtbot) -> None:
    """分析页应完成布局装配并渲染初始线框."""
    page = FeaPage()
    qtbot.addWidget(page)
    assert page._model_combo.count() == 3
    assert page._analysis_combo.count() == 5
    assert page._solve_button.isEnabled()
    assert page._plot.getPlotItem().listDataItems()  # 初始未变形线框已渲染


@pytest.mark.gui
def test_fea_page_panel_width_stable(qtbot) -> None:
    """求解渲染后左面板宽度不得被绘图区挤压（表单防变形回归）."""
    page = FeaPage()
    qtbot.addWidget(page)
    page.resize(1280, 800)
    scroll = page._panel_scroll
    width_before = scroll.width()
    assert scroll.minimumWidth() >= 300
    mesh = build_cantilever_mesh()
    materials = [page.current_material]
    sections = [Section(thickness=page._thickness_spin.value())]
    case = build_cantilever_case(mesh)
    page._on_finished(solve_static(mesh, materials, sections, case))
    assert scroll.width() == width_before


@pytest.mark.gui
def test_fea_page_inputs_never_squashed(qtbot) -> None:
    """小窗口下求解后输入框几何不得被压缩（纵向防变形回归）.

    求解使结果标签增高，若面板无滚动区承载会压缩表单行
    （输入框高度被压小、行距漂移）。
    """
    page = FeaPage()
    qtbot.addWidget(page)
    page.resize(740, 480)  # 模拟小窗口/高 DPI 有效高度不足
    spin = page._young_spin
    geometry_before = (spin.height(), spin.width())

    mesh = build_cantilever_mesh()
    sections = [Section(thickness=page._thickness_spin.value())]
    case = build_cantilever_case(mesh)
    page._on_finished(solve_static(mesh, [page.current_material], sections, case))
    assert (spin.height(), spin.width()) == geometry_before


@pytest.mark.gui
def test_fea_page_param_visibility_associated(qtbot) -> None:
    """参数行按分析类型关联显示：静力隐藏动力学参数，谐响应全部显示."""
    page = FeaPage()
    qtbot.addWidget(page)
    # 静力（默认）：动力学/扫频参数隐藏
    assert not page._density_spin.isVisibleTo(page._params_form.parentWidget())
    assert not page._fmax_spin.isVisibleTo(page._params_form.parentWidget())
    assert page._young_spin.isVisibleTo(page._params_form.parentWidget())
    # 切到谐响应：密度与扫频参数显示
    page._analysis_combo.setCurrentIndex(2)
    assert page._density_spin.isVisibleTo(page._params_form.parentWidget())
    assert page._fmax_spin.isVisibleTo(page._params_form.parentWidget())
    # 切到屈曲：模态阶数显示、扫频参数隐藏
    page._analysis_combo.setCurrentIndex(3)
    assert page._n_modes_spin.isVisibleTo(page._params_form.parentWidget())
    assert not page._fmax_spin.isVisibleTo(page._params_form.parentWidget())
    # 悬臂柱模型：连续体参数（泊松比/厚度）隐藏
    page._model_combo.setCurrentIndex(1)
    assert not page._thickness_spin.isVisibleTo(page._params_form.parentWidget())


@pytest.mark.gui
def test_fea_page_renders_solution(qtbot) -> None:
    """求解完成后应渲染云图并更新结果摘要."""
    page = FeaPage()
    qtbot.addWidget(page)

    mesh = build_cantilever_mesh()
    materials = [page.current_material]
    sections = [Section(thickness=page._thickness_spin.value())]
    case = build_cantilever_case(mesh)
    solution = solve_static(mesh, materials, sections, case)

    page._on_finished(solution)
    assert page._solution is solution
    assert "最大位移" in page._result_label.text()
    # 变形线框 + 云图散点应已加入绘图区
    items = page._plot.getPlotItem().listDataItems()
    assert len(items) >= 2


@pytest.mark.gui
def test_fea_page_modal_renders(qtbot) -> None:
    """模态结果应填充频率表、渲染首阶振型并支持振型切换."""
    page = FeaPage()
    qtbot.addWidget(page)

    mesh = build_cantilever_mesh()
    materials = [page.current_material]
    sections = [Section(thickness=page._thickness_spin.value())]
    case = build_cantilever_case(mesh)
    solution = solve_modal(mesh, materials, sections, case.constraints, n_modes=4)

    page._on_finished(solution)
    assert page._modal_solution is solution
    assert page._solution is None
    assert page._freq_table.isVisibleTo(page)
    assert page._freq_table.rowCount() == 4
    assert "基频" in page._result_label.text()
    # 频率表内容：第 1 阶 ω 数值与解一致
    assert float(page._freq_table.item(0, 1).text()) == pytest.approx(solution.frequencies[0], rel=1e-3)
    # 首阶振型云图已渲染
    assert len(page._plot.getPlotItem().listDataItems()) >= 2
    # 切换到第 2 阶振型
    page._mode_spin.setValue(2)
    assert page._mode_spin.value() == 2


@pytest.mark.gui
def test_fea_page_modal_solve_end_to_end(qtbot) -> None:
    """端到端：模态分析经进程执行器完成并渲染（真实 spawn 子进程）."""
    page = FeaPage()
    qtbot.addWidget(page)
    page._analysis_combo.setCurrentIndex(1)  # 切到模态
    page._on_solve()
    with qtbot.waitSignal(page._bridge.finished, timeout=120_000) as blocker:
        pass
    solution = blocker.args[0]
    assert solution.n_modes == page._n_modes_spin.value()
    assert solution.frequencies[0] > 0.0
    assert page._freq_table.rowCount() == solution.n_modes
    page.shutdown()


@pytest.mark.gui
def test_fea_page_harmonic_renders(qtbot) -> None:
    """谐响应结果应渲染频响曲线并标注峰值."""
    page = FeaPage()
    qtbot.addWidget(page)

    mesh = build_cantilever_mesh()
    materials = [page.current_material]
    sections = [Section(thickness=page._thickness_spin.value())]
    case = build_cantilever_case(mesh)
    solution = solve_harmonic(mesh, materials, sections, case, np.linspace(0.0, 3.0, 30), alpha=0.1)

    page._on_finished(solution)
    assert page._harmonic_solution is solution
    assert page._solution is None
    assert "峰值" in page._result_label.text()
    # 频响曲线 + 峰值散点已加入绘图区
    items = page._plot.getPlotItem().listDataItems()
    assert len(items) >= 2
    # 静力结果路径应恢复云图视图（log 轴关闭）
    static = solve_static(mesh, materials, sections, case)
    page._on_finished(static)
    assert page._solution is static
    assert page._harmonic_solution is None


@pytest.mark.gui
def test_fea_page_harmonic_solve_end_to_end(qtbot) -> None:
    """端到端：谐响应经进程执行器完成并渲染（真实 spawn 子进程）."""
    page = FeaPage()
    qtbot.addWidget(page)
    page._analysis_combo.setCurrentIndex(2)  # 切到谐响应
    page._n_freq_spin.setValue(20)  # 减少频率点加速测试
    page._on_solve()
    with qtbot.waitSignal(page._bridge.finished, timeout=120_000) as blocker:
        pass
    solution = blocker.args[0]
    assert solution.n_frequencies == 20
    assert "峰值" in page._result_label.text()
    page.shutdown()


@pytest.mark.gui
def test_fea_page_model_switch_to_column(qtbot) -> None:
    """切换到悬臂柱模型：信息标签更新并重绘初始线框."""
    page = FeaPage()
    qtbot.addWidget(page)
    page._model_combo.setCurrentIndex(1)
    assert "BEAM2" in page._model_info.text()
    mesh, _materials, sections, case = page._current_model()
    assert mesh.n_nodes == 21
    assert sections[0].inertia == 1.0e-4
    # 柱工况：底部固支 + 顶部压缩（y 负向）
    assert case.constraints[0].dofs == (0, 1, 2)
    assert case.loads[0].forces[1] < 0.0


@pytest.mark.gui
def test_fea_page_buckling_renders(qtbot) -> None:
    """屈曲结果应填充载荷因子表、渲染首阶屈曲模态并支持切换."""
    page = FeaPage()
    qtbot.addWidget(page)
    page._model_combo.setCurrentIndex(1)  # 悬臂柱

    mesh, materials, sections, case = page._current_model()
    solution = solve_buckling(mesh, materials, sections, case, n_modes=3)

    page._on_finished(solution)
    assert page._buckling_solution is solution
    assert page._freq_table.isVisibleTo(page)
    assert page._freq_table.rowCount() == 3
    assert "载荷因子" in page._result_label.text()
    # 一阶因子与欧拉解析解一致（表内数值）
    euler = np.pi**2 * materials[0].e_modulus * sections[0].inertia / (4.0 * 10.0**2)
    assert float(page._freq_table.item(0, 1).text()) == pytest.approx(euler, rel=0.01)
    # 首阶屈曲模态已渲染，可切换
    assert len(page._plot.getPlotItem().listDataItems()) >= 1
    page._mode_spin.setValue(2)
    assert page._mode_spin.value() == 2


@pytest.mark.gui
def test_fea_page_buckling_solve_end_to_end(qtbot) -> None:
    """端到端：屈曲分析经进程执行器完成并渲染（真实 spawn 子进程）."""
    page = FeaPage()
    qtbot.addWidget(page)
    page._model_combo.setCurrentIndex(1)  # 悬臂柱
    page._analysis_combo.setCurrentIndex(3)  # 切到屈曲
    page._on_solve()
    with qtbot.waitSignal(page._bridge.finished, timeout=120_000) as blocker:
        pass
    solution = blocker.args[0]
    assert solution.n_modes == page._n_modes_spin.value()
    assert solution.load_factors[0] > 0.0
    assert page._freq_table.rowCount() == solution.n_modes
    page.shutdown()


@pytest.mark.gui
def test_fea_page_model_switch_to_truss(qtbot) -> None:
    """切换到两杆桁架模型：信息标签更新、截面为面积型、工况为顶点集中力."""
    page = FeaPage()
    qtbot.addWidget(page)
    page._model_combo.setCurrentIndex(2)
    assert "TRUSS2" in page._model_info.text()
    mesh, _materials, sections, case = page._current_model()
    assert mesh.n_nodes == 3
    assert sections[0].area == 1.0
    # 桁架工况：两支座双向固支 + 顶点向下集中力
    assert len(case.constraints) == 2
    assert case.loads[0].node == 1
    assert case.loads[0].forces[1] < 0.0


@pytest.mark.gui
def test_fea_page_nonlinear_renders(qtbot) -> None:
    """非线性结果应渲染变形云图并显示迭代历程摘要."""
    page = FeaPage()
    qtbot.addWidget(page)
    page._model_combo.setCurrentIndex(2)  # 两杆桁架

    mesh, materials, sections, case = page._current_model()
    solution = solve_nonlinear_static(mesh, materials, sections, case, n_increments=5)

    page._on_finished(solution)
    assert page._nonlinear_solution is solution
    assert page._freq_table.isHidden()
    assert page._mode_spin.isHidden()
    assert "Newton 迭代" in page._result_label.text()
    assert "增量步" in page._result_label.text()
    # 大位移已渲染（变形线框 + 散点）
    assert len(page._plot.getPlotItem().listDataItems()) >= 1
    # 两杆桁架浅拱：顶点竖向位移为大位移量级（非线性效应显著）
    apex_vy = float(solution.displacements[1, 1])
    assert abs(apex_vy) > 0.05


@pytest.mark.gui
def test_fea_page_nonlinear_curve_view(qtbot) -> None:
    """非线性视图切换：载荷-位移曲线与变形云图互切，其他结果隐藏切换框."""
    page = FeaPage()
    qtbot.addWidget(page)
    page._model_combo.setCurrentIndex(2)

    mesh, materials, sections, case = page._current_model()
    solution = solve_nonlinear_static(mesh, materials, sections, case, n_increments=4)
    page._on_finished(solution)

    # 默认变形云图，切换框可见
    assert page._nonlinear_view_combo.isVisibleTo(page)
    assert "Newton 迭代" in page._result_label.text()

    # 切到载荷-位移曲线：两条曲线（非线性 + 线性参照）
    page._nonlinear_view_combo.setCurrentIndex(1)
    assert "载荷-位移曲线" in page._result_label.text()
    curves = page._plot.getPlotItem().listDataItems()
    assert len(curves) == 2
    factors, uy = curves[1].getData()
    np.testing.assert_allclose(factors, solution.history_factors)
    np.testing.assert_allclose(uy, solution.history_dof(1, 1))

    # 切回变形云图
    page._nonlinear_view_combo.setCurrentIndex(0)
    assert "Newton 迭代" in page._result_label.text()

    # 非线性之外的结果类型隐藏切换框
    mesh2 = build_cantilever_mesh()
    page._on_finished(
        solve_static(mesh2, [page.current_material], [Section(thickness=1.0)], build_cantilever_case(mesh2))
    )
    assert page._nonlinear_view_combo.isHidden()


@pytest.mark.gui
def test_fea_page_nonlinear_solve_end_to_end(qtbot) -> None:
    """端到端：几何非线性经进程执行器完成并渲染（真实 spawn 子进程）."""
    page = FeaPage()
    qtbot.addWidget(page)
    page._model_combo.setCurrentIndex(2)  # 两杆桁架
    page._analysis_combo.setCurrentIndex(4)  # 切到几何非线性
    page._increments_spin.setValue(5)
    page._on_solve()
    with qtbot.waitSignal(page._bridge.finished, timeout=120_000) as blocker:
        pass
    solution = blocker.args[0]
    assert solution.converged
    assert solution.load_factor == 1.0
    assert "Newton 迭代" in page._result_label.text()
    page.shutdown()


@pytest.mark.gui
def test_fea_page_failed_message(qtbot) -> None:
    """求解失败应显示错误并恢复按钮可用."""
    page = FeaPage()
    qtbot.addWidget(page)
    page._solve_button.setEnabled(False)
    page._on_failed("SolverError: 刚度矩阵奇异")
    assert "奇异" in page._result_label.text()
    assert page._solve_button.isEnabled()
    assert page._status_label.text() == "求解失败"


@pytest.mark.gui
def test_fea_page_progress_updates(qtbot) -> None:
    """进度回调应更新进度条与状态文本."""
    page = FeaPage()
    qtbot.addWidget(page)
    page._on_progress(0.6, "求解线性方程组")
    assert page._progress.value() == 60
    assert page._status_label.text() == "求解线性方程组"


@pytest.mark.gui
def test_fea_page_solve_end_to_end(qtbot) -> None:
    """端到端：点击求解经进程执行器完成并渲染（真实 spawn 子进程）."""
    page = FeaPage()
    qtbot.addWidget(page)
    page._on_solve()
    with qtbot.waitSignal(page._bridge.finished, timeout=120_000) as blocker:
        pass
    solution = blocker.args[0]
    assert solution is not None
    assert solution.strain_energy > 0.0
    assert page._status_label.text() == "求解完成"
    assert page._solve_button.isEnabled()
    page.shutdown()
