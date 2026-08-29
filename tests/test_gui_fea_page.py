"""gui.pages.fea_page 分析页测试."""

from __future__ import annotations

import pytest

from zylab.fea import Section, solve_static
from zylab.gui.pages.fea_page import FeaPage, build_cantilever_case, build_cantilever_mesh


@pytest.mark.gui
def test_fea_page_builds(qtbot) -> None:
    """分析页应完成布局装配并渲染初始线框."""
    page = FeaPage()
    qtbot.addWidget(page)
    assert page._model_combo.count() == 1
    assert page._solve_button.isEnabled()
    assert page._plot.getPlotItem().listDataItems()  # 初始未变形线框已渲染


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
