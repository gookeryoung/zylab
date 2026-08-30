"""gui.widgets.result_view 结果视图测试：各解类型渲染分发与控制条联动."""

from __future__ import annotations

import pytest

from zylab.gui.widgets.result_view import ResultView
from zylab.studio import nodes


def _beam_bundle():
    """小尺寸悬臂梁模型."""
    return nodes.build_cantilever({}, {"nx": 8, "ny": 2})


@pytest.mark.gui
def test_show_mesh_preview(qtbot) -> None:
    """模型预览：线框 + 规模摘要."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    view.show_mesh(_beam_bundle())
    assert "模型预览" in view._summary.text()
    assert "27 节点" in view._summary.text()
    assert view._plot.getPlotItem().listDataItems()


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
    assert not view._freq_table.isVisible()


@pytest.mark.gui
def test_show_modal_solution_and_mode_switch(qtbot) -> None:
    """模态解：频率表 + 振型切换."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    solution = nodes.run_modal({"model": _beam_bundle()}, {"n_modes": 3})
    view.show_solution(solution)
    assert view._freq_table.isVisible()
    assert view._freq_table.rowCount() == 3
    assert view._mode_spin.isVisible()
    assert "基频" in view._summary.text()
    view._mode_spin.setValue(2)  # 切换第二阶振型
    assert view._plot.getPlotItem().listDataItems()


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
    """导出行：有解时可见，清空后隐藏."""
    view = ResultView()
    qtbot.addWidget(view)
    view.show()
    assert not view._export_row.isVisible()
    view.show_solution(nodes.run_static({"model": _beam_bundle()}, {}))
    assert view._export_row.isVisible()
    view.clear()
    assert not view._export_row.isVisible()


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
