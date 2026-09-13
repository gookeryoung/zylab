"""gui.widgets.dsl_result_view DSL 结果视图测试：curve/table/text/cloud 分发渲染."""

from __future__ import annotations

import pyqtgraph as pg
import pytest

from zylab.flowchart.results import (
    CloudData,
    CurveData,
    CurveSeries,
    TableColumn,
    TableData,
    TextData,
)
from zylab.gui.widgets.dsl_result_view import DslResultView, _format_cell


@pytest.mark.gui
def test_placeholder_before_data(qtbot) -> None:
    """初始占位：标题为空 + 尚未运行提示."""
    view = DslResultView()
    qtbot.addWidget(view)
    assert view._title.text() == ""
    assert view._body is not None
    assert "尚未运行" in view._body.text()


@pytest.mark.gui
def test_render_curve(qtbot) -> None:
    """曲线页：多序列图例 + 轴标签."""
    view = DslResultView()
    qtbot.addWidget(view)
    data = CurveData(
        title="扫参曲线",
        x_label="L",
        y_label="uy",
        series=(
            CurveSeries(name="tip", x=(1.0, 2.0), y=(-0.1, -0.4)),
            CurveSeries(name="energy", x=(1.0, 2.0), y=(2.9, 9.8)),
        ),
    )
    view.set_data(data)
    assert view._title.text() == "扫参曲线"
    assert isinstance(view._body, pg.PlotWidget)
    items = view._body.getPlotItem().listDataItems()
    assert len(items) == 2
    xs, ys = items[0].getData()
    assert list(xs) == [1.0, 2.0] and list(ys) == [-0.1, -0.4]
    assert view._body.getAxis("bottom").labelText == "L"
    assert view._body.getAxis("left").labelText == "uy"


@pytest.mark.gui
def test_render_table(qtbot) -> None:
    """表格页：列标题 + 单元格格式化（浮点 6 位有效数字）."""
    view = DslResultView()
    qtbot.addWidget(view)
    data = TableData(
        title="结果表",
        columns=(TableColumn(title="L"), TableColumn(title="uy")),
        rows=((40.0, -0.241234), (60.0, -0.81)),
    )
    view.set_data(data)
    assert view._title.text() == "结果表"
    table = view._body
    assert table.rowCount() == 2 and table.columnCount() == 2
    assert table.horizontalHeaderItem(0).text() == "L"
    assert table.item(0, 0).text() == "40"
    assert table.item(0, 1).text() == "-0.241234"
    assert table.item(1, 1).text() == "-0.81"
    # 文本值不经浮点格式化
    assert _format_cell("abc") == "abc"
    assert _format_cell(3.0) == "3"


@pytest.mark.gui
def test_render_text_and_replace(qtbot) -> None:
    """文本页 + set_data 连续替换（旧正文销毁）."""
    view = DslResultView()
    qtbot.addWidget(view)
    view.set_data(TextData(title="摘要", text="末端挠度 -0.240 mm"))
    assert view._title.text() == "摘要"
    old = view._body
    view.set_data(TextData(title="摘要2", text="应变能 2.9 J"))
    assert view._body is not old
    assert "应变能" in view._body.text()


@pytest.mark.gui
def test_render_cloud_placeholder(qtbot) -> None:
    """cloud 声明显示路由占位提示."""
    view = DslResultView()
    qtbot.addWidget(view)
    view.set_data(CloudData(title="云图", node_id="solve"))
    assert view._title.text() == "云图"
    assert "solve" in view._body.text()


def test_build_curve_widget_log_and_peak(qtbot) -> None:
    """build_curve_widget 覆盖对数轴、mark_peak 极值标注分支."""
    from zylab.flowchart.results import CurveData, CurveSeries
    from zylab.gui.widgets.dsl_result_view import build_curve_widget

    curve = CurveData(
        title="对数峰值",
        x_label="ω",
        y_label="|H|",
        log_x=True,
        log_y=True,
        mark_peak=True,
        series=(CurveSeries("S1", (1.0, 2.0, 3.0), (0.1, 1.0, 0.5)),),
        series_styles=({"color": "primary", "dash": "dashed", "width": 3},),
    )
    w = build_curve_widget(curve)
    qtbot.addWidget(w)
    w.show()
    assert w.isVisible()


def test_dsl_curve_has_context_menu(qtbot) -> None:
    """DSL 曲线视图应有完整中文右键菜单（修复：之前完全缺失）."""
    from zylab.flowchart.results import CurveData, CurveSeries
    from zylab.gui.widgets.dsl_result_view import build_curve_widget
    from zylab.gui.widgets.plot_widget import ZyPlotWidget

    curve = CurveData(
        title="菜单测试",
        series=(CurveSeries("s1", (1.0, 2.0), (3.0, 4.0)),),
    )
    w = build_curve_widget(curve)
    qtbot.addWidget(w)
    assert isinstance(w, ZyPlotWidget)
    plot_item = w.getPlotItem()
    # 默认英文菜单已关闭
    assert plot_item._menuEnabled is False
    assert plot_item.getContextMenus(None) is None
    # 右键菜单已替换为中文菜单
    actions = plot_item.vb.menu.actions()
    texts = [a.text() for a in actions if a.text()]
    assert "恢复默认视角" in texts
    assert "复制图像" in texts
    assert any(t.startswith("导出图像 (PNG)") for t in texts)
    assert any(t.startswith("导出数据 (CSV)") for t in texts)
    assert any(t == "轴自适应" for t in texts)
    # 场景级英文菜单已清空
    assert w.scene().contextMenu == []


def test_dsl_curve_csv_export(tmp_path) -> None:
    """DSL 曲线 CSV 导出：多序列共用第一个序列 x 列."""
    from zylab.flowchart.results import CurveData, CurveSeries
    from zylab.gui.widgets.dsl_result_view import _export_curve_csv

    curve = CurveData(
        title="CSV 导出测试",
        series=(
            CurveSeries("accel", (0.0, 1.0, 2.0), (1.0, 4.0, 9.0)),
            CurveSeries("vel", (0.0, 1.0, 2.0), (0.0, 2.0, 4.0)),
        ),
    )
    path = str(tmp_path / "test.csv")
    _export_curve_csv(curve, path)
    lines = tmp_path.joinpath("test.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "x,accel,vel"
    assert lines[1].strip() == "0.0,1.0,0.0"
    assert lines[2].strip() == "1.0,4.0,2.0"
    assert lines[3].strip() == "2.0,9.0,4.0"


def test_build_curve_widget_no_series_no_legend(qtbot) -> None:
    """空序列 build_curve_widget 跳过图例创建（if data.series 分支）."""
    from zylab.flowchart.results import CurveData
    from zylab.gui.widgets.dsl_result_view import build_curve_widget

    curve = CurveData(title="空序列曲线", series=())
    w = build_curve_widget(curve)
    qtbot.addWidget(w)
    # 空序列时 plotItem.legend 仍为 None（未调用 addLegend）
    assert w.getPlotItem().legend is None


def test_build_curve_widget_dotted_dash(qtbot) -> None:
    """series_styles 的 dash='dotted' 分支覆盖."""
    from zylab.flowchart.results import CurveData, CurveSeries
    from zylab.gui.widgets.dsl_result_view import build_curve_widget

    curve = CurveData(
        title="dotted 测试",
        series=(CurveSeries("s1", (1.0, 2.0), (3.0, 4.0)),),
        series_styles=({"color": "primary", "dash": "dotted", "width": 2},),
    )
    w = build_curve_widget(curve)
    qtbot.addWidget(w)
    items = w.getPlotItem().listDataItems()
    assert len(items) == 1


def test_dsl_curve_csv_empty_series(tmp_path) -> None:
    """空序列 CSV 导出：仅表头，无数据行."""
    from zylab.flowchart.results import CurveData
    from zylab.gui.widgets.dsl_result_view import _export_curve_csv

    curve = CurveData(title="空序列", series=())
    path = str(tmp_path / "empty.csv")
    _export_curve_csv(curve, path)
    lines = tmp_path.joinpath("empty.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "x"
