"""gui.widgets.dsl_result_view DSL 结果视图测试：curve/table/text/cloud 分发渲染."""

from __future__ import annotations

import pyqtgraph as pg
import pytest

from zylab.gui.widgets.dsl_result_view import DslResultView, _format_cell
from zylab.studio.results import (
    CloudData,
    CurveData,
    CurveSeries,
    TableData,
    TextData,
)


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
    data = TableData(title="结果表", columns=("L", "uy"), rows=((40.0, -0.241234), (60.0, -0.81)))
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
