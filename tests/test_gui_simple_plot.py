"""test_gui_simple_plot - SimpleLinePlot / SimpleHeatmap / ThemeController 测试."""

from __future__ import annotations

import os

import numpy as np
import pytest

# 必须在 pytest-qt 创建 QApplication 之前设置
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from zylab.gui import theme
from zylab.gui.controllers import ThemeController
from zylab.gui.widgets.simple_heatmap import SimpleHeatmap, _interp_color
from zylab.gui.widgets.simple_line_plot import SimpleLinePlot

# ---------------------------------------------------------------------------
# ThemeController（QObject，无需 qtbot）
# ---------------------------------------------------------------------------


class TestThemeController:
    """ThemeController 应正确暴露 Palette 全部字段为 Q_PROPERTY."""

    def test_all_palette_fields_exposed(self) -> None:
        """Palette dataclass 的每个字段都应能通过 ThemeController 访问."""
        from dataclasses import fields as dc_fields

        tc = ThemeController()
        pal = theme.current_palette()
        for f in dc_fields(theme.Palette):
            assert hasattr(tc, f.name), f"ThemeController 缺少属性: {f.name}"
            assert getattr(tc, f.name) == getattr(pal, f.name)

    def test_theme_changed_signal_connectable(self) -> None:
        """theme_changed 信号应可连接并在 refresh 时发射."""
        tc = ThemeController()
        received: list[str] = []
        tc.theme_changed.connect(lambda: received.append("fired"))
        tc.refresh()
        assert received == ["fired"]

    def test_refresh_updates_palette(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """refresh() 后属性值应反映当前 Palette."""
        from dataclasses import replace

        tc = ThemeController()
        new_pal = replace(theme.current_palette(), bg_app="#000000", primary="#FFFFFF")
        monkeypatch.setattr(theme, "_current", new_pal)
        tc.refresh()
        assert tc.bg_app == "#000000"
        assert tc.primary == "#FFFFFF"


# ---------------------------------------------------------------------------
# SimpleLinePlot（QWidget，需 qtbot）
# ---------------------------------------------------------------------------


class TestSimpleLinePlot:
    """SimpleLinePlot 公共 API + 绘制应不崩溃."""

    def test_set_data_single_series(self, qtbot) -> None:
        w = SimpleLinePlot()
        qtbot.addWidget(w)
        w.set_data([([0, 1, 2], [0, 1, 4], "curve")], x_label="X", y_label="Y", title="T")
        assert w._title == "T"
        assert len(w._series) == 1

    def test_set_data_multiple_series(self, qtbot) -> None:
        w = SimpleLinePlot()
        qtbot.addWidget(w)
        w.set_data(
            [([0, 1], [0, 1], "a"), ([0, 1], [1, 0], "b")],
            colors=["#FF0000", "#00FF00"],
        )
        assert len(w._series) == 2
        assert w._series[0]["color"] == "#FF0000"

    def test_clear_resets_all(self, qtbot) -> None:
        w = SimpleLinePlot()
        qtbot.addWidget(w)
        w.set_data([([0, 1], [0, 1], "a")], x_label="X", y_label="Y", title="T")
        w.clear()
        assert w._series == []
        assert w._title == ""

    def test_paint_with_data_no_crash(self, qtbot) -> None:
        """有数据时 show + processEvents 应触发 paintEvent 且不崩溃."""
        w = SimpleLinePlot()
        w.resize(300, 200)
        qtbot.addWidget(w)
        w.set_data([([0, 1, 2, 3], [0, 1, 4, 9], "y=x^2")], title="Square")
        w.show()
        qtbot.waitExposed(w, 500)
        w.update()
        qtbot.wait(50)

    def test_paint_empty_no_crash(self, qtbot) -> None:
        """空数据时 paintEvent 应走占位路径且不崩溃."""
        w = SimpleLinePlot()
        w.resize(200, 120)
        qtbot.addWidget(w)
        w.show()
        qtbot.waitExposed(w, 500)
        w.update()
        qtbot.wait(50)


# ---------------------------------------------------------------------------
# SimpleHeatmap（QWidget，需 qtbot）
# ---------------------------------------------------------------------------


class TestSimpleHeatmap:
    """SimpleHeatmap 公共 API + 绘制应不崩溃."""

    def test_set_data_2d_array(self, qtbot) -> None:
        w = SimpleHeatmap()
        qtbot.addWidget(w)
        data = np.arange(12).reshape(3, 4).tolist()
        w.set_data(data, clip_min=0, clip_max=11, title="H")
        assert w._title == "H"
        assert w._data is not None
        assert w._data.shape == (3, 4)

    def test_set_data_with_custom_colormap(self, qtbot) -> None:
        w = SimpleHeatmap()
        qtbot.addWidget(w)
        cm = [(0.0, "#000000"), (0.5, "#888888"), (1.0, "#FFFFFF")]
        w.set_data([[1.0]], colormap=cm)
        assert w._colormap == cm

    def test_clear_resets_all(self, qtbot) -> None:
        w = SimpleHeatmap()
        qtbot.addWidget(w)
        w.set_data([[1.0, 2.0]], title="T")
        w.clear()
        assert w._data is None
        assert w._title == ""

    def test_set_data_empty_input(self, qtbot) -> None:
        w = SimpleHeatmap()
        qtbot.addWidget(w)
        w.set_data([])
        assert w._data is None
        w.set_data([[]])
        assert w._data is None

    def test_paint_with_data_no_crash(self, qtbot) -> None:
        """有数据时 show + processEvents 应触发 paintEvent 且不崩溃."""
        w = SimpleHeatmap()
        w.resize(300, 200)
        qtbot.addWidget(w)
        w.set_data(np.random.rand(5, 6).tolist(), show_values=True, title="Random")
        w.show()
        qtbot.waitExposed(w, 500)
        w.update()
        qtbot.wait(50)

    def test_paint_empty_no_crash(self, qtbot) -> None:
        """空数据时 paintEvent 应走占位路径且不崩溃."""
        w = SimpleHeatmap()
        w.resize(200, 150)
        qtbot.addWidget(w)
        w.show()
        qtbot.waitExposed(w, 500)
        w.update()
        qtbot.wait(50)

    def test_interp_color_midpoint(self) -> None:
        """_interp_color 中点应为两色平均（整数舍入允许 1 偏差）."""
        from PySide2.QtGui import QColor

        c0 = QColor(255, 0, 0)
        c1 = QColor(0, 0, 255)
        mid = _interp_color(c0, c1, 0.5)
        assert abs(mid.red() - 128) <= 1
        assert abs(mid.blue() - 128) <= 1
        assert mid.green() == 0
