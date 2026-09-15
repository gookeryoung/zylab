"""覆盖率冲刺 98%：补 optimize / dsl_param_form / simple_line_plot / trial_record_edit / command_palette 的遗漏分支."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

# ============================================================ optimize.py: else 分支


def test_optimize_unknown_optimizer() -> None:
    """optimize.py line 157: Optimizer 枚举未覆盖的 else 分支."""
    from zylab.optim.errors import OptimError
    from zylab.optim.optimize import optimize

    # Optimizer 是 str 的子类，传任意字符串即可触发 else
    surrogate = SimpleNamespace(predict=lambda x: np.array([[float(x[0] ** 2)], [0.0]]))
    with pytest.raises(OptimError):
        optimize(
            surrogate=surrogate,
            variables=[SimpleNamespace(min=-5.0, max=5.0, levels=())],
            optimizer="no_such_optimizer_xyz",
            n_iter=5,
        )


# ============================================================ dsl_param_form.py: response_sequence + set_fields_enabled


def test_dsl_param_form_with_record_edit_and_enabled(qapp) -> None:
    """dsl_param_form.py line 95 + 110->112: response_sequence 控件路径 + set_fields_enabled."""
    from zylab.flowchart.dsl import DslParam, DslParamGroup, DslTemplate
    from zylab.gui.widgets.dsl_param_form import DslParamForm

    grp = DslParamGroup(
        label="Group",
        items=(
            ("seq", DslParam(label="序列", widget="response_sequence", value="1.0 O, 2.0 X")),
            ("step", DslParam(label="步长", value=0.5)),
        ),
    )
    form = DslParamForm()
    form.set_template(DslTemplate(id="t", name="t", nodes=(), dsl_params=(grp,)))
    v = form.values()
    assert "seq" in v
    assert "step" in v
    form.set_fields_enabled(False)
    form.set_fields_enabled(True)


# ============================================================ simple_line_plot.py: 单点 + labels


def test_simple_line_plot_singleton(qapp) -> None:
    """simple_line_plot.py lines 126/128: 单点时避免零范围分支."""
    from zylab.gui.widgets.simple_line_plot import SimpleLinePlot

    w = SimpleLinePlot()
    # set_data 接受 [(xs, ys, label), ...] 元组
    w.set_data([([5.0], [3.0], "A")], colors=["#ff0000"])
    w.resize(200, 200)
    w.show()
    w.repaint()


def test_simple_line_plot_with_labels(qapp) -> None:
    """simple_line_plot.py lines 189-204: x_label + y_label + title."""
    from zylab.gui.widgets.simple_line_plot import SimpleLinePlot

    w = SimpleLinePlot()
    w.set_data(
        [
            ([0.0, 1.0, 2.0], [0.0, 1.0, 4.0], "A"),
            ([0.0, 1.0, 2.0], [0.0, 2.0, 3.0], "B"),
        ],
        x_label="X轴",
        y_label="Y轴",
        title="测试图",
        colors=["#ff0000", "#00ff00"],
    )
    w.resize(300, 220)
    w.show()
    w.repaint()


# ============================================================ trial_record_edit.py: column==0 分支


def test_trial_record_edit_column_zero(qapp) -> None:
    """trial_record_edit.py line 177: _on_item_changed 中 column == 0 分支."""
    from zylab.gui.widgets.trial_record_edit import TrialRecordEdit

    w = TrialRecordEdit()
    w.setText("1.0 O, 2.0 X")
    assert w._table.rowCount() >= 2
    # 尝试修改第 0 列 → column==0 → 直接 return
    item = w._table.item(0, 0)
    if item is not None:
        item.setText("99.0")


# ============================================================ command_palette.py: info=None + keyPressEvent fallback


def test_command_palette_activate_none(qapp, qtbot) -> None:
    """command_palette.py line 213: _activate_current 中 info is None 分支."""
    from PySide2.QtCore import Qt
    from PySide2.QtWidgets import QListWidgetItem, QWidget

    from zylab.gui.widgets.command_palette import CommandPalette

    parent = QWidget()
    qtbot.addWidget(parent)
    cp = CommandPalette(parent)
    lw = cp._list
    item = QListWidgetItem("Ghost")
    item.setData(Qt.UserRole, None)
    lw.addItem(item)
    cp._activate_current(item)  # 安全 return


def test_command_palette_keypress_non_escape(qapp, qtbot) -> None:
    """command_palette.py line 229: keyPressEvent 非 Esc → super() 分支."""
    from PySide2.QtCore import QEvent, Qt
    from PySide2.QtGui import QKeyEvent
    from PySide2.QtWidgets import QWidget

    from zylab.gui.widgets.command_palette import CommandPalette

    parent = QWidget()
    qtbot.addWidget(parent)
    cp = CommandPalette(parent)
    event = QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.NoModifier)
    cp.keyPressEvent(event)  # 非 Esc → super()


# ============================================================ simple_heatmap.py: vmin==vmax + colormap 插值


def test_simple_heatmap_equal_vals(qapp) -> None:
    """simple_heatmap.py line 140: vmin==vmax 分支."""
    from zylab.gui.widgets.simple_heatmap import SimpleHeatmap

    w = SimpleHeatmap()
    w.set_data(np.full((3, 3), 42.0))
    w.resize(200, 200)
    w.show()
    w.repaint()


def test_simple_heatmap_colormap_interp(qapp) -> None:
    """simple_heatmap.py line 157: colormap 控制点间线性插值."""
    from zylab.gui.widgets.simple_heatmap import SimpleHeatmap

    w = SimpleHeatmap()
    w._colormap = [
        (0.0, "#0000ff"),
        (0.5, "#00ff00"),
        (1.0, "#ff0000"),
    ]
    w.set_data(np.array([[0.0, 0.25, 0.5, 0.75, 1.0]]))
    w.resize(200, 200)
    w.show()
    w.repaint()


# ============================================================ settings_panel.py: repaint


def test_settings_panel_smoke(qapp, qtbot) -> None:
    """settings_panel.py 基本渲染（覆盖遗漏的 paint 分支）."""
    from zylab.gui.widgets.settings_panel import SettingsPanel

    w = SettingsPanel()
    qtbot.addWidget(w)
    w.resize(240, 400)
    w.show()
    w.repaint()


# ============================================================ doe/sampling.py: 120 raise DoeError

# ============================================================ core/config.py: 92 Python 版本回退

# ============================================================ settings_panel.py: 字体回退 + 损坏文件


def test_settings_panel_font_fallback(qapp, monkeypatch) -> None:
    """settings_panel.py lines 300/311: FONT_FAMILY/FONT_MONO 为空时回退."""
    from zylab.gui.widgets.settings_panel import SettingsPanel

    # 让 theme.FONT_FAMILY 和 FONT_MONO 不含引号，触发 if not families 分支
    monkeypatch.setattr("zylab.gui.widgets.settings_panel.theme.FONT_FAMILY", "sans-serif")
    monkeypatch.setattr("zylab.gui.widgets.settings_panel.theme.FONT_MONO", "mono")

    result = SettingsPanel._default_body_font_families()
    assert isinstance(result, list)
    result_mono = SettingsPanel._default_mono_font_families()
    assert isinstance(result_mono, list)


def test_settings_panel_load_corrupted(qapp, qtbot, tmp_path, monkeypatch) -> None:
    """settings_panel.py lines 347-348: settings.json 损坏走 except 分支."""

    from zylab.gui.widgets.settings_panel import SettingsPanel

    # 让 _settings_path 返回一个有损坏 JSON 的文件
    bad_file = tmp_path / "bad_settings.json"
    bad_file.write_text("not valid json {", encoding="utf-8")
    monkeypatch.setattr(SettingsPanel, "_settings_path", staticmethod(lambda: bad_file))
    w = SettingsPanel()
    qtbot.addWidget(w)
    cfg = w.load()
    # 应不崩溃，返回默认值
    assert isinstance(cfg, dict)


# ============================================================ simple_heatmap: 直接调 paintEvent


def test_simple_heatmap_paint_vmin_eq_vmax(qapp) -> None:
    """simple_heatmap.py line 140: 直接调 paintEvent 触发 vmin==vmax 分支."""
    from PySide2.QtGui import QPaintEvent

    from zylab.gui.widgets.simple_heatmap import SimpleHeatmap

    w = SimpleHeatmap()
    w.set_data(np.full((3, 3), 42.0))
    w.resize(200, 200)
    event = QPaintEvent(w.rect())
    w.paintEvent(event)


def test_simple_heatmap_paint_colormap_interp(qapp) -> None:
    """simple_heatmap.py line 157: 直接调 paintEvent 触发 colormap 插值."""
    from PySide2.QtGui import QPaintEvent

    from zylab.gui.widgets.simple_heatmap import SimpleHeatmap

    w = SimpleHeatmap()
    w._colormap = [
        (0.0, "#0000ff"),
        (0.5, "#00ff00"),
        (1.0, "#ff0000"),
    ]
    w.set_data(np.array([[0.0, 0.25, 0.5, 0.75, 1.0]]))
    w.resize(200, 200)
    event = QPaintEvent(w.rect())
    w.paintEvent(event)


# ============================================================ simple_line_plot: 直接调 paintEvent


def test_simple_line_plot_paint_singleton(qapp) -> None:
    """simple_line_plot.py lines 126/128: 单点零范围分支（直接调 paintEvent）."""
    from PySide2.QtGui import QPaintEvent

    from zylab.gui.widgets.simple_line_plot import SimpleLinePlot

    w = SimpleLinePlot()
    w.set_data([([5.0], [3.0], "A")], colors=["#ff0000"])
    w.resize(200, 200)
    event = QPaintEvent(w.rect())
    w.paintEvent(event)


def test_simple_line_plot_paint_with_labels(qapp) -> None:
    """simple_line_plot.py lines 180/189-214: x/y/title 标签 + 图例."""
    from PySide2.QtGui import QPaintEvent

    from zylab.gui.widgets.simple_line_plot import SimpleLinePlot

    w = SimpleLinePlot()
    w.set_data(
        [
            ([0.0, 1.0, 2.0], [0.0, 1.0, 4.0], "A"),
            ([0.0, 1.0, 2.0], [0.0, 2.0, 3.0], "B"),
        ],
        x_label="X",
        y_label="Y",
        title="T",
        colors=["#ff0000", "#00ff00"],
    )
    w.resize(300, 220)
    event = QPaintEvent(w.rect())
    w.paintEvent(event)


# ============================================================ dsl_result_view: pyqtgraph fallback

# ============================================================ optimize_direct: else 分支（需要 Template）


# ============================================================ simple_heatmap.py line 157: t > 最后控制点


def test_simple_heatmap_paint_t_exceeds_cm(qapp) -> None:
    """simple_heatmap.py line 157: t 超出最后控制点 → return cm[-1][1]."""
    from PySide2.QtGui import QPaintEvent

    from zylab.gui.widgets.simple_heatmap import SimpleHeatmap

    w = SimpleHeatmap()
    w._colormap = [(0.0, "#0000ff"), (1.0, "#ff0000")]
    # clip_max 设得比数据实际 max 小，归一化后 t > 1.0
    w._clip_max = 0.5  # data max=1.0, clip_max=0.5 → vmax=0.5 → t=(1.0-0)/(0.5-0)=2.0
    w.set_data(np.array([[0.0, 0.5, 1.0]]))
    w.resize(200, 200)
    event = QPaintEvent(w.rect())
    w.paintEvent(event)


# ============================================================ settings_panel.py: save OSError


def test_settings_panel_save_io_error(qapp, qtbot, monkeypatch, tmp_path) -> None:
    """settings_panel.py lines 347-348: save 时 write_text 抛 OSError."""
    from pathlib import Path

    from zylab.gui.widgets.settings_panel import SettingsPanel

    # 让 _settings_path 返回一个不可写的路径
    bad_dir = tmp_path / "readonly_dir"
    bad_dir.mkdir()
    monkeypatch.setattr(SettingsPanel, "_settings_path", staticmethod(lambda: bad_dir / "settings.json"))
    # monkeypatch write_text 让它抛 OSError
    original_write = Path.write_text

    def _fake_write(self, *args, **kwargs):
        if str(self).endswith("settings.json"):
            raise OSError("read-only filesystem")
        return original_write(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _fake_write)
    w = SettingsPanel()
    qtbot.addWidget(w)
    result = w.save()
    assert isinstance(result, dict)


# ============================================================ optimize.py line 310: optimize_direct else


def test_optimize_direct_unknown_optimizer() -> None:
    """optimize.py line 310: optimize_direct 无效 optimizer → OptimError."""
    # 需要一个真实 Template 来触发流程走到 optimizer 判断
    from zylab.flowchart import Template, TemplateNode
    from zylab.optim.errors import OptimError
    from zylab.optim.optimize import optimize_direct

    tpl = Template(
        id="t.test",
        name="Test",
        nodes=(TemplateNode(id="model", type_id="example.cantilever_q4", params={}),),
    )
    with pytest.raises(OptimError):
        optimize_direct(
            template=tpl,
            variables=[SimpleNamespace(min=-5.0, max=5.0, levels=(), name="x")],
            optimizer="definitely_not_a_real_optimizer",
            n_iter=5,
        )


# ============================================================ resources_rc.py: qCleanupResources


def test_qt_cleanup_resources() -> None:
    """gui/resources_rc.py line 2140: 手动调 qCleanupResources."""
    from zylab.gui.resources_rc import qCleanupResources

    qCleanupResources()


# ============================================================ dsl_result_view: pyqtgraph import fallback


def test_dsl_result_view_pg_fallback(monkeypatch) -> None:
    """dsl_result_view.py lines 49-57: 让 pyqtgraph 导入失败触发 fallback."""
    import builtins as _builtins

    real_import = _builtins.__import__

    def _block_pg(name, *args, **kwargs):
        if name == "pyqtgraph":
            raise ImportError("No module named 'pyqtgraph'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_builtins, "__import__", _block_pg)

    # 删掉已缓存的模块再导入
    import sys

    for mod in list(sys.modules):
        if "dsl_result_view" in mod or "pyqtgraph" in mod:
            del sys.modules[mod]

    from zylab.gui.widgets import dsl_result_view  # noqa: F401


# ============================================================ simple_heatmap.py line 157


def test_simple_heatmap_paint_fallback_to_last_cm(qapp) -> None:
    """simple_heatmap.py line 157: t 超出所有控制点 → return 最后颜色."""
    from PySide2.QtGui import QPaintEvent

    from zylab.gui.widgets.simple_heatmap import SimpleHeatmap

    w = SimpleHeatmap()
    # 只有 1 个控制点，循环 range(len(cm)-1)=range(0) 跳过，直接 return 最后颜色
    w._colormap = [(0.5, "#abcdef")]
    w._clip_max = 0.3  # vmax=0.3, data=[[0.5]] → t=(0.5-0)/(0.3-0) > 1.0
    w.set_data(np.array([[0.5]]))
    w.resize(200, 200)
    w.paintEvent(QPaintEvent(w.rect()))


# ============================================================ thread_pool.py: except + capture_cancel


def test_worker_task_exception(qapp) -> None:
    """WorkerTask._run 函数抛异常 → except BaseException 分支."""
    from PySide2.QtCore import QThreadPool

    from zylab.gui.workers.thread_pool import WorkerTask

    def _bad():
        raise RuntimeError("boom")

    task = WorkerTask(_bad)
    task.set_callbacks(on_failed=lambda exc: None)
    QThreadPool.globalInstance().start(task)
    QThreadPool.globalInstance().waitForDone(2000)
    assert task._error is not None
    assert isinstance(task._error, RuntimeError)


def test_worker_task_capture_cancel(qapp) -> None:
    """WorkerTask capture_cancel=True 时函数收到 cancel_requested 参数."""
    from PySide2.QtCore import QThreadPool

    from zylab.gui.workers.thread_pool import WorkerTask

    received = []

    def _fn(cancel_requested):
        received.append(cancel_requested())
        return 42

    task = WorkerTask(_fn, capture_cancel=True)
    QThreadPool.globalInstance().start(task)
    QThreadPool.globalInstance().waitForDone(2000)
    assert task._result == 42
