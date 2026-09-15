"""最终冲刺：覆盖 param_form.py line 72 + simple_heatmap.py lines 140,157."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def test_param_form_non_numeric_skip(qapp) -> None:
    """param_form.py line 72: param_type 非 numeric 时 continue."""

    from zylab.gui.widgets.param_form import ParamForm

    param_spec = SimpleNamespace(param_type="str")
    node_spec = SimpleNamespace(params=[param_spec], param=lambda k: param_spec)
    node = SimpleNamespace(spec=node_spec, id="n1", name="Node1", params={})
    graph = SimpleNamespace(node=lambda nid: node, nodes=lambda: [node])
    groups = (SimpleNamespace(title="Test", params=("n1.myparam",)),)

    form = ParamForm()
    form.set_graph(graph, groups)
    # STR 被跳过 → _fields 应空
    assert form._fields == {}


def test_simple_heatmap_equal_values(qapp) -> None:
    """simple_heatmap.py line 140: vmin == vmax 分支."""
    from zylab.gui.widgets.simple_heatmap import SimpleHeatmap

    w = SimpleHeatmap()
    # 所有值相同 → 触发 vmin==vmax 分支
    data = np.ones((3, 3)) * 42.0
    w.set_data(data)
    w.resize(200, 200)
    w.show()
    w.repaint()


def test_simple_heatmap_cm_with_zero_gap(qapp, monkeypatch) -> None:
    """simple_heatmap.py line 157: colormap 有相同 t 控制点."""
    from zylab.gui.widgets.simple_heatmap import SimpleHeatmap

    w = SimpleHeatmap()
    # 设一个 t1 == t0 的 colormap（触发 if t1 != t0 的 else 分支）
    w._colormap = [(0.0, "#000000"), (0.5, "#808080"), (0.5, "#808080"), (1.0, "#ffffff")]
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    w.set_data(data)
    w.resize(200, 200)
    w.show()
    w.repaint()
