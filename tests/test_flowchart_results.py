"""flowchart.results DSL 结果视图数据解析测试."""

from __future__ import annotations

import re

import numpy as np
import pytest

from zylab.flowchart.dsl import DslResult
from zylab.flowchart.errors import TemplateError
from zylab.flowchart.results import (
    CloudData,
    CurveData,
    TableData,
    TextData,
    build_result,
)

__all__ = []


def _outputs() -> dict:
    """扫参节点典型输出载荷（P6/P7 模板的曲线/表格数据源）."""
    return {
        "sweep": {
            "var": "L",
            "values": [40.0, 60.0, 80.0],
            "series": {"tip": [-0.24, -0.81, -1.92], "energy": [2.9, 9.8, 23.3]},
        },
        "tip": -0.24,
    }


def _result(kind: str, spec: dict, rid: str = "r1", title: str = "结果") -> DslResult:
    """构造 DSL 结果声明."""
    return DslResult(id=rid, kind=kind, title=title, spec=spec)


# ------------------------------------------------ curve


def test_curve_single_series() -> None:
    """单序列曲线：x/y 引用路径解析 + 标签透传."""
    result = _result("curve", {"x": "sweep.values", "y": "sweep.series.tip", "x_label": "L", "y_label": "uy"})
    data = build_result(result, _outputs())
    assert isinstance(data, CurveData)
    assert data.title == "结果"
    assert data.x_label == "L"
    assert data.y_label == "uy"
    (series,) = data.series
    assert series.name == "tip"
    assert series.x == (40.0, 60.0, 80.0)
    assert series.y == (-0.24, -0.81, -1.92)


def test_curve_multiple_series() -> None:
    """y 为引用列表时多序列（序列名取引用末段）."""
    result = _result("curve", {"x": "sweep.values", "y": ["sweep.series.tip", "sweep.series.energy"]})
    data = build_result(result, _outputs())
    assert [s.name for s in data.series] == ["tip", "energy"]


def test_curve_numpy_array_payload() -> None:
    """numpy 数组载荷收敛为元组."""
    outputs = {"calc": np.array([1.0, 4.0, 9.0])}
    result = _result("curve", {"x": "calc", "y": "calc"})
    data = build_result(result, outputs)
    assert data.series[0].y == (1.0, 4.0, 9.0)


def test_curve_length_mismatch_rejected() -> None:
    """y 序列长度与 x 不匹配报错."""
    outputs = {"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0]}
    result = _result("curve", {"x": "a", "y": "b"})
    with pytest.raises(TemplateError, match="长度 2 与 x 长度 3 不匹配"):
        build_result(result, outputs)


def test_curve_empty_y_refs_rejected() -> None:
    """y 引用列表为空报错."""
    result = _result("curve", {"x": "sweep.values", "y": []})
    with pytest.raises(TemplateError, match="y 引用为空"):
        build_result(result, _outputs())


def test_curve_missing_node_rejected() -> None:
    """引用未运行节点报错."""
    result = _result("curve", {"x": "missing.values", "y": "sweep.series.tip"})
    with pytest.raises(TemplateError, match="无输出"):
        build_result(result, _outputs())


def test_curve_missing_path_rejected() -> None:
    """路径段不存在报错."""
    result = _result("curve", {"x": "sweep.no_such", "y": "sweep.series.tip"})
    with pytest.raises(TemplateError, match="无法解析"):
        build_result(result, _outputs())


# ------------------------------------------------ table


def test_table_columns_transposed_to_rows() -> None:
    """columns 各列序列转置为行（对象声明 + 引用简写两种形式）."""
    result = _result(
        "table",
        {"columns": [{"title": "长度 L", "ref": "sweep.values"}, "sweep.series.tip"]},
    )
    data = build_result(result, _outputs())
    assert isinstance(data, TableData)
    assert data.column_titles == ("长度 L", "tip")
    assert data.rows == ((40.0, -0.24), (60.0, -0.81), (80.0, -1.92))


def test_table_column_length_mismatch_rejected() -> None:
    """各列长度不一致报错."""
    outputs = {"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0]}
    result = _result("table", {"columns": ["a", "b"]})
    with pytest.raises(TemplateError, match="长度不一致"):
        build_result(result, outputs)


def test_table_empty_columns_rejected() -> None:
    """columns 空列表报错."""
    result = _result("table", {"columns": []})
    with pytest.raises(TemplateError, match="columns"):
        build_result(result, _outputs())


# ------------------------------------------------ text / cloud


def test_text_format_with_bindings() -> None:
    """text 模板 + values 绑定格式化（支持格式规格）."""
    result = _result("text", {"text": "末端挠度 {tip:.3f} mm", "values": {"tip": "tip"}})
    data = build_result(result, _outputs())
    assert isinstance(data, TextData)
    assert data.text == "末端挠度 -0.240 mm"


def test_text_missing_placeholder_rejected() -> None:
    """模板占位符缺绑定报错."""
    result = _result("text", {"text": "{missing} mm", "values": {}})
    with pytest.raises(TemplateError, match="格式化失败"):
        build_result(result, _outputs())


def test_text_non_mapping_values_rejected() -> None:
    """values 非 Mapping 报错."""
    result = _result("text", {"text": "x", "values": ["tip"]})
    with pytest.raises(TemplateError, match="values 应为对象"):
        build_result(result, _outputs())


def test_numpy_scalar_payload_converged() -> None:
    """嵌套 numpy 标量收敛为 Python 内建值（可格式化）."""
    outputs = {"calc": {"tip": np.float64(2.5)}}
    result = _result("text", {"text": "{v}", "values": {"v": "calc.tip"}})
    data = build_result(result, outputs)
    assert data.text == "2.5"
    assert isinstance(data.text, str)


def test_cloud_ref() -> None:
    """cloud 声明产出节点 id 指向（由解算视图渲染）."""
    result = _result("cloud", {"ref": "solve"})
    data = build_result(result, {"solve": object()})
    assert isinstance(data, CloudData)
    assert data.node_id == "solve"
    assert data.payload is not None  # 载荷随视图携带（报告渲染用）


def test_cloud_spec_fields() -> None:
    """cloud 声明的 field/cmap/deform 透传到视图数据."""
    result = _result("cloud", {"ref": "solve", "field": "temperature", "cmap": "inferno", "deform": 50.0})
    data = build_result(result, {"solve": object()})
    assert data.field == "temperature"
    assert data.cmap == "inferno"
    assert data.deform == 50.0


def test_cloud_node_missing_rejected() -> None:
    """cloud 引用未运行节点报错."""
    result = _result("cloud", {"ref": "ghost"})
    with pytest.raises(TemplateError, match="无输出"):
        build_result(result, {})


# ------------------------------------------------ 对象属性与数组索引引用


class _FakeSolution:
    """属性访问引用的假解对象（公开字段 + property）."""

    times = [0.0, 0.1, 0.2]
    displacements = np.array([[0.0, 1.0, 4.0], [0.0, 2.0, 8.0]])

    @property
    def t_max(self) -> float:
        """最高温度（property 引用目标）."""
        return 85.3


def test_reference_object_attribute() -> None:
    """引用路径支持解对象公开属性/property（下划线属性不可达）."""
    result = _result("text", {"text": "最高 {tmax:.1f}", "values": {"tmax": "solve.t_max"}})
    data = build_result(result, {"solve": _FakeSolution()})
    assert data.text == "最高 85.3"


def test_reference_object_attribute_nested() -> None:
    """属性与数组下标混用：末行（负索引）全时程序列."""
    result = _result("curve", {"x": "solve.times", "y": "solve.displacements.-1"})
    data = build_result(result, {"solve": _FakeSolution()})
    assert data.series[0].x == (0.0, 0.1, 0.2)
    assert data.series[0].y == (0.0, 2.0, 8.0)


def test_reference_private_attribute_rejected() -> None:
    """下划线开头属性引用拒绝（防内部属性逃逸）."""
    result = _result("text", {"text": "{v}", "values": {"v": "solve._secret"}})
    with pytest.raises(TemplateError, match="无法解析"):
        build_result(result, {"solve": _FakeSolution()})


def test_reference_ndarray_out_of_range_rejected() -> None:
    """ndarray 数字下标越界报错."""
    result = _result("curve", {"x": "solve.times", "y": "solve.displacements.99"})
    with pytest.raises(TemplateError, match="无法解析"):
        build_result(result, {"solve": _FakeSolution()})


# ------------------------------------------------ DSL 端到端（声明 -> 解析）


def test_build_result_from_dsl_yaml() -> None:
    """DSL 模板 results 声明 + 节点输出端到端解析."""
    from zylab.flowchart.dsl import dsl_from_yaml

    yaml_text = """
meta: {id: t.curve, name: 曲线参数化计算}
params:
  sweep:
    items:
      lo: {value: 0.0}
      hi: {value: 2.0}
pipeline:
  - id: sweep
    type: compute.sweep
    params:
      var: x
      from: "$lo"
      to: "$hi"
      count: 3
      body:
        nodes:
          - id: y
            type: compute.expr
            params: {expr: "x ** 2", vars: {x: "$x"}}
        collect: ["y"]
results:
  - id: curve_y
    kind: curve
    title: 平方曲线
    x: sweep.values
    y: sweep.series.y
    x_label: x
    y_label: x²
"""
    template = dsl_from_yaml(yaml_text)
    from zylab.flowchart.nodes import compute_sweep

    outputs = {"sweep": compute_sweep({}, template.node("sweep").params)}
    (result,) = template.dsl_results
    data = build_result(result, outputs)
    assert data.series[0].x == (0.0, 1.0, 2.0)
    assert data.series[0].y == (0.0, 1.0, 4.0)


@pytest.mark.slow()
def test_build_result_from_reliability_templates() -> None:
    """十个感度试验 DSL 模板端到端：节点执行 + 曲线/文本/表格三视图解析.

    固定种子蒙特卡洛一致性：μ̂（Weibull 为 η̂）落在真值 1.5 容差内、
    曲线概率单调非降且落在 [0, 1]，试验记录表两列等长；实测记录模板
    μ̂ 与 Excel 金标准 3.225 对齐。
    """
    from zylab.flowchart import BUILTIN_TEMPLATES
    from zylab.flowchart.nodes import analyze_updown_records_node, run_sensitivity_test_node

    templates = [t for t in BUILTIN_TEMPLATES if t.discipline == "reliability"]
    assert len(templates) == 10
    for template in templates:
        node = template.node("test")
        params = node.params
        if node.type_id == "reliability.updown_records":
            outputs = {"test": analyze_updown_records_node({}, params)}
        else:
            outputs = {"test": run_sensitivity_test_node({}, params)}
        views = {result.id: build_result(result, outputs) for result in template.dsl_results}

        curve = views["curve"]
        assert isinstance(curve, CurveData)
        probabilities = np.asarray(curve.series[0].y, dtype=float)
        assert np.all(np.diff(probabilities) >= -1.0e-9)
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
        text = views["summary"]
        assert isinstance(text, TextData)
        if template.id == "dsl.sensitivity_updown_records":
            # Excel 金标准：24 发实测记录 μ̂ = 3.225
            mu_hat = float(re.split("[，（]", text.text.partition("μ̂ = ")[2])[0])
            assert mu_hat == pytest.approx(3.225, abs=0.01)
        elif params.get("model") == "weibull":
            # Weibull 参数化：μ 为尺度 η、σ 为形状 k
            assert "η̂" in text.text
            eta_hat = float(text.text.partition("η̂ = ")[2].split("（")[0])
            assert eta_hat == pytest.approx(10.0, abs=1.5)
            k_hat = float(text.text.partition("k̂ = ")[2].split("（")[0])
            assert 1.0 < k_hat < 2.0
        else:
            assert "μ̂" in text.text
            mu_hat = float(re.split("[，（]", text.text.partition("μ̂ = ")[2])[0])
            assert mu_hat == pytest.approx(10.0, abs=1.5)
        if template.id in ("dsl.sensitivity_neyer", "dsl.sensitivity_weibull"):
            # 轮廓似然置信区间：端点有限且下界 < 上界
            assert "95% CI" in text.text
            bounds = text.text.partition("CI [")[2].split("]")[0].split(", ")
            low, high = float(bounds[0]), float(bounds[1])
            assert low < high
        table = views["records"]
        assert isinstance(table, TableData)
        assert table.column_titles == ("刺激量", "响应")
        assert len(table.rows) > 0
        assert all(len(row) == 2 for row in table.rows)
        if template.id in ("dsl.sensitivity_updown", "dsl.sensitivity_updown_records"):
            # 升降法模板：Dixon-Mood 中间参数文本 + 响应点估计表
            params_text = views["dixon_mood_params"]
            assert isinstance(params_text, TextData)
            assert "n = " in params_text.text and "ρ = " in params_text.text
            points = views["response_table"]
            assert isinstance(points, TableData)
            assert points.column_titles == ("响应概率", "刺激量估计", "标准误", "置信下限", "置信上限")
            assert len(points.rows) > 0
            for row in points.rows:
                low, high = float(row[3]), float(row[4])
                assert low < high


def test_text_format_and_style_passthrough() -> None:
    """TextData 透传 DSL 层的 format / style 字段."""
    from zylab.flowchart.dsl import DslResult

    result = DslResult(
        id="r1",
        kind="text",
        title="摘要",
        spec={"text": "ok"},
        format="markdown",
        style="success",
    )
    data = build_result(result, {})
    assert isinstance(data, TextData)
    assert data.format == "markdown"
    assert data.style == "success"


def test_curve_log_and_peak_passthrough() -> None:
    """CurveData 透传 DSL 层的 log_y / mark_peak / series_styles."""
    from zylab.flowchart.dsl import DslResult

    result = DslResult(
        id="c1",
        kind="curve",
        title="曲线",
        spec={
            "x": "sweep.values",
            "y": "sweep.series.tip",
            "log_y": True,
            "mark_peak": True,
            "series": [{"color": "primary"}],
        },
    )
    data = build_result(result, _outputs())
    assert isinstance(data, CurveData)
    assert data.log_y is True
    assert data.mark_peak is True
    assert data.series_styles == ({"color": "primary"},)


def test_table_column_format_and_align_passthrough() -> None:
    """TableColumn 透传 DSL 层的 format / align 列级字段."""
    result = _result(
        "table",
        {
            "columns": [
                {"title": "L", "ref": "sweep.values", "format": ".4g", "align": "right"},
                "sweep.series.tip",
            ],
        },
    )
    data = build_result(result, _outputs())
    assert isinstance(data, TableData)
    assert data.columns[0].title == "L"
    assert data.columns[0].format == ".4g"
    assert data.columns[0].align == "right"
    # 简写字符串列：format/align 空、title 取末段
    assert data.columns[1].title == "tip"
    assert data.columns[1].format == ""
    assert data.column_titles == ("L", "tip")


# ------------------------------------------------ table: list ref / dict ref
def test_table_column_ref_as_list_literal():
    result = _result(
        "table", {"columns": [{"title": "pos", "ref": ["A", "B", "C"]}, {"title": "val", "ref": "sweep.values"}]}
    )
    data = build_result(result, _outputs())
    assert isinstance(data, TableData)
    assert data.column_titles == ("pos", "val")
    assert data.rows == (("A", 40.0), ("B", 60.0), ("C", 80.0))


def test_table_column_entry_direct_list():
    result = _result("table", {"columns": [["X", "Y", "Z"], "sweep.values"]})
    data = build_result(result, _outputs())
    assert isinstance(data, TableData)
    assert data.column_titles == ("列", "values")
    assert data.rows == (("X", 40.0), ("Y", 60.0), ("Z", 80.0))


def test_table_column_ref_as_dict_values():
    outputs = {"angles": {"m_upper": -170.0, "m_lower": -130.0}}
    result = _result(
        "table", {"columns": [{"title": "pos", "ref": ["top", "bot"]}, {"title": "angle", "ref": "angles"}]}
    )
    data = build_result(result, outputs)
    assert isinstance(data, TableData)
    assert len(data.rows) == 2


def test_table_column_ref_list_literal_column_format():
    result = _result("table", {"columns": [{"title": "label", "ref": ["A", "B", "C"]}, "sweep.values"]})
    data = build_result(result, _outputs())
    assert isinstance(data, TableData)
    assert data.columns[0].format == ""
    assert data.columns[0].align == ""


def test_table_direct_list_entry_column_title():
    result = _result("table", {"columns": [["one", "two", "three"], "sweep.values"]})
    data = build_result(result, _outputs())
    assert isinstance(data, TableData)
    assert data.columns[0].title == "列"


def test_resolve_sequence_non_sequence_rejected():
    result = _result("table", {"columns": [{"title": "X", "ref": "tip"}]})
    with pytest.raises(TemplateError, match="应为序列"):
        build_result(result, _outputs())


def test_curve_y_entry_as_mapping_with_custom_name():
    """曲线 y entry 为 Mapping（带 ref/x/name）时正确识别."""
    from zylab.flowchart.results import build_result

    result = _result(
        "curve",
        {
            "x": "disp.x",
            "y": [
                {"ref": "disp.y", "x": "disp.z", "name": "自定义名"},
            ],
        },
    )
    # 需要有 x/y/z 数据的节点输出
    outputs = {
        "disp": {
            "x": [0.0, 1.0, 2.0],
            "y": [10.0, 20.0, 30.0],
            "z": [100.0, 200.0, 300.0],
        }
    }
    data = build_result(result, outputs)
    assert data.series[0].name == "自定义名"
    assert list(data.series[0].x) == [100.0, 200.0, 300.0]  # 用 entry 自定义的 x 源
    assert list(data.series[0].y) == [10.0, 20.0, 30.0]


def test_resolve_input_missing_node_returns_none():
    """resolve_input 对尚未运行的上游节点（不在 outputs）应返回 None."""
    from zylab.flowchart.results import resolve_input

    # 引用 "ghost.model.solution" 但 ghost 不在 outputs
    assert resolve_input("ghost.model.solution", {}) is None
    # 引用存在节点但端口段跳过后下行
    assert resolve_input("model.port.field", {"model": {"field": 42}}) == 42


def test_descend_list_index_access():
    """_descend 对 list/tuple 按数字段取下标."""
    from zylab.flowchart.results import _descend

    data = [10, 20, 30, 40, 50]
    assert _descend(data, "2", "test") == 30
    assert _descend(data, "-1", "test") == 50
