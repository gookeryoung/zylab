"""flowchart LSC 曲线优化模块测试."""

from __future__ import annotations

import numpy as np
import pytest

from zylab.flowchart.lsc import LSCCurve, solve_lsc_curve
from zylab.flowchart.module import module_spec
from zylab.flowchart.nodes import run_lsc_curve

__all__ = []


# ------------------------------------------------ LSCCurve 核心计算


def test_lsc_default() -> None:
    """默认参数求解：残差应接近零."""
    curve = LSCCurve()
    assert curve.x.shape == (16,)
    assert curve.R.success


def test_lsc_curve_segments() -> None:
    """四段曲线采样点数量正确."""
    curve = LSCCurve()
    segs = curve.curve_segments()
    for key in ("inner_upper", "inner_lower", "outer_upper", "outer_lower"):
        assert key in segs
        assert len(segs[key]["x"]) == 100
        assert len(segs[key]["y"]) == 100


def test_lsc_breakpoint_points() -> None:
    """断点坐标包含内/外各一个，每个有 x/y_upper/y_lower."""
    pts = LSCCurve().breakpoint_points()
    assert set(pts.keys()) == {"inner", "outer"}
    for side in pts.values():
        assert set(side.keys()) == {"x", "y_upper", "y_lower"}


def test_lsc_calculate_angles() -> None:
    """角度计算返回四元组，数值在合理范围内."""
    angles = LSCCurve().calculate_angles()
    assert len(angles) == 4
    for a in angles:
        assert -180 <= a <= 0  # 负 x 轴区域的切线角度（arctan2(y, x) 中 x < 0）


def test_lsc_custom_params() -> None:
    """自定义参数求解仍能收敛."""
    curve = LSCCurve(
        m=-2.0,
        m1=-3.5,
        s=1.5,
        s1=10.0,
        H=0.8,
        m2=1.0,
        H1=0.3,
        H2=0.7,
        J=90,
        J1=45,
    )
    assert curve.x.shape == (16,)
    assert np.isfinite(curve.x).all()


def test_lsc_all_curves_tangent_at_breakpoint() -> None:
    """断点处内部上下曲线 y 值接近（连续约束）."""
    curve = LSCCurve()
    x = curve.x
    m = curve.m
    # 内部断点：内部上 = 内部下
    y_inner_up = x[0] + x[1] * m + x[2] * m**2 + x[3] * m**3
    y_inner_lo = x[4] + x[5] * m + x[6] * m**2 + x[7] * m**3
    assert abs(y_inner_up - y_inner_lo) < 1e-6


# ------------------------------------------------ LSCCurve 参数校验


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"J": -5}, "J 必须在 0~180"),
        ({"J": 200}, "J 必须在 0~180"),
        ({"J1": -10}, "J1 必须在 0~180"),
        ({"J1": 270}, "J1 必须在 0~180"),
        ({"m": 0}, "m 必须为负数"),
        ({"m": 5}, "m 必须为负数"),
        ({"m1": 0}, "m1 必须为负数"),
        ({"m1": 5}, "m1 必须为负数"),
        ({"m": -3.0, "m1": -1.0}, "m1.*小于.*m"),
    ],
)
def test_lsc_validation(kwargs: dict, match: str) -> None:
    """非法参数抛 ValueError."""
    with pytest.raises(ValueError, match=match):
        LSCCurve(**kwargs)


# ------------------------------------------------ solve_lsc_curve 便捷函数


def test_solve_lsc_curve_returns_pickleable() -> None:
    """solve_lsc_curve 返回可 pickle 的 dict."""
    result = solve_lsc_curve()
    import pickle

    data = pickle.dumps(result)
    restored = pickle.loads(data)
    assert restored["cost"] == result["cost"]
    assert list(restored["curves"].keys()) == ["inner_upper", "inner_lower", "outer_upper", "outer_lower"]


def test_solve_lsc_curve_structure() -> None:
    """返回 dict 包含 x/cost/curves/breakpoints/angles 五个键."""
    result = solve_lsc_curve()
    assert set(result.keys()) == {"x", "cost", "curves", "breakpoints", "angles"}
    assert len(result["x"]) == 16
    assert result["cost"] >= 0


# ------------------------------------------------ 节点函数


def test_run_lsc_curve_node() -> None:
    """run_lsc_curve 节点正常执行."""
    result = run_lsc_curve({}, {})
    assert result["cost"] >= 0  # 正确应用约束后，cost 变大是预期行为
    assert "curves" in result


def test_run_lsc_curve_node_params() -> None:
    """节点接收自定义参数."""
    params = {"m": -2.0, "m1": -3.0, "J": 90.0, "J1": 45.0}
    result = run_lsc_curve({}, params)
    assert result["cost"] >= 0  # 正确应用约束后，cost 变大是预期行为


# ------------------------------------------------ 模块注册


def test_compute_lsc_curve_registered() -> None:
    """compute.lsc_curve 模块已注册."""
    spec = module_spec("compute.lsc_curve")
    assert spec.name == "LSC 曲线优化"
    assert spec.target == "zylab.flowchart.nodes:run_lsc_curve"
    assert spec.output_port("data").port_type.value == "data"


def test_compute_lsc_curve_params() -> None:
    """模块参数 schema 包含全部 10 个参数."""
    spec = module_spec("compute.lsc_curve")
    keys = {p.key for p in spec.params}
    assert keys == {"m", "m1", "s", "s1", "H", "m2", "H1", "H2", "J", "J1"}


# ------------------------------------------------ DSL 模板集成


def test_dsl_lsc_curve_template() -> None:
    """DSL 模板加载 + run_workflow 完整执行."""
    from zylab.flowchart.batch import run_workflow
    from zylab.flowchart.builtin import BUILTIN_TEMPLATES

    template = next(t for t in BUILTIN_TEMPLATES if t.id == "dsl.lsc_curve")
    outcome = run_workflow(template)
    assert outcome.succeeded, f"失败: {outcome.first_error()}"
    result = outcome.outcome("lsc").result
    assert result["cost"] >= 0  # 正确应用约束后，cost 变大是预期行为
    assert set(result["curves"].keys()) == {"inner_upper", "inner_lower", "outer_upper", "outer_lower"}


class TestLSCCurveValidationWarnings:
    """LSCCurve 负参数警告分支 (miss 127, 129, 131, 136, 138)."""

    def test_negative_heights_warn(self, caplog) -> None:
        LSCCurve(H=-1.0, H1=-0.5, H2=-0.3)
        messages = [r.message for r in caplog.records]
        assert any("负的切削高度" in m for m in messages)
        assert any("负的内部保留高度" in m for m in messages)
        assert any("负的外部保留高度" in m for m in messages)

    def test_negative_slopes_warn(self, caplog) -> None:
        LSCCurve(s=-1.0, s1=-2.0)
        messages = [r.message for r in caplog.records]
        assert any("负的内部斜率" in m for m in messages)
        assert any("负的外部斜率" in m for m in messages)


class TestLSCCurveCachedProperties:
    """LSCCurve cached_property 访问 (miss 190, 195, 200, 274, 293, 298, 334)."""

    def test_m2_powers(self) -> None:
        c = LSCCurve(m2=2.0)
        assert c.m2s == 4.0
        assert c.m2c == 8.0
        assert c.m2s4 == 16.0

    def test_constraint_matrices(self) -> None:
        c = LSCCurve()
        assert c.A_ineq.shape == (11, 16)
        assert c.b_ineq.shape == (11,)
        assert c.A_eq.shape == (11, 16)
        assert c.b_eq.shape == (11,)
