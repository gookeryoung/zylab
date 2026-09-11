"""optimize_direct 单元测试."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from zylab.doe import DesignSpace, DesignVariable
from zylab.flowchart import OutputParam, TemplateRegistry
from zylab.optim import OptimError, Optimizer, optimize_direct


def _template_with_target(name: str = "E"):
    """取 cantilever_static + 一个 output_param 的模板."""
    tpl = TemplateRegistry.with_builtin().get("structural.cantilever_static")
    return dataclasses.replace(
        tpl,
        output_params=(OutputParam(name=name, source="solve.strain_energy"),),
    )


def _variables():
    return DesignSpace.from_variables(
        [
            DesignVariable(name="model.nx", lower=4, upper=16),
            DesignVariable(name="model.ny", lower=2, upper=8),
        ]
    ).variables


class TestOptimizeDirect:
    def test_de_minimize_finds_real_optimum(self) -> None:
        """DE minimize：最优应在边界角点."""
        tpl = _template_with_target()
        res = optimize_direct(tpl, _variables(), n_iter=10, seed=0)
        assert res.optimizer == "differential_evolution"
        assert res.best_y > 0
        # 边界内
        assert 4 <= res.best_x[0] <= 16
        assert 2 <= res.best_x[1] <= 8
        # nx 越小应变能越低
        assert res.best_y == pytest.approx(0.1385, rel=0.05)

    def test_maximize_returns_positive(self) -> None:
        """maximize=True：内部翻负，返回值应为正值."""
        tpl = _template_with_target()
        res = optimize_direct(tpl, _variables(), n_iter=10, seed=0, maximize=True)
        assert res.best_y > 0

    def test_shgo_optimizer(self) -> None:
        """SHGO 优化器能直接工作."""
        tpl = _template_with_target()
        res = optimize_direct(tpl, _variables(), optimizer="shgo", n_iter=10, seed=0)
        assert res.optimizer == "shgo"
        assert res.best_y > 0

    def test_target_override(self) -> None:
        """target= 指定非默认输出参数."""
        tpl = _template_with_target()
        res = optimize_direct(tpl, _variables(), target="E", n_iter=5, seed=0)
        assert res.optimizer == "differential_evolution"

    def test_no_output_params_raises(self) -> None:
        """template 未声明 output_params → OptimError."""
        tpl = TemplateRegistry.with_builtin().get("structural.cantilever_static")
        with pytest.raises(OptimError, match="未声明 output_params"):
            optimize_direct(tpl, _variables())

    def test_bad_variable_name_format_raises(self) -> None:
        """变量名不是 dotted 格式 → OptimError."""
        tpl = _template_with_target()
        ds = DesignSpace.from_variables([DesignVariable(name="nx", lower=4, upper=16)])
        with pytest.raises(OptimError, match=r"node_id\.param_key"):
            optimize_direct(tpl, ds.variables)

    def test_unknown_target_raises(self) -> None:
        """target 指定的输出参数不存在 → OptimError."""
        tpl = _template_with_target()
        with pytest.raises(OptimError, match="无名为"):
            optimize_direct(tpl, _variables(), target="nope")

    def test_callback_receives_every_eval(self) -> None:
        """callback 应被每一次评估调用，拿到 (x, f) 真实值."""
        tpl = _template_with_target()
        trajectory: list[tuple[np.ndarray, float]] = []
        optimize_direct(
            tpl,
            _variables(),
            n_iter=5,
            seed=0,
            callback=lambda x, f: trajectory.append((x.copy(), f)),
        )
        assert len(trajectory) > 0
        for _x, f in trajectory:
            assert f > 0

    def test_optimizer_enum_accepted(self) -> None:
        """optimizer 参数可以接受 Optimizer 枚举."""
        tpl = _template_with_target()
        res = optimize_direct(tpl, _variables(), optimizer=Optimizer.DUAL_ANNEALING, n_iter=5, seed=0)
        assert res.optimizer == "dual_annealing"
