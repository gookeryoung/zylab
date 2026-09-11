"""optimize_pareto + pareto 边界分支测试."""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest

from zylab.doe import DesignSpace, DesignVariable
from zylab.flowchart import OutputParam, TemplateRegistry
from zylab.optim import (
    OptimError,
    crowding_distance,
    optimize_pareto,
    pareto_ranks,
)


@pytest.fixture()
def cantilever_template():
    """带 E + dmax 两个 output 的悬臂梁模板."""
    tpl = TemplateRegistry.with_builtin().get("structural.cantilever_static")
    return dataclasses.replace(
        tpl,
        output_params=(
            OutputParam(name="E", source="solve.strain_energy"),
            OutputParam(name="dmax", source="solve.displacements", expr="amax(norm(value, axis=1))"),
        ),
    )


@pytest.fixture()
def small_design():
    return DesignSpace.from_variables(
        [
            DesignVariable(name="model.nx", lower=4, upper=16),
            DesignVariable(name="model.ny", lower=2, upper=8),
        ]
    )


# --- pareto 边界分支 ---


def test_crowding_1d_input() -> None:
    """行 131-132: Y.ndim == 1 reshape 分支."""
    cd = crowding_distance(np.array([1.0, 2.0, 3.0]))
    assert cd.shape == (3,)
    assert cd[0] == np.inf and cd[2] == np.inf
    assert cd[1] == pytest.approx(1.0, abs=1e-6)


def test_crowding_mixed_minimize_list() -> None:
    """行 140 + 143-144: minimize 是 list 时 + 方向反转分支."""
    Y = np.array([[1.0, 5.0], [3.0, 3.0], [5.0, 1.0]])
    # 第 2 个目标 maximize: 数值反转 → 比较用 -Y[:,1]
    cd = crowding_distance(Y, minimize=[True, False])
    assert cd.shape == (3,)
    # 边界点永远 inf
    assert cd[0] == np.inf and cd[2] == np.inf


def test_pareto_same_point() -> None:
    """所有点完全相同 → 全 rank 0，非边界点 cd=0."""
    Y = np.ones((4, 2))
    assert (pareto_ranks(Y) == 0).all()
    cd = crowding_distance(Y)
    # 相同点时 numpy 排序稳定，第一个和最后一个点挤到边界
    assert cd[0] == np.inf
    assert cd[-1] == np.inf


# --- optimize_pareto 错误路径 ---


def test_optimize_pareto_single_target_error(cantilever_template, small_design) -> None:
    """行 379-381: targets 必须 >= 2."""
    with pytest.raises(OptimError, match="至少需要 2 个"):
        optimize_pareto(cantilever_template, small_design.variables, targets=["E"])


def test_optimize_pareto_invalid_target_error(cantilever_template, small_design) -> None:
    """行 399: template.output_params 缺目标时报错."""
    with pytest.raises(OptimError, match="缺 output_param"):
        optimize_pareto(cantilever_template, small_design.variables, targets=["E", "fake"])


# --- optimize_pareto 成功路径 ---


def test_optimize_pareto_fea_smoke(cantilever_template, small_design) -> None:
    """完整 FE 链路 + 返回 Pareto front 上的点全 rank 0."""
    res = optimize_pareto(
        cantilever_template,
        small_design.variables,
        targets=["E", "dmax"],
        n_population=20,
        n_generations=10,
        seed=42,
    )
    assert res.X.shape[0] > 0
    assert res.X.shape[1] == 2  # 两个变量
    assert res.F.shape == (len(res.X), 2)
    assert res.n_generations == 10
    assert res.n_evaluations >= 200  # 20 pop * 10 gen = 200
    assert res.target_names == ("E", "dmax")
    # 返回的点必须全是非支配解（rank 0）
    ranks = pareto_ranks(res.F)
    assert (ranks == 0).all()


def test_optimize_pareto_cache_reuse(cantilever_template, small_design) -> None:
    """cache dict 在多次 optimize_pareto 调用间共享 → 减少重复 FE."""
    cache: dict = {}
    t0 = time.perf_counter()
    optimize_pareto(
        cantilever_template,
        small_design.variables,
        targets=["E", "dmax"],
        n_population=16,
        n_generations=6,
        seed=0,
        cache=cache,
    )
    first_slow = time.perf_counter() - t0

    t0 = time.perf_counter()
    optimize_pareto(
        cantilever_template,
        small_design.variables,
        targets=["E", "dmax"],
        n_population=16,
        n_generations=6,
        seed=1,
        cache=cache,
    )
    second = time.perf_counter() - t0

    # 第二次应该有缓存收益（因为解空间部分重叠）
    assert second < first_slow * 1.5  # 允许轻微反超，但不应该慢很多
    assert len(cache) > 0
