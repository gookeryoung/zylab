"""Pareto 多目标分析单元测试."""

from __future__ import annotations

import numpy as np
import pytest

from zylab.optim import (
    ParetoSummary,
    crowding_distance,
    pareto_front,
    pareto_ranks,
    pareto_summary,
    scalarize,
)

# 9 个手工构造的 2 目标样本点——front 0 共 7 个互不支配点，front 1 共 2 个
_Y_TEST = np.array(
    [
        [1.0, 5.0],  # front 0
        [2.0, 4.0],  # front 0
        [3.0, 3.5],  # front 0
        [4.0, 2.0],  # front 0
        [5.0, 3.0],  # front 1 — 被 (2,4) 支配？不，(1,5) 和 (0.5,6) 也在 front 0
        [6.0, 1.5],  # front 0
        [0.5, 6.0],  # front 0
        [3.5, 4.5],  # front 1
        [7.0, 0.5],  # front 0
    ]
)


class TestParetoRanks:
    def test_basic_two_fronts(self) -> None:
        ranks = pareto_ranks(_Y_TEST)
        assert ranks.shape == (9,)
        assert ranks.max() == 1
        assert int((ranks == 0).sum()) == 7
        assert int((ranks == 1).sum()) == 2

    def test_front_zero_points_are_not_mutually_dominated(self) -> None:
        """front 0 内任意两个点互不支配."""
        idx0 = pareto_front(_Y_TEST, rank=0)
        f0 = _Y_TEST[idx0]
        for i in range(len(f0)):
            for j in range(i + 1, len(f0)):
                dominated_ij = (f0[i] <= f0[j]).all() and (f0[i] < f0[j]).any()
                dominated_ji = (f0[j] <= f0[i]).all() and (f0[j] < f0[i]).any()
                assert not dominated_ij and not dominated_ji

    def test_maximize_targets(self) -> None:
        """minimize=[False, False] 转成最大化."""
        ranks_min = pareto_ranks(_Y_TEST, minimize=[False, False])
        # 边界上 (0.5, 6.0) 两个维度都是最大的 → 不会被任何点支配 → 一定在 front 0
        assert ranks_min[6] == 0

    def test_single_dimension(self) -> None:
        """1 目标时：排序就是支配关系."""
        Y1d = np.array([3.0, 1.0, 2.0]).reshape(-1, 1)
        ranks = pareto_ranks(Y1d)
        # 1.0 是最小 → front 0；2.0 front 1；3.0 front 2
        assert ranks[1] == 0
        assert ranks[2] == 1
        assert ranks[0] == 2

    def test_tie_all_equal(self) -> None:
        """所有点完全相同 → 没人支配谁 → 全部 rank 0."""
        Y_tie = np.array([[2.0, 3.0]] * 5)
        ranks = pareto_ranks(Y_tie)
        assert (ranks == 0).all()

    def test_triple_dominance(self) -> None:
        """3 个完全支配链的点."""
        Y = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        ranks = pareto_ranks(Y)
        assert list(ranks) == [0, 1, 2]


class TestParetoFront:
    def test_rank_out_of_range_returns_empty(self) -> None:
        idx = pareto_front(_Y_TEST, rank=99)
        assert len(idx) == 0


class TestCrowdingDistance:
    def test_boundary_points_are_inf(self) -> None:
        """front 边界点（某一目标维度的 min/max）拥挤距离应为 inf."""
        idx0 = pareto_front(_Y_TEST, rank=0)
        cd = crowding_distance(_Y_TEST[idx0])
        # y=0.5 是 dmax 维度最小 → 边界
        # y=6.0 是 dmax 维度最大 → 边界
        assert np.isinf(cd).any()

    def test_single_point(self) -> None:
        cd = crowding_distance(np.array([[1.0, 2.0]]))
        assert cd.shape == (1,)
        assert np.isinf(cd[0])

    def test_two_points(self) -> None:
        cd = crowding_distance(np.array([[1.0, 1.0], [2.0, 2.0]]))
        assert (np.isinf(cd)).all()

    def test_uniform_grid_has_internal_values(self) -> None:
        """均匀网格的拥挤距离——边界 inf，内部点因相邻点同值而接近 0."""
        grid = np.array([[i, j] for i in range(3) for j in range(3)], dtype=float)
        cd = crowding_distance(grid)
        # NSGA-II 按每个目标维度独立排序的首尾点算边界 → 至少 2 个 inf
        inf_count = np.isinf(cd).sum()
        assert inf_count >= 2
        # 中心 (1,1) 在 x 排序后前后同值、y 排序后前后同值 → cd ≈ 0
        center_idx = np.where((grid[:, 0] == 1) & (grid[:, 1] == 1))[0][0]
        assert cd[center_idx] == 0.0


class TestScalarize:
    def test_weighted_sum(self) -> None:
        s = scalarize(_Y_TEST, weights=[0.5, 0.5])
        expected = _Y_TEST[:, 0] * 0.5 + _Y_TEST[:, 1] * 0.5
        np.testing.assert_allclose(s, expected)

    def test_chebyshev(self) -> None:
        s = scalarize(_Y_TEST, method="chebyshev", ref_point=[3.0, 3.0], weights=[1.0, 1.0])
        # (2.0, 4.0) → max(1*|2-3|, 1*|4-3|) = max(1, 1) = 1
        assert np.isclose(s[1], 1.0)

    def test_epsilon_violation_is_inf(self) -> None:
        """违反 ref_point 的点应返回 inf."""
        s = scalarize(_Y_TEST, method="epsilon", ref_point=[4.0, 4.0])
        # (6.0, 1.5) 的 x=6 > 4 → 违反 → inf
        assert np.isinf(s[5])
        # (2.0, 4.0) 刚好边界 → 不违反 → 3.0
        assert np.isclose(s[1], 3.0)

    def test_bad_method_raises(self) -> None:
        with pytest.raises(ValueError, match="未知标量化方法"):
            scalarize(_Y_TEST, method="not_a_method")

    def test_chebyshev_missing_ref_uses_zero(self) -> None:
        s = scalarize(_Y_TEST, method="chebyshev")
        assert not np.any(np.isnan(s))


class TestParetoSummary:
    def test_summary_shape(self) -> None:
        ps = pareto_summary(_Y_TEST)
        assert isinstance(ps, ParetoSummary)
        assert ps.ranks.shape == (9,)
        assert ps.n_fronts == 2
        assert list(ps.front_sizes) == [7, 2]
        assert ps.best_idx in pareto_front(_Y_TEST, rank=0)

    def test_best_idx_is_from_front_zero(self) -> None:
        ps = pareto_summary(_Y_TEST)
        f0 = pareto_front(_Y_TEST, rank=0)
        assert ps.best_idx in f0
        # best_idx 一定来自 front 0；具体选哪个由拥挤距离 argmax 决定
        assert _Y_TEST[ps.best_idx].shape == (2,)


class TestParetoRealFE:
    """对接 run_batch_outputs 的端到端验证."""

    def test_real_fe_pareto(self) -> None:
        import dataclasses

        from zylab.doe import DesignSpace, DesignVariable, SamplingMethod
        from zylab.flowchart import (
            OutputParam,
            TemplateRegistry,
            run_batch_outputs,
        )

        tpl = TemplateRegistry.with_builtin().get("structural.cantilever_static")
        tpl = dataclasses.replace(
            tpl,
            output_params=(
                OutputParam(name="E", source="solve.strain_energy"),
                OutputParam(
                    name="dmax",
                    source="solve.displacements",
                    expr="amax(norm(value, axis=1))",
                ),
            ),
        )
        ds = DesignSpace.from_variables(
            [
                DesignVariable(name="model.nx", lower=4, upper=16),
                DesignVariable(name="model.ny", lower=2, upper=8),
            ]
        )
        rows = ds.to_input_rows(ds.sample(n_samples=40, method=SamplingMethod.LATIN_HYPERCUBE, seed=0))
        X, Y = run_batch_outputs(tpl, rows)
        assert Y.shape[1] == 2

        ps = pareto_summary(Y)
        assert ps.n_fronts >= 2  # 应该不止一个前沿
        assert 0 <= ps.best_idx < len(X)
        best_x = X[ps.best_idx]
        assert best_x.shape == (2,)


class TestParetoEdgeCases:
    def test_1d_input(self) -> None:
        ranks = pareto_ranks(np.array([1.0, 3.0, 2.0]))
        assert list(ranks) == [0, 2, 1]

    def test_minimize_list_with_false(self) -> None:
        Y = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 6.0]])
        ranks = pareto_ranks(Y, minimize=[True, False])
        assert ranks.max() >= 1

    def test_chebyshev_default_ref_zero(self) -> None:
        s = scalarize(np.array([[1.0, 2.0], [3.0, 4.0]]), method="chebyshev")
        assert s.shape == (2,)
        assert not np.any(np.isnan(s))

    def test_epsilon_missing_ref_raises(self) -> None:
        with pytest.raises(ValueError, match=r"epsilon.*ref_point"):
            scalarize(np.array([[1.0, 2.0]]), method="epsilon")

    def test_crowding_distance_zero_span(self) -> None:
        Y = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
        cd = crowding_distance(Y)
        assert cd.shape == (3,)

    def test_single_rank_one_point(self) -> None:
        Y = np.array([[1.0, 1.0]])
        ps = pareto_summary(Y)
        assert ps.best_idx == 0
