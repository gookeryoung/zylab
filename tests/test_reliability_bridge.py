"""ReliabilityBridge — reliability + workflow integration tests.

测试维度：

1. ``make_limit_state`` 闭包能正确把 RV 采样向量 → overrides → FE → eval 表达式
2. ``run_form`` 与 ``run_sorm`` 对含 FE 参数的桥接场景能收敛
3. FORM / SORM 结果 pf 数量级一致（同一设计点）
4. ``run_mc`` 接口通畅（小样本）
5. FE 失败时返回 ``1e10`` 哨兵值（保护 FORM 迭代）
"""

from __future__ import annotations

import numpy as np
import pytest

from zylab.flowchart import TemplateRegistry
from zylab.reliability import (
    Distribution,
    RandomVariable,
    ReliabilityBridge,
)

# ---------------------------------------------------------------------------
# fixture: cantilever bridge 实例
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bridge() -> ReliabilityBridge:
    """2-D 桥接：height 是 FE 参数 RV + sigma_y 是纯计算 RV."""
    tpl = TemplateRegistry.with_builtin().get("structural.cantilever_static")
    return ReliabilityBridge(
        template=tpl,
        rv_mapping={
            "model.height": RandomVariable(
                name="h",
                dist=Distribution.NORMAL,
                params={"loc": 100.0, "scale": 10.0},
            ),
            "sigma_y": RandomVariable(
                name="sigma_y",
                dist=Distribution.LOGNORMAL,
                params={"loc": 250.0, "scale": 15.0},
            ),
        },
        limit_state_expr="rv.sigma_y - fe.max_stress",
    )


@pytest.fixture(scope="module")
def g(bridge: ReliabilityBridge):
    return bridge.make_limit_state()


# ---------------------------------------------------------------------------
# 1. make_limit_state 基本功能
# ---------------------------------------------------------------------------


class TestMakeLimitState:
    @pytest.mark.slow()
    def test_mean_point_positive(self, g):
        """均值点 g ≈ 110.7 > 0（安全域）."""
        # Normal(100,10) + Lognormal mean ≈ 249.55
        x_mean = np.array([100.0, 249.55])
        val = float(g(x_mean))
        assert val > 50, f"均值点应在安全域, g={val}"
        assert np.isfinite(val)

    @pytest.mark.slow()
    def test_fe_failure_sentinel(self):
        """FE 参数触发越界校验时返回 1e10 哨兵，避免 FORM 迭代崩溃."""
        tpl = TemplateRegistry.with_builtin().get("structural.cantilever_static")
        bridge = ReliabilityBridge(
            template=tpl,
            rv_mapping={
                "model.height": RandomVariable(
                    name="h",
                    dist=Distribution.NORMAL,
                    params={"loc": 100.0, "scale": 10.0},
                ),
            },
            limit_state_expr="fe.strain_energy",
        )
        g = bridge.make_limit_state()
        # height=-1 触发 ParamError，bridge 捕获返回 1e10
        val = float(g(np.array([-1.0])))
        assert val == 1e10

    def test_x_dimension_mismatch_raises(self, bridge: ReliabilityBridge, g):
        """x 维度与 rv_mapping 数量不匹配应报错."""
        with pytest.raises(ValueError, match="维度"):
            g(np.array([1.0]))  # 2-D bridge 传 1-D 输入


# ---------------------------------------------------------------------------
# 2. FORM / SORM 收敛性
# ---------------------------------------------------------------------------


@pytest.mark.slow()
class TestFormSormConvergence:
    def test_form_converges(self, bridge: ReliabilityBridge):
        res = bridge.run_form(max_iter=30, tol=1e-5)
        assert res.converged
        assert res.n_iter <= 30
        assert np.isfinite(res.g_star)
        assert abs(res.g_star) < 1e-3, "设计点应在极限面上 (g≈0)"
        assert res.beta > 0
        assert 1e-10 < res.pf_form < 1e-2

    def test_sorm_converges(self, bridge: ReliabilityBridge):
        sorm = bridge.run_sorm(max_iter=30, tol=1e-5)
        assert np.isfinite(sorm.pf_breitung)
        assert np.isfinite(sorm.pf_hohenbichler)
        assert sorm.pf_breitung > 0
        assert sorm.pf_hohenbichler > 0

    def test_form_sorm_pf_order_magnitude(self, bridge: ReliabilityBridge):
        """FORM 和 SORM pf 应在同一数量级（设计点相同，仅曲率修正不同）."""
        form = bridge.run_form(max_iter=30, tol=1e-5)
        sorm = bridge.run_sorm(max_iter=30, tol=1e-5)
        ratio = form.pf_form / sorm.pf_breitung
        # Breitung 修正通常只改几 %，允许 0.3~3 倍差异
        assert 0.3 < ratio < 3.0, f"FORM={form.pf_form:.4e} SORM={sorm.pf_breitung:.4e} ratio={ratio}"

    def test_sorm_kappa_dimension(self, bridge: ReliabilityBridge):
        """SORM 在极限面切平面上降维 -> kappa 是 (N-1,) 主曲率."""
        sorm = bridge.run_sorm(max_iter=30, tol=1e-5)
        n = len(bridge.rv_mapping)
        assert sorm.kappa.shape == (n - 1,)
        np.testing.assert_array_less(np.abs(sorm.kappa), 10.0)


# ---------------------------------------------------------------------------
# 3. FE 参数映射验证
# ---------------------------------------------------------------------------


class TestFeParamMapping:
    def test_fe_param_key_format(self):
        """只有含 '.' 的 key 才是 FE 参数."""
        tpl = TemplateRegistry.with_builtin().get("structural.cantilever_static")
        bridge = ReliabilityBridge(
            template=tpl,
            rv_mapping={
                "model.height": RandomVariable(
                    name="h",
                    dist=Distribution.NORMAL,
                    params={"loc": 100.0, "scale": 10.0},
                ),
                "sigma_y": RandomVariable(
                    name="sigma_y",
                    dist=Distribution.LOGNORMAL,
                    params={"loc": 250.0, "scale": 15.0},
                ),
                "fe.load_factor": RandomVariable(
                    name="lf",
                    dist=Distribution.NORMAL,
                    params={"loc": 1.0, "scale": 0.1},
                ),
            },
            limit_state_expr="rv.sigma_y - fe.max_stress",
        )
        assert bridge._fe_param_keys() == ["model.height", "fe.load_factor"]
        assert bridge._compute_keys() == ["sigma_y"]

    def test_overrides_builder(self):
        """_build_overrides 把 x 向量按 rv_mapping 顺序正确拆分."""
        tpl = TemplateRegistry.with_builtin().get("structural.cantilever_static")
        bridge = ReliabilityBridge(
            template=tpl,
            rv_mapping={
                "model.height": RandomVariable(
                    name="h",
                    dist=Distribution.NORMAL,
                    params={"loc": 100.0, "scale": 10.0},
                ),
                "sigma_y": RandomVariable(
                    name="sigma_y",
                    dist=Distribution.LOGNORMAL,
                    params={"loc": 250.0, "scale": 15.0},
                ),
            },
            limit_state_expr="rv.sigma_y - fe.max_stress",
        )
        x = np.array([80.0, 260.0])
        overrides = bridge._build_overrides(x)
        assert overrides == {"model": {"height": 80.0}}
        # 缓存里有 rv_values
        rv_vals = bridge._cache["_last_rv_values"]
        assert rv_vals == {"model.height": 80.0, "sigma_y": 260.0}


# ---------------------------------------------------------------------------
# 4. FE 输出提取器
# ---------------------------------------------------------------------------


class TestFeOutputExtractor:
    @pytest.mark.slow()
    def test_collect_static_solution(self):
        """cantilever_static 必须能提取 strain_energy 和 max_stress."""
        from zylab.flowchart import run_workflow
        from zylab.reliability.bridge import _collect_fe_outputs

        tpl = TemplateRegistry.with_builtin().get("structural.cantilever_static")
        outcome = run_workflow(tpl)
        fe = _collect_fe_outputs(outcome)
        assert "strain_energy" in fe
        assert fe["strain_energy"] > 0
        assert "max_stress" in fe
        assert fe["max_stress"] > 0

    def test_eval_expr_with_simple_namespace(self):
        """eval 表达式支持 rv / fe 名空间访问."""
        from types import SimpleNamespace

        fe_ns = SimpleNamespace(max_stress=138.9, strain_energy=539.8)
        rv_ns = SimpleNamespace(sigma_y=250.0)
        expr = "rv.sigma_y - fe.max_stress"
        val = float(eval(expr, {"__builtins__": {}}, {"fe": fe_ns, "rv": rv_ns}))
        assert abs(val - 111.1) < 0.01


# ---------------------------------------------------------------------------
# 5. MC 接口（小样本，仅验证通畅）
# ---------------------------------------------------------------------------


@pytest.mark.slow()
class TestMcInterface:
    def test_mc_runs_small(self, bridge: ReliabilityBridge):
        """MC 20 样本能跑完且 pf 在 [0,1]."""
        mc = bridge.run_mc(n_samples=20, method="crude", seed=7)
        assert 0 <= mc.pf <= 1
        assert mc.n_samples == 20

    def test_mc_sobol(self, bridge: ReliabilityBridge):
        """Sobol 低差异序列 MC 接口通畅."""
        mc = bridge.run_mc(n_samples=8, method="sobol", seed=42)
        assert 0 <= mc.pf <= 1


class TestMiscExtractors:
    """补覆盖率：extractors 兜底路径 + parse_namespace_refs."""

    def test_extract_generic_empty(self):
        from zylab.reliability.bridge import _extract_generic

        assert _extract_generic(object()) == {}

    def test_extract_generic_private_only(self):
        from zylab.reliability.bridge import _extract_generic

        class _OnlyPrivate:
            def __init__(self):
                self._x = 1
                self._y = 2

        assert _extract_generic(_OnlyPrivate()) == {}

    def test_extract_generic_mix(self):
        from zylab.reliability.bridge import _extract_generic

        class _Mix:
            def __init__(self):
                self.a = 1.0
                self.b = np.float64(2.5)
                self.c = np.array(3.14)  # 0-d
                self.d = np.array([1, 2, 3])  # 1-d 跳过
                self._skip = 99
                self.bool_ = True  # 跳过 bool

        out = _extract_generic(_Mix())
        assert out == {"a": 1.0, "b": 2.5, "c": 3.14}

    def test_collect_fe_outputs_non_result(self):
        from zylab.reliability.bridge import _collect_fe_outputs

        assert _collect_fe_outputs(None) == {}
        assert _collect_fe_outputs(42) == {}

    def test_parse_namespace_refs_simple(self):
        from zylab.reliability.bridge import _parse_namespace_refs

        fe, rv = _parse_namespace_refs("rv.sigma_y - fe.max_stress")
        assert fe == {"max_stress"}
        assert rv == {"sigma_y"}

    def test_parse_namespace_refs_complex(self):
        from zylab.reliability.bridge import _parse_namespace_refs

        fe, rv = _parse_namespace_refs("fe.strain_energy * 0.01 + rv.L / 500")
        assert fe == {"strain_energy"}
        assert rv == {"L"}

    def test_parse_namespace_refs_bad_syntax(self):
        from zylab.reliability.bridge import _parse_namespace_refs

        # SyntaxError 时返回空集
        fe, rv = _parse_namespace_refs("rv.foo - bar(")
        assert fe == set()
        assert rv == set()

    def test_collect_fe_outputs_none_result_node(self):
        """覆盖 _collect_fe_outputs 的 r is None 分支."""
        from zylab.reliability.bridge import _collect_fe_outputs

        class _NodeOut:
            def __init__(self, r=None):
                self.result = r

        class _RO:
            def __init__(self, outcomes):
                self.outcomes = outcomes

        class _StrainOnly:
            def __init__(self):
                self.strain_energy = 42.0

        ro = _RO([_NodeOut(r=None), _NodeOut(r=_StrainOnly())])
        out = _collect_fe_outputs(ro)
        assert out == {"strain_energy": 42.0}

    def test_parse_namespace_refs_rv_branch(self):
        """覆盖 rv.xxx elif 分支."""
        from zylab.reliability.bridge import _parse_namespace_refs

        fe, rv = _parse_namespace_refs("rv.a + rv.b")
        assert fe == set()
        assert rv == {"a", "b"}

    def test_make_limit_state_ndim_check(self):
        """覆盖 x.ndim != 1 检查."""
        from zylab.flowchart import TemplateRegistry
        from zylab.reliability import Distribution, RandomVariable, ReliabilityBridge

        tpl = TemplateRegistry.with_builtin().get("structural.cantilever_static")
        bridge = ReliabilityBridge(
            template=tpl,
            rv_mapping={
                "model.height": RandomVariable(
                    name="h",
                    dist=Distribution.NORMAL,
                    params={"loc": 100.0, "scale": 10.0},
                ),
            },
            limit_state_expr="fe.strain_energy",
        )
        g = bridge.make_limit_state()
        with pytest.raises(ValueError, match="一维"):
            g(np.array([[1.0]]))  # 2-d → 触发 ndim 检查


@pytest.mark.slow()
class TestMcParallel:
    """Bridge.run_mc n_workers 并行化验证."""

    def test_serial_parallel_consistency(self, bridge: ReliabilityBridge):
        """相同 seed + n_samples 下，串行和并行结果完全一致."""
        res_s = bridge.run_mc(n_samples=50, method="crude", seed=7, n_workers=1)
        res_p = bridge.run_mc(n_samples=50, method="crude", seed=7, n_workers=2)
        assert res_s.pf == res_p.pf
        assert res_s.n_fail == res_p.n_fail
        assert res_s.beta == res_p.beta
        assert res_s.cov == res_p.cov
        assert res_s.pf_ci_95 == res_p.pf_ci_95

    def test_parallel_lhc(self, bridge: ReliabilityBridge):
        """LHC + 并行通畅."""
        res = bridge.run_mc(n_samples=32, method="lhc", seed=42, n_workers=2)
        assert 0 <= res.pf <= 1
        assert res.n_samples == 32

    def test_parallel_sobol(self, bridge: ReliabilityBridge):
        """Sobol + 并行通畅."""
        res = bridge.run_mc(n_samples=16, method="sobol", seed=42, n_workers=2)
        assert 0 <= res.pf <= 1

    def test_workers_below_2_is_serial(self, bridge: ReliabilityBridge):
        """n_workers=None/1 默认走串行分支（和不指定一致）."""
        res_default = bridge.run_mc(n_samples=30, seed=11)
        res_w1 = bridge.run_mc(n_samples=30, seed=11, n_workers=1)
        res_w2 = bridge.run_mc(n_samples=30, seed=11, n_workers=None)
        assert res_default.pf == res_w1.pf == res_w2.pf

    def test_parallel_all_fail(self):
        """pf=1.0 branch + Wilson CI upper bound (n_fail==n_samples)."""
        from zylab.reliability import Distribution, RandomVariable

        tpl = TemplateRegistry.with_builtin().get("structural.cantilever_static")
        b = ReliabilityBridge(
            template=tpl,
            rv_mapping={
                "model.height": RandomVariable(
                    name="h",
                    dist=Distribution.NORMAL,
                    params={"loc": 100.0, "scale": 10.0},
                ),
            },
            limit_state_expr="-1",
        )
        res = b.run_mc(n_samples=30, method="crude", seed=1, n_workers=2)
        assert res.pf == 1.0
        assert res.beta == -float("inf")
        assert res.n_fail == 30
        assert res.pf_ci_95 == (0.05 / 30, 1.0)

    def test_worker_direct_call(self):
        """Direct main-process call to cover _mc_chunk_worker internals."""
        import pickle as _pickle

        from zylab.reliability import Distribution, RandomVariable
        from zylab.reliability.bridge import _mc_chunk_worker

        tpl = TemplateRegistry.with_builtin().get("structural.cantilever_static")
        b = ReliabilityBridge(
            template=tpl,
            rv_mapping={
                "model.height": RandomVariable(
                    name="h",
                    dist=Distribution.NORMAL,
                    params={"loc": 100.0, "scale": 10.0},
                ),
                "sigma_y": RandomVariable(
                    name="sy",
                    dist=Distribution.LOGNORMAL,
                    params={"loc": 250.0, "scale": 15.0},
                ),
            },
            limit_state_expr="rv.sigma_y - fe.max_stress",
        )
        import numpy as np

        X = np.array([[100.0, 249.55], [100.0, 100.0], [100.0, 500.0]])
        bridge_bytes = _pickle.dumps(b)
        g_vals = _mc_chunk_worker((bridge_bytes, X))
        assert g_vals.shape == (3,)
        assert np.isfinite(g_vals).all()

    def test_parallel_intermediate_pf(self):
        """0 < pf < 1 + Wilson CI else branch + beta finite branch."""
        from zylab.reliability import Distribution, RandomVariable

        tpl = TemplateRegistry.with_builtin().get("structural.cantilever_static")
        b = ReliabilityBridge(
            template=tpl,
            rv_mapping={
                "model.height": RandomVariable(
                    name="h",
                    dist=Distribution.NORMAL,
                    params={"loc": 70.0, "scale": 30.0},
                ),
                "sigma_y": RandomVariable(
                    name="sy",
                    dist=Distribution.NORMAL,
                    params={"loc": 200.0, "scale": 20.0},
                ),
            },
            limit_state_expr="rv.sigma_y - fe.max_stress",
        )
        res = b.run_mc(n_samples=100, method="crude", seed=42, n_workers=2)
        assert 0 < res.pf < 1
        assert np.isfinite(res.beta)
        assert np.isfinite(res.pf_ci_95[0])
        assert np.isfinite(res.pf_ci_95[1])
        assert res.pf_ci_95[0] < res.pf_ci_95[1]


class TestBridgeEdgeCases:
    """Bridge 边界条件测试：参数校验、FE 失败哨兵."""

    def test_build_overrides_dim_mismatch(self, bridge: ReliabilityBridge) -> None:
        """_build_overrides 在 x 维度与 rv_mapping 不匹配时抛 ValueError."""
        wrong = np.array([1.0])  # 只有 1 个元素，但 rv_mapping 有多个
        with pytest.raises(ValueError, match="维度"):
            bridge._build_overrides(wrong)

    def test_limit_state_fe_failed_returns_sentinel(self, bridge: ReliabilityBridge, monkeypatch) -> None:
        """make_limit_state 闭包在 run_workflow 返回 succeeded=False 时返回 1e10."""
        from types import SimpleNamespace

        fake_outcome = SimpleNamespace(succeeded=False)
        # run_workflow 在闭包内 from zylab.flowchart import run_workflow
        monkeypatch.setattr("zylab.flowchart.run_workflow", lambda *_a, **_kw: fake_outcome)

        g = bridge.make_limit_state()
        x = np.array([0.0, 0.0])  # bridge fixture 的 rv_mapping 有 2 个键
        assert g(x) == 1e10
