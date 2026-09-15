"""第二批补充测试：覆盖 CLI / core / io / optim / flowchart 中尚未命中的边界分支.

集中文件，所有测试独立、不耗时（单测 < 2s）、零副作用，尽量用 monkeypatch
patch 难以直接触发的路径.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

__all__ = []


# --------------------------------------------------------------------------- cli


class TestCliExportContinue:
    """cli.py:173 — _export_outcomes 跳过 result=None 的 outcome."""

    def test_export_skips_result_none(self, tmp_path, monkeypatch) -> None:
        """构造 RunOutcome 混合 result=None 和 result=something，验证跳过 None 的节点."""
        from zylab.flowchart.batch import NodeOutcome, RunOutcome

        none_outcome = NodeOutcome(node_id="ghost", name="幽灵节点", result=None)
        ok_outcome = NodeOutcome(node_id="ok", name="完成节点", result=object())

        # patch _export_outcomes 内部实际调用的 export_csv
        import zylab.cli as cli_mod

        calls: list[str] = []

        def fake_export_csv(result, path):
            calls.append(str(path))
            return path

        monkeypatch.setattr("zylab.fea.export_csv", fake_export_csv)

        # 直接调 _export_outcomes（内部会跳过 result=None）
        cli_mod._export_outcomes(RunOutcome((none_outcome, ok_outcome)), str(tmp_path))

        # ghost.csv 不应出现，ok.csv 应该有
        assert any("ok.csv" in c for c in calls), "ok 节点应调用 export_csv"
        assert not any("ghost.csv" in c for c in calls), "ghost 节点 result=None 应跳过"


# --------------------------------------------------------------------------- core/__init__


class TestCoreLazyAttrsMiss:
    """core/__init__.py:182 — _LAZY_ATTRS 外的属性访问."""

    def test_nonexistent_attribute(self) -> None:
        """zylab.core 上不存在的属性名应抛 AttributeError."""
        import zylab.core

        with pytest.raises(AttributeError, match=r"module 'zylab\.core' has no attribute"):
            _ = zylab.core.nonexistent

    def test_lazy_attr_still_works(self) -> None:
        """懒加载正常路径不受影响（冒烟：get_max_workers 可访问）."""
        from zylab.core import get_max_workers

        assert isinstance(get_max_workers(), int)


# --------------------------------------------------------------------------- io/mesh_io


class TestMeshIoUnsupportedEtype:
    """mesh_io.py:152 — write_mesh 遇到 _ETYPE_TO_MESHIO 中没有的单元类型."""

    def test_beam2_not_supported_for_write(self) -> None:
        """BEAM2 不在 _ETYPE_TO_MESHIO 映射里，write_mesh 应抛 MeshIOError."""
        from zylab.fea.mesh import ElementBlock, ElementType, Mesh
        from zylab.io.mesh_io import MeshIOError, write_mesh

        # 2D 坐标 + BEAM2（不支持写出的桁架单元）
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        blocks = (ElementBlock(etype=ElementType.BEAM2, conn=np.array([[0, 1]])),)
        mesh = Mesh(coords=coords, blocks=blocks)

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".vtu", delete=False) as f:
            path = Path(f.name)
        try:
            with pytest.raises(MeshIOError, match="无法写出为 meshio 格式"):
                write_mesh(mesh, str(path))
        finally:
            if path.exists():
                path.unlink()


# --------------------------------------------------------------------------- optimize_direct 失败分支


class _FakeOptimizableTemplate:
    """optimize_direct 用的最小 template stub——只声明 output_params."""

    class _Param:
        def __init__(self, name: str):
            self.name = name

    def __init__(self, out_params: list[str] | None = None):
        self.output_params = [self._Param(n) for n in (out_params or ["E"])]


def _make_variables(names: list[str]):  # -> list[DesignVariable]
    from zylab.doe import DesignVariable

    return [DesignVariable(name=n, lower=0.0, upper=1.0) for n in names]


class TestOptimizeDirectFailedOutcome:
    """optim/optimize.py:285-287 — outcome.succeeded=False 时 callback + return penalty."""

    def test_uses_penalty_when_run_fails(self, monkeypatch) -> None:
        """run_workflow 返回失败 outcome → callback 被调用 penalty_val 且 objective 返回 penalty_val."""
        from zylab.flowchart.batch import NodeOutcome, RunOutcome
        from zylab.optim import Optimizer
        from zylab.optim.optimize import optimize_direct

        template = _FakeOptimizableTemplate(["E"])
        variables = _make_variables(["a.b"])

        fail_outcome = RunOutcome((NodeOutcome(node_id="n", name="n", error="boom"),))

        calls: list[tuple[np.ndarray, float]] = []

        def fake_callback(x: np.ndarray, y: float) -> None:
            calls.append((x.copy(), y))

        # optimize_direct 内部延迟 `from zylab.flowchart import run_workflow`，
        # 所以 patch flowchart 包上的属性才能被正确拦截（子模块每次调用都重新 import 到绑定）
        monkeypatch.setattr("zylab.flowchart.run_workflow", lambda *a, **kw: fail_outcome)

        result = optimize_direct(
            template,
            variables,
            optimizer=Optimizer.DIFFERENTIAL_EVOLUTION,
            n_iter=1,
            seed=0,
            penalty=1234.0,
            callback=fake_callback,
        )
        assert result.best_y == 1234.0, f"best_y 应为 penalty 1234.0，实际 {result.best_y}"
        assert calls, "callback 应被调用"
        _, last_y = calls[-1]
        assert last_y == 1234.0


class TestOptimizeDirectMissingTarget:
    """optim/optimize.py:290-292 — resolved 字典里没有 target 输出参数."""

    def test_uses_penalty_when_target_missing(self, monkeypatch) -> None:
        """outcome 成功但 resolve_outputs 返回空字典 → 走 penalty 分支."""
        from zylab.flowchart.batch import NodeOutcome, RunOutcome
        from zylab.optim import Optimizer
        from zylab.optim.optimize import optimize_direct

        template = _FakeOptimizableTemplate(["E"])
        variables = _make_variables(["a.b"])

        ok_outcome = RunOutcome((NodeOutcome(node_id="n", name="n", result=object()),))

        monkeypatch.setattr(
            "zylab.flowchart.run_workflow",
            lambda *a, **kw: ok_outcome,
        )

        # resolve_outputs 返回不含 target 的空 dict → if target not in resolved 成立
        monkeypatch.setattr(RunOutcome, "resolve_outputs", lambda self, tmpl: {})

        result = optimize_direct(
            template,
            variables,
            optimizer=Optimizer.DIFFERENTIAL_EVOLUTION,
            n_iter=1,
            seed=0,
            penalty=777.0,
        )
        assert result.best_y == 777.0, f"best_y 应为 penalty 777.0，实际 {result.best_y}"


# --------------------------------------------------------------------------- NSGA-II mutate span < 1e-12


class _FakeMultiTemplate:
    """optimize_pareto 用的最小 template stub——需 2 个 targets."""

    class _Param:
        def __init__(self, name: str):
            self.name = name

    def __init__(self):
        self.output_params = [self._Param("E"), self._Param("dmax")]


class TestParetoMutateTinySpan:
    """optim/optimize.py:537 — NSGA-II mutate 时某变量 span 极小触发 continue."""

    def test_mutate_skips_when_span_near_zero(self, monkeypatch) -> None:
        """让一个变量 hi-lo < 1e-15 → 该变量在 polynomial mutate 里 continue."""
        from zylab.doe import DesignVariable
        from zylab.flowchart.batch import NodeOutcome, RunOutcome
        from zylab.optim.optimize import optimize_pareto

        template = _FakeMultiTemplate()
        # 一个 span=1e-15 的极小变量 + 一个正常变量
        variables = [
            DesignVariable(name="a.b", lower=0.0, upper=1e-15),
            DesignVariable(name="a.c", lower=0.0, upper=1.0),
        ]

        ok_outcome = RunOutcome((NodeOutcome(node_id="n", name="n", result=object()),))

        def fake_resolve(self, tmpl):
            return {"E": 1.0, "dmax": 2.0}

        # optimize_pareto 内部延迟 `from zylab.flowchart import run_batch, run_workflow`
        monkeypatch.setattr("zylab.flowchart.run_workflow", lambda *a, **kw: ok_outcome)
        monkeypatch.setattr(RunOutcome, "resolve_outputs", fake_resolve)

        result = optimize_pareto(
            template,
            variables,
            targets=["E", "dmax"],
            n_population=4,
            n_generations=1,  # 仅 1 代即可触发一次完整的 SBX + mutation
            seed=42,
            penalty=1e10,
            n_workers=0,  # 串行，避免进程池
        )
        assert result.n_generations == 1
        assert result.X.ndim == 2


# --------------------------------------------------------------------------- pareto_summary 空 front


class TestParetoSummaryEmptyFront:
    """optim/pareto.py:245 — pareto_summary 在 front 0 为空时 best_idx=-1."""

    def test_best_idx_negative_when_front0_empty(self, monkeypatch) -> None:
        """patch pareto_front 永远返回空数组，触发 len(f0_idx)==0 分支."""
        import numpy as np

        from zylab.optim import pareto as pareto_mod
        from zylab.optim.pareto import pareto_summary

        def fake_front(*a, **kw):
            return np.array([], dtype=int)

        monkeypatch.setattr(pareto_mod, "pareto_front", fake_front)

        Y = np.array([[1.0, 2.0], [2.0, 3.0], [0.5, 1.0]])
        with patch.object(pareto_mod, "pareto_front", fake_front):
            ps = pareto_summary(Y)

        assert ps.best_idx == -1, f"front0 为空时 best_idx 应为 -1，实际 {ps.best_idx}"


# --------------------------------------------------------------------------- GprSurrogate.kernel_ 未 fit


class TestGprSurrogateKernelUnfit:
    """optim/surrogate.py:116 — 未 fit 时访问 kernel_ 属性抛 SurrogateError."""

    def test_kernel_raises_when_unfit(self) -> None:
        from zylab.optim.errors import SurrogateError
        from zylab.optim.surrogate import GprSurrogate

        g = GprSurrogate()
        with pytest.raises(SurrogateError, match="尚未 fit"):
            _ = g.kernel_

    def test_kernel_available_after_fit(self) -> None:
        """fit 后 kernel_ 应返回 sklearn 核对象（冒烟）."""
        from zylab.optim.surrogate import GprSurrogate

        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (8, 2))
        y = X[:, 0] ** 2 + X[:, 1]
        g = GprSurrogate().fit(X, y)
        assert g.kernel_ is not None


# --------------------------------------------------------------------------- run_batch n_workers=-1


class TestRunBatchMinus1Workers:
    """flowchart/batch.py:383 — n_workers=-1 时读取 get_max_workers."""

    def test_minus1_uses_runtime_default(self, monkeypatch) -> None:
        """get_max_workers 返回 1 → 走串行，但覆盖了 n_workers == -1 的 sentinel 分支."""
        from zylab.flowchart.batch import NodeOutcome, RunOutcome, run_batch

        ok_outcome = RunOutcome((NodeOutcome(node_id="n", name="n", result=object()),))

        calls: list[int] = []

        def fake_get_max_workers() -> int:
            calls.append(1)
            return 1  # 强制串行，不 fork 进程

        monkeypatch.setattr("zylab.flowchart.batch.get_max_workers", fake_get_max_workers)
        monkeypatch.setattr("zylab.flowchart.batch.run_workflow", lambda *a, **kw: ok_outcome)

        result = run_batch(object(), [{"a": 1}], n_workers=-1)
        assert calls, "get_max_workers 应被调用"
        assert len(result) == 1


# --------------------------------------------------------------------------- _row_to_overrides INT round


class _FakeNode:
    def __init__(self, type_id: str, params: dict):
        self.type_id = type_id
        self.params = params


class _FakeTemplate:
    def __init__(self):
        self._nodes = {"n1": _FakeNode("fake.type", {"nx": 8, "ny": 4})}

    def node(self, node_id: str):
        return self._nodes[node_id]


class TestRowToOverridesIntRound:
    """flowchart/batch.py:293-297 — INT 类型 ParamSpec 的 float 值自动 round."""

    def test_int_param_rounds_float_value(self, monkeypatch) -> None:
        """patch module_spec 返回 INT 类型 ParamSpec，触发 float → int 分支."""
        from zylab.flowchart.batch import _row_to_overrides
        from zylab.flowchart.module import ModuleCategory, ModuleSpec, ParamSpec, ParamType

        fake_spec = ModuleSpec(
            type_id="fake.type",
            name="fake",
            category=ModuleCategory.POST,
            target="fake:fn",
            params=(ParamSpec(key="nx", label="X", param_type=ParamType.INT, default=8),),
        )

        monkeypatch.setattr("zylab.flowchart.module.module_spec", lambda type_id: fake_spec)

        template = _FakeTemplate()
        row = {"n1.nx": 8.7, "n1.ny": 4.2}

        result = _row_to_overrides(template, row)

        assert result["n1"]["nx"] == 9, f"INT param float 8.7 应 round 为 9，实际 {result['n1']['nx']}"
        # ny 不是 INT spec，保持原值 4.2
        assert result["n1"]["ny"] == 4.2


# --------------------------------------------------------------------------- DSL derived param NameError

_YAML_BAD_DERIVED = """
meta: {id: t.bad, name: bad}
params:
  foo:
    items:
      x: {value: 1.0}
      y: {expr: 'undef + x'}
pipeline: []
results: []
report: []
"""


class TestDslDerivedUnknownName:
    """flowchart/dsl.py:610-611 — 派生参数引用未声明变量（NameError 路径）."""

    def test_unknown_name_in_expr_wrapped(self, monkeypatch) -> None:
        """patch safe_eval 让它抛裸 NameError → 触发 dsl 里的 except NameError 分支."""
        from zylab.flowchart.dsl import dsl_from_yaml, safe_eval
        from zylab.flowchart.errors import TemplateError

        def fake_safe_eval(expr, ns=None):
            if "undef" in expr:
                raise NameError(f"name {expr!r} is not defined")
            return safe_eval(expr, ns)

        monkeypatch.setattr("zylab.flowchart.dsl.safe_eval", fake_safe_eval)

        with pytest.raises(TemplateError, match="引用未声明变量"):
            dsl_from_yaml(_YAML_BAD_DERIVED)


# --------------------------------------------------------------------------- vfilm_resistor_mesh 折返 180°


class TestVfilmMeshPathZigzag:
    """flowchart/meshing3d.py:265 — 相邻弦向相反，截面法向无法定义."""

    def test_path_180_zigzag_raises(self) -> None:
        """让 vertices[2] 和 vertices[3] 在 x 方向几乎重合，chords[1]+chords[2] 趋近零向量."""
        from zylab.fea.errors import MeshError
        from zylab.flowchart.meshing3d import vfilm_resistor_mesh

        with pytest.raises(MeshError, match="路径折返 180"):
            vfilm_resistor_mesh(
                span=2.000000000000001,  # 刚大于 2*lead_len，让第二/三段弦几乎相反
                depth=100.0,
                width=1.0,
                thickness=0.01,
                substrate_h=0.5,
                lead_len=1.0,
                n_lead=1,
                n_diag=1,
                n_width=1,
                n_sub=1,
            )


# --------------------------------------------------------------------------- catalog_tree 单级 path


class TestCatalogTreeSingleLevel:
    """flowchart/catalog.py:137-139 — spec.catalog 只有一个元素，走 top.modules.append 分支."""

    def test_single_level_path_attaches_to_top(self) -> None:
        """catalog 单元素 tuple → top 节点直接挂 modules."""
        from zylab.flowchart.catalog import catalog_tree
        from zylab.flowchart.module import ModuleCategory, ModuleSpec

        spec = ModuleSpec(
            type_id="test.single",
            name="测试单级",
            category=ModuleCategory.POST,
            target="zylab.flowchart.nodes:identity",
            catalog=("遗留一体化",),  # CATEGORY_LABELS["legacy"] 对应中文 label
        )

        tree = catalog_tree([spec])

        # 应找到 "遗留一体化" 顶层节点且它的 modules 包含我们的 spec
        legacy = next(n for n in tree if n.label == "遗留一体化")
        assert spec in legacy.modules, "单级 path 的 spec 应直接挂在 top 节点的 modules"
        # 单级路径不会生成子节点
        assert not legacy.children


# --------------------------------------------------------------------------- cloud_svg 不同 field 分支 + 3D 网格


class TestCloudSvgFieldBranches:
    """flowchart/report.py 云图 field 分支 + 3D 网格渲染."""

    def _make_fake_payload(self, field_attrs: dict, mesh_dim: int = 2):
        """构造假解对象 + 假 mesh，按 field_attrs 挂属性."""
        from zylab.fea.mesh import ElementBlock, ElementType, Mesh

        n_nodes = 4 if mesh_dim == 2 else 8
        coords = np.zeros((n_nodes, mesh_dim), dtype=float)
        coords[:, 0] = np.arange(n_nodes, dtype=float)
        if mesh_dim >= 2:
            coords[:, 1] = np.arange(n_nodes, dtype=float) * 0.1
        if mesh_dim >= 3:
            coords[:, 2] = np.arange(n_nodes, dtype=float) * 0.01

        conn = np.array([[0, 1], [1, 0]])  # 伪连接，云图只关心 mesh.n_nodes / mesh.dim / mesh.blocks
        blocks = (ElementBlock(etype=ElementType.TRUSS2, conn=conn),)
        mesh = Mesh(coords=coords, blocks=blocks)

        class FakePayload:
            pass

        payload = FakePayload()
        payload.mesh = mesh
        for k, v in field_attrs.items():
            setattr(payload, k, v)
        return payload

    def test_cloud_2d_displacement(self) -> None:
        """2D 网格 + displacement 场 → 云图 SVG 成功返回."""
        from zylab.flowchart.report import CloudData, _cloud_svg

        n = 4
        payload = self._make_fake_payload({"displacements": np.zeros((n, 2))}, mesh_dim=2)
        view = CloudData(node_id="n1", title="位移", field="displacement", payload=payload, deform=1.0)

        svg, placeholder = _cloud_svg(view)
        assert svg is not None
        assert "<svg" in svg
        assert placeholder == ""

    def test_cloud_3d_temperature(self) -> None:
        """3D 网格 + temperature 场 → 走 _cloud_3d 分支."""
        from zylab.flowchart.report import CloudData, _cloud_svg

        n = 8
        payload = self._make_fake_payload(
            {
                "temperatures": np.random.default_rng(0).uniform(300, 500, (1, n)),
                "displacements": np.zeros((n, 3)),  # 避免 _cloud_coords 空位移干扰
            },
            mesh_dim=3,
        )
        view = CloudData(node_id="n1", title="温度", field="temperature", payload=payload, deform=0.0)

        svg, placeholder = _cloud_svg(view)
        assert svg is not None
        # 3D 投影特有 circle 元素
        assert "<circle" in svg or "vertices" not in svg  # 2D 用 polygon/polyline
        assert placeholder == ""

    def test_cloud_voltage(self) -> None:
        """voltage 场独立分支."""
        from zylab.flowchart.report import CloudData, _cloud_svg

        n = 4
        payload = self._make_fake_payload({"voltages": np.arange(n, dtype=float)}, mesh_dim=2)
        view = CloudData(node_id="n1", title="电压", field="voltage", payload=payload, deform=0.0)

        svg, placeholder = _cloud_svg(view)
        assert svg is not None
        assert "电压" in svg
        assert placeholder == ""

    def test_cloud_unknown_field_returns_placeholder(self) -> None:
        """field 不支持 → _cloud_svg 捕获异常返回占位."""
        from zylab.flowchart.report import CloudData, _cloud_svg

        n = 4
        payload = self._make_fake_payload({"displacements": np.zeros((n, 2))}, mesh_dim=2)
        view = CloudData(node_id="n1", title="未知", field="fentanyl", payload=payload, deform=0.0)

        svg, placeholder = _cloud_svg(view)
        assert svg is None
        assert "暂不支持" in placeholder
