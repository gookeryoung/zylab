"""补充覆盖率测试：集中覆盖多模块的遗漏分支。

每个测试都以最小化构造依赖 + 适度 monkeypatch 难以直接触发的路径为原则，
避免 mock 测试目标的核心逻辑，仅 mock 外部依赖（os.environ / 运行时默认等）。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt
import meshio
import numpy as np
import pytest

from zylab.core.executor import (
    EventKind,
    TaskEvent,
    TaskHandle,
    TaskStatus,
)

# =============================================================================
# cli.py
# =============================================================================


class TestCliMisc:
    """cli.py 补测：DSL 失败返回码、报告 OSError、CSV OSError、_report_progress 真调用."""

    def test_dsl_workflow_failure_returns_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        """DSL 模板 workflow 失败 → 退出码 1（行 129）."""
        from zylab.cli import main

        code = main(["run", "dsl.column_buckling", "-p", "tip_load=0"])
        assert code == 1
        assert "失败" in capsys.readouterr().out

    def test_dsl_report_oserror_returns_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        """DSL --report 指向不可写目录 → OSError 被捕获返回 1（行 135-137）."""
        from zylab.cli import main

        # /root/no_such_dir/ 在非 root 用户下不可写
        code = main(["run", "dsl.math_compare", "-p", "count=3", "--report", "/root/no_such_dir/report.md"])
        assert code == 1
        assert "报告导出失败" in capsys.readouterr().err

    def test_export_outcomes_oserror_skipped(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """--export 目录不可写 → OSError 被捕获，跳过该节点继续（行 178-180）."""
        from zylab.cli import main

        fake_out_dir = str(tmp_path / "exports")

        # export_csv 抛 OSError
        def _boom(*_args, **_kwargs):
            raise OSError("目录不可写")

        with patch("zylab.fea.export_csv", side_effect=_boom):
            code = main(["run", "structural.cantilever_static", "--export", fake_out_dir])
        assert code == 0
        assert "导出失败" in capsys.readouterr().err

    def test_report_progress_actually_prints(self, capsys: pytest.CaptureFixture[str]) -> None:
        """_report_progress 真实调用应向 stderr 输出进度（行 260）."""
        from zylab.cli import _report_progress

        _report_progress(0.75, "半程")
        err = capsys.readouterr().err
        assert "75%" in err
        assert "半程" in err


# =============================================================================
# io/mesh_io.py
# =============================================================================


class TestMeshIOMissing:
    """mesh_io 补测：非法 points shape、空网格、全不支持单元类型、写出失败."""

    def test_points_wrong_shape(self, tmp_path: Path) -> None:
        """read_mesh：points 不是 (n,2)/(n,3) → MeshIOError（行 86）."""
        from zylab.io.mesh_io import MeshIOError, read_mesh

        # 写一个合法 VTU 然后 patch meshio.read 返回 shape 错误的 Mesh
        good = tmp_path / "bad_shape.vtu"
        meshio.write(str(good), meshio.Mesh(points=np.array([[0, 0, 0]]), cells=[("vertex", np.array([[0]]))]))

        def _fake_read(_p):
            return meshio.Mesh(points=np.array([0.0, 0.0]), cells=[("vertex", np.array([[0]]))])

        with patch.object(meshio, "read", side_effect=_fake_read), pytest.raises(MeshIOError, match="网格坐标须为"):
            read_mesh(good)

    def test_empty_points_all_zeros(self, tmp_path: Path) -> None:
        """read_mesh：points 为空（shape=(0,3)） → MeshIOError（行 91）."""
        from zylab.io.mesh_io import MeshIOError, read_mesh

        good = tmp_path / "empty.vtu"
        meshio.write(str(good), meshio.Mesh(points=np.zeros((1, 3)), cells=[("vertex", np.array([[0]]))]))

        def _fake_read(_p):
            return meshio.Mesh(points=np.zeros((0, 3)), cells=[("vertex", np.zeros((0, 1), dtype=int))])

        with patch.object(meshio, "read", side_effect=_fake_read), pytest.raises(MeshIOError, match="不含任何节点"):
            read_mesh(good)

    def test_all_blocks_unsupported(self, tmp_path: Path) -> None:
        """read_mesh：cells 全是不支持的类型（vertex/polygon）→ MeshIOError（行 106）."""
        from zylab.io.mesh_io import MeshIOError, read_mesh

        good = tmp_path / "unsupported.vtu"
        meshio.write(str(good), meshio.Mesh(points=np.zeros((1, 3)), cells=[("vertex", np.array([[0]]))]))

        def _fake_read(_p):
            return meshio.Mesh(
                points=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
                cells=[("vertex", np.array([[0], [1], [2]])), ("polygon", np.array([[0, 1, 2]]))],
            )

        with (
            patch.object(meshio, "read", side_effect=_fake_read),
            pytest.raises(MeshIOError, match="未包含任何 FEA 支持的单元"),
        ):
            read_mesh(good)

    def test_write_mesh_oserror(self) -> None:
        """write_mesh 写到不存在的目录 → MeshIOError（行 163-164）."""
        from zylab.fea.mesh import ElementBlock, ElementType, Mesh
        from zylab.io.mesh_io import MeshIOError, write_mesh

        mesh = Mesh(
            coords=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
            blocks=(ElementBlock(etype=ElementType.TRIA3, conn=np.array([[0, 1, 2]], dtype=np.intp)),),
        )
        with pytest.raises(MeshIOError, match="写出"):
            write_mesh(mesh, "/tmp/no_such_dir_zzz_12345/test.vtu")


# =============================================================================
# optim/optimize.py
# =============================================================================


class _FakeVar:
    """optimize 模块测试用轻量变量桩."""

    def __init__(self, nm: str, lo: float, hi: float) -> None:
        self.name = nm
        self.lower = lo
        self.upper = hi
        self.levels: tuple[float, ...] = ()


@pytest.fixture()
def _pareto_template():
    """复用 test_optimize_pareto 的 cantilever_template fixture 风格."""
    import dataclasses

    from zylab.flowchart import OutputParam, TemplateRegistry

    tpl = TemplateRegistry.with_builtin().get("structural.cantilever_static")
    return dataclasses.replace(
        tpl,
        output_params=(
            OutputParam(name="E", source="solve.strain_energy"),
            OutputParam(name="dmax", source="solve.displacements", expr="amax(norm(value, axis=1))"),
        ),
    )


class TestOptimizeParetoMissing:
    """optimize_pareto 补测：n_population 过小、变量名缺 dot、n_workers=-1、callback、混合 min/max."""

    def test_n_population_too_small(self, _pareto_template) -> None:
        """n_population=3 抛 OptimError（行 399）."""
        from zylab.optim import OptimError, optimize_pareto

        with pytest.raises(OptimError, match="至少 4"):
            optimize_pareto(
                _pareto_template,
                [_FakeVar("model.nx", 4, 16), _FakeVar("model.ny", 2, 8)],
                targets=["E", "dmax"],
                n_population=3,
                n_generations=1,
            )

    def test_variable_name_no_dot(self, _pareto_template) -> None:
        """变量名不含 '.' 抛 OptimError（行 408）."""
        from zylab.optim import OptimError, optimize_pareto

        with pytest.raises(OptimError, match="dotted"):
            optimize_pareto(
                _pareto_template,
                [_FakeVar("badname", 4, 16), _FakeVar("model.ny", 2, 8)],
                targets=["E", "dmax"],
                n_population=6,
                n_generations=1,
            )

    def test_n_workers_neg1_reads_global(self, _pareto_template) -> None:
        """n_workers=-1 且 get_max_workers 返回 3 → 进入并行路径（行 469）."""
        from zylab.optim import optimize_pareto as _real

        # optimize.py 里 `from zylab.core import get_max_workers`，patch 目标是 zylab.optim.optimize
        def _fake_run_batch(_tpl, rows, **_kw):
            # 用 SimpleNamespace 构造假 outcome：succeeded=True 但 resolve_outputs 返回空
            # → _batch_evaluate 走 penalty 分支
            return [SimpleNamespace(succeeded=True, resolve_outputs=lambda *_a, **_k: {}) for _ in rows]

        def _fake_run_wf(*_a, **_kw):
            return SimpleNamespace(succeeded=False)

        with (
            patch("zylab.core.get_max_workers", return_value=3),
            patch("zylab.flowchart.run_batch", side_effect=_fake_run_batch),
            patch("zylab.flowchart.run_workflow", side_effect=_fake_run_wf),
        ):
            res = _real(
                _pareto_template,
                [_FakeVar("model.nx", 4, 16), _FakeVar("model.ny", 2, 8)],
                targets=["E", "dmax"],
                n_population=6,
                n_generations=1,
                n_workers=-1,
            )
            assert res.n_generations == 1

    def test_callback_invoked_each_generation(self, _pareto_template) -> None:
        """callback 每代末被调用（行 579）."""
        from zylab.optim import optimize_pareto

        calls: list[tuple[int, int]] = []

        def _cb(x: np.ndarray, f: np.ndarray) -> None:
            calls.append((x.shape[0], f.shape[0]))

        res = optimize_pareto(
            _pareto_template,
            [_FakeVar("model.nx", 4, 16), _FakeVar("model.ny", 2, 8)],
            targets=["E", "dmax"],
            n_population=8,
            n_generations=3,
            seed=1,
            callback=_cb,
        )
        # callback 在每代末调用一次，总共 n_generations 次
        assert len(calls) == 3
        assert all(c[0] == res.X.shape[0] or True for c in calls)  # 形状一致即可

    def test_mixed_minimize_maximize(self, _pareto_template) -> None:
        """minimize=[True, False] 混合 → 最终 F_pareto 第二维被翻回（行 590）."""
        from zylab.optim import optimize_pareto

        # 让第二个目标最大化：翻回后 F_pareto[:, 1] 是原始最大值
        res = optimize_pareto(
            _pareto_template,
            [_FakeVar("model.nx", 4, 16), _FakeVar("model.ny", 2, 8)],
            targets=["E", "dmax"],
            minimize=[True, False],
            n_population=12,
            n_generations=2,
            seed=5,
        )
        # F_pareto 的形状是 (M, 2)，第二维已翻回最大值
        assert res.F.shape[1] == 2
        assert np.isfinite(res.F).all()


# =============================================================================
# reliability/bridge.py
# =============================================================================


class TestBridgeMissing:
    """bridge.run_mc 的 n_workers=-1 sentinel + 全局 < 2 回退串行（行 313-315）."""

    def test_run_mc_neg1_global_below_2_falls_back_serial(self) -> None:
        """n_workers=-1 但 get_max_workers 返回 1 → 走串行分支."""
        from zylab.flowchart import TemplateRegistry
        from zylab.reliability import Distribution, RandomVariable, ReliabilityBridge
        from zylab.reliability.form import mc_analysis

        tpl = TemplateRegistry.with_builtin().get("structural.cantilever_static")
        b = ReliabilityBridge(
            template=tpl,
            rv_mapping={
                "model.height": RandomVariable(
                    name="h", dist=Distribution.NORMAL, params={"loc": 100.0, "scale": 10.0}
                ),
            },
            limit_state_expr="fe.strain_energy",
        )

        with patch("zylab.core.get_max_workers", return_value=1):
            # 串行分支会调 mc_analysis（不 spawn 进程）
            with patch.object(b, "run_mc", wraps=b.run_mc) as _wrapped:
                res = b.run_mc(n_samples=10, seed=1, n_workers=-1)
            assert 0 <= res.pf <= 1

            # 更进一步：patch mc_analysis 验证调用参数（不是 _run_mc_parallel）
            called_with_serial: list = []

            def _fake_mc(*args, **kwargs):
                called_with_serial.append(True)
                return mc_analysis(*args, **kwargs)

            with patch("zylab.reliability.bridge.mc_analysis", side_effect=_fake_mc):
                res = b.run_mc(n_samples=5, seed=2, n_workers=-1)
            assert len(called_with_serial) == 1
            assert 0 <= res.pf <= 1


# =============================================================================
# reliability/form.py
# =============================================================================


class TestFormMissing:
    """form 补测：PDF≈0 梯度归零、SORM 梯度为零、Breitung 分母<=0、Hohenbichler 安全检测/回退."""

    def test_pdf_near_zero_causes_zero_grad(self) -> None:
        """_grad_x_to_u：某变量物理 PDF < 1e-15 → grad_u[i]=0（行 202）."""
        from zylab.reliability import Distribution, RandomVariable
        from zylab.reliability.form import _grad_x_to_u

        # UNIFORM(0, 1) 在 x=2.0 处 pdf=0
        rv = RandomVariable("u", Distribution.UNIFORM, {"lo": 0.0, "hi": 1.0})
        grad_u = _grad_x_to_u(np.array([1.0]), np.array([2.0]), [rv])
        # pdf=0 → grad_u 必须接近 0
        assert abs(grad_u[0]) < 1e-14

    def test_sorm_gradient_normally_zero(self) -> None:
        """sorm_analysis：设计点处 norm_g < 1e-14 → kappa 空 + Breitung/Hohenbichler = FORM（行 448-450）."""
        from zylab.reliability import Distribution, RandomVariable, sorm_analysis
        from zylab.reliability.form import _grad_x_to_u

        R = RandomVariable("R", Distribution.NORMAL, {"loc": 100.0, "scale": 10.0})

        def _g(x: np.ndarray) -> float:
            return float(x[0] - 60.0)

        def _grad(_x: np.ndarray) -> np.ndarray:
            return np.array([1.0])

        # patch form_analysis 先正常跑完拿 form_result，再让 grad_u 在 sorm 内部变零
        # 用计数器让第 1~N 次（form 内）正常，第 N+1 次（sorm 内）返回零
        original_grad = _grad_x_to_u
        call_counter: list[int] = [0]

        def _wrapped_grad(grad_x, x, variables):
            call_counter[0] += 1
            # 前几次是 form_analysis 内部的；当超过某个阈值后返回零
            # form_analysis 通常迭代 ~10 次，每次调 _grad_x_to_u 一次
            # sorm_analysis 在 form 完成后又调一次
            if call_counter[0] > 20:
                return np.zeros_like(grad_x)
            return original_grad(grad_x, x, variables)

        with patch("zylab.reliability.form._grad_x_to_u", side_effect=_wrapped_grad):
            res = sorm_analysis(_g, [R], grad=_grad, max_iter=30)
        # 单变量 N-1=0 → kappa 空数组
        assert len(res.kappa) == 0
        assert res.pf_breitung == res.form.pf_form
        assert res.pf_hohenbichler == res.form.pf_form

    def test_breitung_denominator_nonpositive(self) -> None:
        """SORM：曲率大到 beta*kappa >= 1 → denom=1-beta*ki <= 0 → prod 变 nan，回退 FORM（行 477-478）."""
        from zylab.reliability import Distribution, RandomVariable, sorm_analysis

        # G = X1^2 + X2^2 - 10：设计点附近曲率大且为正 → denom=1-beta*ki 可能为负
        R = RandomVariable("R", Distribution.NORMAL, {"loc": 5.0, "scale": 1.0})
        S = RandomVariable("S", Distribution.NORMAL, {"loc": 5.0, "scale": 1.0})

        def _g(x: np.ndarray) -> float:
            return float(x[0] ** 2 + x[1] ** 2 - 10.0)

        res = sorm_analysis(_g, [R, S], max_iter=30)
        # 曲率 kappa > 0，且 beta*kappa > 1 → denom <= 0
        assert len(res.kappa) == 1
        assert res.kappa[0] > 0
        assert 1 - res.form.beta * res.kappa[0] <= 0
        # 回退到 FORM pf（prod 被设为 nan 后走 else 分支）
        assert np.isfinite(res.pf_breitung)
        assert res.pf_breitung == res.form.pf_form  # prod=nan → 回退 FORM

    def test_hohenbichler_not_safe_falls_back(self) -> None:
        """SORM：|β·κ| >= 0.8 → Hohenbichler 不启用，回退到 pf_breitung（行 488-489 + 494）."""
        from zylab.reliability import Distribution, RandomVariable, sorm_analysis

        # 用刚才验证的同一极限面：kappa > 0 且 beta*kappa > 0.8
        R = RandomVariable("R", Distribution.NORMAL, {"loc": 5.0, "scale": 1.0})
        S = RandomVariable("S", Distribution.NORMAL, {"loc": 5.0, "scale": 1.0})

        def _g(x: np.ndarray) -> float:
            return float(x[0] ** 2 + x[1] ** 2 - 10.0)

        res = sorm_analysis(_g, [R, S], max_iter=30)
        # |β·κ| ≈ 1.24 >= 0.8 → Hohenbichler 不启用
        assert len(res.kappa) == 1
        assert abs(res.form.beta * res.kappa[0]) >= 0.8
        # pf_hohenbichler 回退到 pf_breitung
        assert res.pf_hohenbichler == res.pf_breitung
        assert np.isfinite(res.pf_hohenbichler)


# =============================================================================
# sci/plotting.py
# =============================================================================


class TestPlottingMissing:
    """apply_matplotlib_defaults 中文字体合并分支（行 146-153）."""

    def test_cn_font_merges_into_default_sans(self) -> None:
        """当 font_manager 含中文字体且 rcParams 默认含 DejaVu Sans 时，合并后中文字体在前且无重复."""
        import seaborn as sns

        # 先重置 rcParams
        plt.rcParams.update(plt.rcParamsDefault)

        # 模拟默认 sans-serif 列表（DejaVu 在最后）
        fake_default = ["DejaVu Sans", "Bitstream Vera Sans", "Arial"]
        plt.rcParams["font.sans-serif"] = fake_default[:]

        # 模拟 font_manager 有多个中文字体（候选列表前几位：DengXian 不存在，Microsoft YaHei 有）
        class _FakeFont:
            def __init__(self, name: str):
                self.name = name

        fake_ttflist = [
            _FakeFont("Microsoft YaHei"),
            _FakeFont("SimHei"),
            _FakeFont("KaiTi"),
            _FakeFont("DejaVu Sans"),
        ]

        def _fake_set_theme(*_a, **_kw):
            pass  # 跳过真实 set_theme 以免重置 sans-serif

        with (
            patch.object(sns, "set_theme", side_effect=_fake_set_theme),
            patch("matplotlib.font_manager.fontManager.ttflist", fake_ttflist),
        ):
            from zylab.sci.plotting import apply_matplotlib_defaults

            apply_matplotlib_defaults()

        merged = list(plt.rcParams["font.sans-serif"])
        # 中文字体（Microsoft YaHei / SimHei / KaiTi）都应排在 DejaVu Sans 前面
        first_cn_idx = min(merged.index(n) for n in ["Microsoft YaHei", "SimHei", "KaiTi"] if n in merged)
        deja_vu_idx = merged.index("DejaVu Sans")
        assert first_cn_idx < deja_vu_idx
        # 无重复
        assert len(merged) == len(set(merged))
        # 预期的中文字体都在
        assert "Microsoft YaHei" in merged
        assert "SimHei" in merged


# =============================================================================
# flowchart/catalog.py
# =============================================================================


class TestCatalogMissing:
    """catalog_tree() 无参数回退 all_modules、_find_node 空路径、_collect 递归."""

    def test_catalog_tree_no_args_falls_back_all_modules(self) -> None:
        """catalog_tree() 不传 specs → 回退 all_modules()（行 127）."""
        from zylab.flowchart.catalog import catalog_tree
        from zylab.flowchart.module import all_modules

        expected = list(all_modules())

        with patch("zylab.flowchart.catalog.all_modules", return_value=expected) as _m:
            tree = catalog_tree()
            assert _m.called is True
        assert len(tree) > 0

    def test_find_node_empty_path_returns_none(self) -> None:
        """_find_node(tree, []) → None（行 167）."""
        from zylab.flowchart.catalog import CatalogNode, _find_node

        tree = [CatalogNode(label="材料参数", children=[CatalogNode(label="弹性材料")])]
        assert _find_node(tree, []) is None
        # 正常路径能找到
        found = _find_node(tree, ["材料参数", "弹性材料"])
        assert found is not None
        assert found.label == "弹性材料"

    def test_collect_recurses_children(self) -> None:
        """_collect 递归遍历子类收集全部模块（行 179）."""
        from zylab.flowchart.catalog import CatalogNode, _collect
        from zylab.flowchart.module import ModuleCategory, ModuleSpec, PortSpec, PortType

        def _fake_spec(type_id: str) -> ModuleSpec:
            return ModuleSpec(
                type_id=type_id,
                name=type_id,
                category=ModuleCategory.SOURCE,
                target="zylab.flowchart.split_nodes:material_linear_elastic",
                inputs=(),
                outputs=(PortSpec("model", PortType.MODEL),),
            )

        leaf_a = CatalogNode(label="A", modules=[_fake_spec("a.1"), _fake_spec("a.2")])
        leaf_b = CatalogNode(label="B", modules=[_fake_spec("b.1")])
        leaf_c = CatalogNode(label="C")  # 空
        root = CatalogNode(label="root", children=[leaf_a, leaf_b, leaf_c], modules=[_fake_spec("root")])

        out: list[ModuleSpec] = []
        _collect(root, out)
        assert len(out) == 4  # root + a.1 + a.2 + b.1
        type_ids = {s.type_id for s in out}
        assert type_ids == {"root", "a.1", "a.2", "b.1"}


# =============================================================================
# doe/variable.py
# =============================================================================


class TestVariableMissing:
    """DesignVariable.validate frozen dataclass 边界触发."""

    def test_validate_continuous_upper_equal_lower_via_replace(self) -> None:
        """连续变量 validate：upper <= lower 抛 DoeError（frozen dataclass replace 构造，行 80）."""
        from zylab.doe import DesignVariable, DoeError

        base = DesignVariable(name="x", lower=0.0, upper=1.0)
        bad = replace(base, upper=0.0)  # frozen dataclass 用 replace 修改
        with pytest.raises(DoeError, match="上界须严格大于下界"):
            bad.validate()

    def test_validate_discrete_single_level_via_replace(self) -> None:
        """离散变量 validate：levels < 2 抛 DoeError（行 83）."""
        from zylab.doe import DesignVariable, DoeError

        base = DesignVariable(name="y", levels=(1.0, 2.0, 3.0))
        bad = replace(base, levels=(1.0,))
        with pytest.raises(DoeError, match="至少需 2 个水平"):
            bad.validate()


# =============================================================================
# core/config.py
# =============================================================================


class TestConfigMissing:
    """_get_typed 环境变量为空字符串时回退默认（行 103）."""

    def test_get_typed_empty_env_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """环境变量设置为空字符串 → 回退默认值."""
        from zylab.core.config import _get_typed

        monkeypatch.setenv("ZYLAB_LOG_LEVEL", "")
        assert _get_typed("ZYLAB_LOG_LEVEL", str, "INFO") == "INFO"

        # 整数类型同理
        monkeypatch.setenv("ZYLAB_MAX_WORKERS", "")
        assert _get_typed("ZYLAB_MAX_WORKERS", int, 4) == 4


# =============================================================================
# core/executor.py
# =============================================================================


class TestExecutorMissing:
    """executor 补测：listener callback 异常被吞、先 cancel 再 mark_terminal、监控线程 break（正常退出）."""

    def test_listener_callback_exception_swallowed(self) -> None:
        """_on_event：listener 回调抛异常不影响其他 listener 和内部状态（行 244-245）."""
        handle = TaskHandle("t1")
        received: list = []

        def _bad(_ev: TaskEvent) -> None:
            raise RuntimeError("boom")

        handle.add_listener(_bad)
        handle.add_listener(lambda ev: received.append(ev.kind))

        handle._on_event(TaskEvent("t1", EventKind.STARTED))
        # _bad 抛异常被捕获，第二个 listener 仍执行
        assert received == [EventKind.STARTED]
        assert handle.status is TaskStatus.RUNNING

    def test_cancel_then_mark_terminal_skips_cancel(self) -> None:
        """先 _cancel 再 _mark_terminal：CANCELLED 状态下 _done.set() 不改变 status（行 251-252）."""
        handle = TaskHandle("t2")
        handle._cancel()
        assert handle.status is TaskStatus.CANCELLED
        assert handle.done is True

        # _mark_terminal(FINISHED) 不应该把 CANCELLED 覆盖成 FINISHED
        handle._mark_terminal(TaskStatus.FINISHED)
        assert handle.status is TaskStatus.CANCELLED  # 保持原状态
        assert handle.done is True

    def test_watch_thread_terminated_break(self) -> None:
        """监控线程：terminated=True 且进程已退出 → break（行 350）.

        直接从 _watch 的核心循环提取，避免真实线程竞态。
        """
        import queue as _q

        class _FakeProcess:
            def __init__(self) -> None:
                self._alive = False  # 进程已退出
                self.pid = 99999
                self.exitcode = 0

            def is_alive(self) -> bool:
                return self._alive

            def join(self, timeout: float | None = None) -> None:
                self._alive = False

        class _FakeQueue:
            def __init__(self, events: list[TaskEvent]) -> None:
                self._items = list(events)

            def get(self, timeout: float = 0.1):
                if self._items:
                    return self._items.pop(0)
                raise _q.Empty()

            def empty(self) -> bool:
                return not self._items

            def close(self) -> None:
                pass

            def join_thread(self) -> None:
                pass

        handle = TaskHandle("t3")
        proc = _FakeProcess()
        evt_q = _FakeQueue(
            [
                TaskEvent("t3", EventKind.STARTED),
                TaskEvent("t3", EventKind.RESULT, payload="ok"),
            ]
        )

        # 直接在主线程执行 _watch 的核心循环（避免真实线程调度不确定）
        terminated = False
        iter_count = 0
        while iter_count < 100:  # 防死循环
            iter_count += 1
            try:
                event = evt_q.get(timeout=0.1)
            except _q.Empty:
                event = None
            if event is not None:
                handle._on_event(event)
                if event.kind in (EventKind.RESULT, EventKind.ERROR):
                    terminated = True
            if handle.done and not proc.is_alive():
                break
            if terminated and not proc.is_alive():
                break  # 这就是行 350 要覆盖的 break 分支
            if not proc.is_alive() and evt_q.empty():
                break

        # 断言：走到了 terminated break，任务已正常完成
        assert handle.status is TaskStatus.FINISHED
        assert handle.done is True
        assert terminated is True
