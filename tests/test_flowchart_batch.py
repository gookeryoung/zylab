"""flowchart.batch 批处理执行测试：进程内拓扑执行 + 参数覆盖/扫描 + 失败中止 + 摘要."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from zylab.fea import StaticSolution
from zylab.flowchart import (
    ModelBundle,
    NodeOutcome,
    OutputParam,
    RunOutcome,
    Template,
    TemplateRegistry,
    explore_doe,
    resolve_target,
    run_batch,
    run_batch_outputs,
    run_scan,
    run_workflow,
    summarize,
)

__all__ = []


def _template(template_id: str) -> Template:
    """从内置注册表取模板."""
    return TemplateRegistry.with_builtin().get(template_id)


def _tip_displacement(outcome) -> float:
    """取静力解最大位移模长."""

    solution = outcome.outcome("solve").result
    assert isinstance(solution, StaticSolution)
    return float(np.linalg.norm(solution.displacements, axis=1).max())


class TestRunWorkflow:
    """进程内拓扑执行."""

    def test_builtin_static_template(self) -> None:
        """内置悬臂梁静力模板进程内运行成功，节点结果类型正确."""
        outcome = run_workflow(_template("structural.cantilever_static"))
        assert outcome.succeeded
        model = outcome.outcome("model")
        assert isinstance(model.result, ModelBundle)
        solve = outcome.outcome("solve")
        assert isinstance(solve.result, StaticSolution)
        assert solve.elapsed > 0.0

    def test_param_overrides(self) -> None:
        """覆盖网格密度参数后节点数发生变化."""
        base = run_workflow(_template("structural.cantilever_static"))
        coarse = run_workflow(_template("structural.cantilever_static"), {"model": {"nx": 4, "ny": 2}})
        n_base = base.outcome("model").result.mesh.n_nodes
        n_coarse = coarse.outcome("model").result.mesh.n_nodes
        assert n_base != n_coarse
        assert coarse.succeeded

    def test_input_attribute_path_binding(self) -> None:
        """连接引用支持 '节点id.端口名.属性路径'（compute.expr 取解对象位移阵）."""
        template = _template("dsl.cantilever_harmonic")
        outcome = run_workflow(template)
        assert outcome.succeeded

        amp = outcome.outcome("amp").result
        solve = outcome.outcome("solve").result
        assert isinstance(amp, np.ndarray)
        assert amp.shape == (solve.n_frequencies,)
        assert np.all(amp >= 0.0)  # 复位移取模为非负幅值
        peak = outcome.outcome("peak").result
        assert float(peak) == pytest.approx(float(amp.max()))

    def test_failure_aborts_downstream(self) -> None:
        """节点失败即中止：下游保持未执行，first_error 定位失败节点."""
        # 无压缩轴力时屈曲特征值无正因子，屈曲节点确定性失败
        outcome = run_workflow(
            _template("structural.column_buckling"),
            {"model": {"tip_load": 0.0}},
        )
        assert not outcome.succeeded
        buckling = outcome.outcome("solve")
        assert buckling.error
        assert buckling.result is None
        assert "solve" in outcome.first_error()
        assert not buckling.ok

    def test_progress_report_forwarded(self) -> None:
        """进度回调透传到节点函数（solve_static 会派发进度消息）."""
        messages: list[str] = []

        def report(_progress: float, message: str) -> None:
            messages.append(message)

        run_workflow(_template("structural.cantilever_static"), report=report)
        assert messages

    def test_failure_skips_downstream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """中游节点失败后下游节点不再执行（mock 第二个节点抛错）."""
        import zylab.flowchart.batch as batch_mod

        calls: list[str] = []

        def fake_resolve(target: str) -> object:
            def fn(inputs: object, params: object, report: object = None) -> object:
                calls.append(target)
                if len(calls) == 2:
                    raise RuntimeError("boom")
                return object()

            return fn

        monkeypatch.setattr(batch_mod, "resolve_target", fake_resolve)
        outcome = run_workflow(_template("structural.cantilever_combo"))
        assert len(calls) == 2  # 第三节点被跳过
        assert not outcome.succeeded
        assert outcome.outcome("static").error
        assert outcome.outcome("modal").result is None
        assert outcome.outcome("modal").error == ""


class TestRunScan:
    """参数化扫描."""

    def test_linear_scaling(self) -> None:
        """线弹性下载荷扫描位移严格成比例."""
        runs = run_scan(_template("structural.cantilever_static"), "model.tip_load", (1.0, 2.0, 4.0))
        assert len(runs) == 3
        assert all(r.succeeded for r in runs)
        d1, d2, d4 = (_tip_displacement(r) for r in runs)
        assert d2 == pytest.approx(2.0 * d1)
        assert d4 == pytest.approx(4.0 * d1)

    def test_bad_ref_raises(self) -> None:
        """非法参数引用格式抛 ValueError."""
        with pytest.raises(ValueError, match=r"节点id\.参数键"):
            run_scan(_template("structural.cantilever_static"), "tip_load", (1.0,))

    def test_unknown_node_raises(self) -> None:
        """引用不存在的节点抛 ValueError."""
        with pytest.raises(ValueError, match="无节点"):
            run_scan(_template("structural.cantilever_static"), "ghost.tip_load", (1.0,))


class TestSummarizeAndResolve:
    """摘要生成与目标解析."""

    @pytest.mark.parametrize(
        ("template_id", "keyword"),
        [
            ("structural.cantilever_modal", "模态"),
            ("structural.cantilever_harmonic", "谐响应"),
            ("structural.cantilever_transient", "瞬态"),
            ("structural.column_buckling", "屈曲"),
            ("structural.truss_nonlinear", "非线性"),
            ("thermal.joule_plate_2d", "电热"),
        ],
    )
    def test_summarize_solution_types(self, template_id: str, keyword: str) -> None:
        """各解类型模板运行成功且摘要含类型关键字（覆盖 _describe 分支）."""
        outcome = run_workflow(_template(template_id))
        assert outcome.succeeded, outcome.first_error()
        assert keyword in summarize(outcome)

    def test_describe_unknown_type(self) -> None:
        """未知输出类型退化为类名描述."""
        text = summarize(RunOutcome((NodeOutcome(node_id="x", name="未知", result=object()),)))
        assert "object" in text

    def test_summarize_lines(self) -> None:
        """摘要包含节点 id 与解类型关键字."""
        outcome = run_workflow(_template("structural.cantilever_static"))
        text = summarize(outcome)
        assert "[model]" in text
        assert "模型" in text
        assert "[solve]" in text
        assert "静力" in text

    def test_summarize_failure(self) -> None:
        """失败运行摘要包含错误与未执行标记."""
        outcome = run_workflow(_template("structural.column_buckling"), {"model": {"tip_load": 0.0}})
        text = summarize(outcome)
        assert "失败" in text
        assert "未执行" not in text  # 单下游失败即整体两节点内已含失败行

    def test_summarize_skipped_node(self) -> None:
        """失败中止后的未执行节点在摘要中标记上游失败."""
        failed = NodeOutcome(node_id="static", name="静力分析", error="SolverError: 失败")
        skipped = NodeOutcome(node_id="buckling", name="屈曲分析")
        text = summarize(RunOutcome((failed, skipped)))
        assert "失败: SolverError" in text
        assert "未执行（上游失败）" in text

    def test_resolve_target(self) -> None:
        """目标字符串解析为可调用节点函数."""
        fn = resolve_target("zylab.flowchart.nodes:run_static")
        assert callable(fn)

    def test_outcome_lookup_missing(self) -> None:
        """RunOutcome.outcome 对未知节点抛 KeyError."""
        outcome = run_workflow(_template("structural.cantilever_static"))
        with pytest.raises(KeyError, match="nope"):
            outcome.outcome("nope")


def test_node_outcome_defaults() -> None:
    """未执行节点（无结果无错误）不构成成功运行."""
    skipped = RunOutcome((NodeOutcome(node_id="x", name="示例"),))
    assert not skipped.succeeded
    assert skipped.outcome("x").ok
    assert skipped.outcome("x").result is None
    assert skipped.first_error() == ""


class TestRunBatch:
    """run_batch / run_batch_outputs —— 参数批量运行 + 扁平 X/Y 输出."""

    def test_run_batch_multi_row_flat_param_keys(self) -> None:
        """多组 'node_id.param_key' 扁平参数批量运行，全部成功."""
        tpl = _template("structural.cantilever_static")
        rows = [
            {"model.nx": 4, "model.ny": 2},
            {"model.nx": 6, "model.ny": 3},
            {"model.nx": 8, "model.ny": 4},
        ]
        outcomes = run_batch(tpl, rows)
        assert len(outcomes) == 3
        for o in outcomes:
            assert o.succeeded, o.first_error()

    def test_run_batch_report_callback_invoked(self) -> None:
        """report 回调在运行期间被调用（节点级进度更新）."""
        tpl = _template("structural.cantilever_static")
        rows = [{"model.nx": 4, "model.ny": 2}, {"model.nx": 6, "model.ny": 3}]
        call_count: list[int] = []

        def report(_progress: float, _msg: str) -> None:
            call_count.append(1)

        outcomes = run_batch(tpl, rows, report=report)
        assert len(call_count) > 0  # 至少被调用一次
        for o in outcomes:
            assert o.succeeded, o.first_error()

    def test_run_batch_outputs_scalar_yields(self) -> None:
        """带 output_params 的模板 → run_batch_outputs 返回 (N, D_in) X + (N, D_out) Y."""
        tpl = _template("structural.cantilever_static")
        new_tpl = dataclasses.replace(
            tpl,
            output_params=(
                OutputParam(
                    name="strain_energy",
                    source="solve.strain_energy",
                    label="应变能",
                    unit="J",
                ),
            ),
        )
        rows = [
            {"model.nx": 4, "model.ny": 2},
            {"model.nx": 6, "model.ny": 3},
        ]
        X, Y = run_batch_outputs(new_tpl, rows)
        assert X.shape == (2, 2)  # 两行 × 两个输入参数
        assert Y.shape == (2, 1)  # 两行 × 一个输出参数
        # 更细网格的应变能更大（应力更集中）——定性关系
        assert Y[1, 0] > Y[0, 0]

    def test_run_batch_empty_rows_raises(self) -> None:
        """param_rows 为空 —— ValueError 快速失败."""
        tpl = _template("structural.cantilever_static")
        with pytest.raises(ValueError, match="param_rows 不能为空"):
            run_batch(tpl, [])

    def test_run_batch_outputs_no_output_params_raises(self) -> None:
        """run_batch_outputs 需要 template 声明 output_params."""
        tpl = _template("structural.cantilever_static")
        with pytest.raises(Exception, match="未声明 output_params"):
            run_batch_outputs(tpl, [{"model.nx": 4}])

    def test_row_to_overrides_internal(self) -> None:
        """_row_to_overrides 内部：无 '.' 键跳过 / 未知节点跳过."""
        from zylab.flowchart.batch import _row_to_overrides

        tpl = _template("structural.cantilever_static")
        # 不含 . 的键被静默跳过
        r1 = _row_to_overrides(tpl, {"plain_key": 5})
        assert r1 == {}
        # 未知节点 id 也被跳过（抛 ValueError 被捕获）
        r2 = _row_to_overrides(tpl, {"unknown_node.x": 5, "model.nx": 4})
        assert "model" in r2
        assert "unknown_node" not in r2

    def test_row_to_overrides_int_rounds_float(self) -> None:
        """INT 类型 param 收到 float 自动 round —— 适配 DOE 采样浮点."""
        from zylab.flowchart.batch import _row_to_overrides

        tpl = _template("structural.cantilever_static")
        r = _row_to_overrides(tpl, {"model.nx": 10.7, "model.ny": 4.2})
        # nx 是 INT param，10.7 → 11；ny 是 INT param，4.2 → 4
        assert r["model"]["nx"] == 11
        assert r["model"]["ny"] == 4

    def test_run_batch_outputs_skips_failed_rows(self) -> None:
        """run_batch_outputs 混合成功+失败 → 跳过失败行只返回成功的 X/Y."""
        from unittest.mock import patch

        tpl = _template("structural.cantilever_static")
        new_tpl = dataclasses.replace(
            tpl,
            output_params=(OutputParam(name="e", source="solve.strain_energy"),),
        )
        ok = RunOutcome((NodeOutcome(node_id="solve", name="s", result=object()),))
        bad = RunOutcome((NodeOutcome(node_id="solve", name="s", error="SolverError: failed"),))

        with patch.object(RunOutcome, "resolve_outputs", return_value={"e": 0.5}), patch(
            "zylab.flowchart.batch.run_batch", return_value=[ok, bad]
        ):
            X, Y = run_batch_outputs(new_tpl, [{"model.nx": 4}, {"model.nx": 5}])
            assert X.shape[0] == 1  # 只保留成功行
            assert Y.shape[0] == 1

    def test_run_batch_outputs_all_failed_raises(self) -> None:
        """全部运行失败 → FlowchartError 快速失败."""
        from unittest.mock import patch

        tpl = _template("structural.cantilever_static")
        new_tpl = dataclasses.replace(
            tpl,
            output_params=(OutputParam(name="e", source="solve.strain_energy"),),
        )
        bad = RunOutcome((NodeOutcome(node_id="solve", name="s", error="failed"),))

        with patch("zylab.flowchart.batch.run_batch", return_value=[bad, bad]), pytest.raises(
            Exception, match="所有运行均失败"
        ):
            run_batch_outputs(new_tpl, [{"model.nx": 4}, {"model.nx": 5}])

    def test_resolve_outputs_failed_runoutcome_raises(self) -> None:
        """失败 RunOutcome 调 resolve_outputs —— FlowchartError."""
        tpl = _template("structural.cantilever_static")
        new_tpl = dataclasses.replace(
            tpl,
            output_params=(OutputParam(name="e", source="solve.strain_energy"),),
        )
        failed = RunOutcome((NodeOutcome(node_id="solve", name="s", error="SolverError: boom"),))
        assert not failed.succeeded
        with pytest.raises(Exception, match="失败节点"):
            failed.resolve_outputs(new_tpl)


class TestExploreDoe:
    """explore_doe 一行走完 DOE → batch → surrogate → Sobol 敏感性分解."""

    def test_full_pipeline_lhc(self) -> None:
        """完整链路：LHC 采样 24 点 + RBF surrogate + Sobol 敏感性."""
        from zylab.doe import DesignSpace, DesignVariable, SamplingMethod

        ds = DesignSpace.from_variables(
            [
                DesignVariable(name="model.nx", lower=4, upper=16),
                DesignVariable(name="model.ny", lower=2, upper=8),
            ]
        )
        tpl = _template("structural.cantilever_static")
        new_tpl = dataclasses.replace(
            tpl,
            output_params=(OutputParam(name="strain_energy", source="solve.strain_energy"),),
        )
        result = explore_doe(
            new_tpl,
            ds,
            n_samples=24,
            method=SamplingMethod.LATIN_HYPERCUBE,
            fit_surrogate=True,
            sensitivity=True,
            sobol_N=512,
            seed=42,
        )
        # X/Y 维度
        assert result.X.shape == (24, 2)
        assert result.Y.shape == (24, 1)
        # surrogate 已拟合
        assert result.surrogate is not None
        # Sobol 指数归一化在 [0,1]
        assert result.si is not None
        assert result.sti is not None
        assert np.all(result.si >= -0.1) and np.all(result.si <= 1.1)
        assert np.all(result.sti >= -0.1) and np.all(result.sti <= 1.1)
        assert result.variance is not None and result.variance > 0

    def test_basic_no_surrogate(self) -> None:
        """最小配置：只做 DOE 批量求解，不拟合 surrogate 也不做敏感性."""
        from zylab.doe import DesignSpace, DesignVariable

        ds = DesignSpace.from_variables([DesignVariable(name="model.nx", lower=4, upper=12)])
        tpl = _template("structural.cantilever_static")
        new_tpl = dataclasses.replace(
            tpl,
            output_params=(OutputParam(name="strain_energy", source="solve.strain_energy"),),
        )
        result = explore_doe(new_tpl, ds, n_samples=8, fit_surrogate=False, sensitivity=False)
        assert result.X.shape == (8, 1)
        assert result.Y.shape == (8, 1)
        assert result.surrogate is None
        assert result.si is None
        assert result.sti is None
        assert result.variance is None

    def test_optimize_true_minimize_returns_best(self) -> None:
        """optimize=True（默认 minimize）→ 返回 best_x / best_y / best_x_int / optimizer."""
        from zylab.doe import DesignSpace, DesignVariable, SamplingMethod

        ds = DesignSpace.from_variables(
            [DesignVariable(name="model.nx", lower=4, upper=12), DesignVariable(name="model.ny", lower=2, upper=6)]
        )
        tpl = _template("structural.cantilever_static")
        new_tpl = dataclasses.replace(
            tpl,
            output_params=(OutputParam(name="strain_energy", source="solve.strain_energy"),),
        )
        result = explore_doe(
            new_tpl,
            ds,
            n_samples=12,
            method=SamplingMethod.LATIN_HYPERCUBE,
            optimize=True,
            opt_n_iter=30,
            seed=42,
        )
        assert result.best_x is not None
        assert result.best_y is not None
        assert result.best_x_int is not None
        assert result.optimizer == "differential_evolution"
        # best_x_int 在边界内
        assert 4 <= result.best_x_int[0] <= 12
        assert 2 <= result.best_x_int[1] <= 6
        # best_y 是真实应变能（正的、物理合理）
        assert result.best_y > 0

    def test_optimize_true_maximize_uses_neg_surrogate(self) -> None:
        """minimize=False → 内部用 _NegSurrogate 包装代理，返回的 best_y 取原始尺度."""
        from zylab.doe import DesignSpace, DesignVariable

        ds = DesignSpace.from_variables(
            [DesignVariable(name="model.nx", lower=4, upper=12), DesignVariable(name="model.ny", lower=2, upper=6)]
        )
        tpl = _template("structural.cantilever_static")
        new_tpl = dataclasses.replace(
            tpl,
            output_params=(OutputParam(name="strain_energy", source="solve.strain_energy"),),
        )
        result = explore_doe(
            new_tpl,
            ds,
            n_samples=12,
            optimize=True,
            opt_n_iter=30,
            minimize=False,
            seed=42,
        )
        assert result.best_y is not None
        # 最大化后仍应有真实正值（验证环节取真实求解值）
        assert result.best_y > 0
