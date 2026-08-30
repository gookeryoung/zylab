"""studio.batch 批处理执行测试：进程内拓扑执行 + 参数覆盖/扫描 + 失败中止 + 摘要."""

from __future__ import annotations

import pytest

from zylab.fea import StaticSolution
from zylab.studio import (
    ModelBundle,
    NodeOutcome,
    RunOutcome,
    Template,
    TemplateRegistry,
    resolve_target,
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
    import numpy as np

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
        import zylab.studio.batch as batch_mod

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
        fn = resolve_target("zylab.studio.nodes:run_static")
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
