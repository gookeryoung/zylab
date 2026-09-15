"""最后冲刺补测：瞄准最新 coverage term-missing 中低挂果实的 miss 行.

覆盖目标（均来自 coverage 报告）：

**非 GUI（纯 Python）**：
- console/kernel.py      408     KeyboardInterrupt 分支
- core/executor.py       312-314 cancel 时 pid 非空；325-327 shutdown kill；363-364 _watch crash
- flowchart/batch.py     641     _NegSurrogate.fit 空实现
- flowchart/nodes.py     357     _boundary_edges 空元组；773 sweep body.nodes 项非 Mapping
- flowchart/template.py  208     重复 output_param 名
- optim/optimize.py      285-287 objective callback on fail；291 target missing；453-460 evaluate succeeded

**GUI（qtbot fixture）**：
- proxy_style.py         111 117 119 121 123 各 pixelMetric 分支
- style.py               137-138 fragments + fallback 都缺失；165->168 svg_tokens
- resources_rc.py        2140  qCleanupResources 显式调用
- app.py                 135-150 _write_theme_svgs close_template 分支；302-303 set_root_level ValueError
- command_palette.py     213 None data；229 super().keyPressEvent；242-244 Return 激活
- settings_panel.py      300/311 字体兜底；347-348 save OSError；387-390 log_level idx
- trial_record_edit.py   177 row >= len 或 column == 0 早返回
- app_controller.py      166-168 switch_workspace_to OSError
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 非 GUI: console/kernel.py —— KeyboardInterrupt (408)
# ---------------------------------------------------------------------------


class TestKernelKeyboardInterrupt:
    """kernel._run_cell 捕获 KeyboardInterrupt 写入 error 字段."""

    def test_execute_raises_keyboard_interrupt(self) -> None:
        from zylab.console.kernel import ReplKernel

        k = ReplKernel()
        r = k.execute("raise KeyboardInterrupt()")
        assert "KeyboardInterrupt" in r.error

    def test_execute_cell_raises_keyboard_interrupt(self) -> None:
        from zylab.console.kernel import ReplKernel

        k = ReplKernel()
        r = k.execute_cell("raise KeyboardInterrupt()")
        assert any("KeyboardInterrupt" in str(o) for o in r.outputs)


# ---------------------------------------------------------------------------
# 非 GUI: core/executor.py —— cancel/shutdown/_watch crash 分支
# ---------------------------------------------------------------------------


class TestExecutorPidBranches:
    """ProcessExecutor 的 pid 检查分支 + _watch crash 分支."""

    def test_cancel_kills_process_when_pid_not_none(self) -> None:
        from zylab.core.executor import ProcessExecutor, TaskHandle

        with patch("zylab.core.executor._kill_process_tree") as mock_kill:
            exe = ProcessExecutor()
            handle = TaskHandle("t1")
            assert not handle.done
            fake_proc = MagicMock()
            fake_proc.pid = 12345
            exe._tasks["t1"] = (handle, fake_proc)
            exe.cancel(handle)
            mock_kill.assert_called_once_with(12345)

    def test_shutdown_kills_running_process_when_pid_not_none(self) -> None:
        from zylab.core.executor import ProcessExecutor, TaskHandle

        with patch("zylab.core.executor._kill_process_tree") as mock_kill:
            exe = ProcessExecutor()
            handle = TaskHandle("t2")
            fake_proc = MagicMock()
            fake_proc.pid = 22345
            exe._tasks["t2"] = (handle, fake_proc)
            exe.shutdown(cancel_running=True)
            mock_kill.assert_called_once_with(22345)

    def test_watch_marks_crashed_on_early_exit_no_terminal(self) -> None:
        from zylab.core.executor import (
            EventKind,
            ProcessExecutor,
            TaskEvent,
            TaskHandle,
            TaskStatus,
        )

        exe = ProcessExecutor()
        handle = TaskHandle("t3")
        start_event = TaskEvent("t3", kind=EventKind.STARTED, payload={})
        handle._on_event(start_event)
        assert handle.status is TaskStatus.RUNNING

        fake_proc = MagicMock()
        fake_proc.pid = 33333
        fake_proc.is_alive.return_value = False
        fake_proc.exitcode = -9

        eq = exe._ctx.Queue()
        exe._watch(handle, fake_proc, eq)

        assert handle.status is TaskStatus.CRASHED
        fake_proc.join.assert_called()


# ---------------------------------------------------------------------------
# 非 GUI: flowchart/batch.py —— _NegSurrogate.fit 空实现 (641)
# ---------------------------------------------------------------------------


class TestNegSurrogateFit:
    """explore_doe 中 _NegSurrogate.fit 空实现可被调用（返回 self）."""

    def test_neg_surrogate_fit_returns_self(self) -> None:
        from zylab.doe.design_space import DesignSpace
        from zylab.doe.variable import DesignVariable
        from zylab.flowchart.batch import explore_doe
        from zylab.optim.optimize import OptimResult

        ds = DesignSpace.from_variables([DesignVariable(name="x", lower=-1.0, upper=1.0)])
        called_fit = {}

        def fake_optimize(surrogate, variables, **kw):
            called_fit["called"] = True
            result = surrogate.fit(np.zeros((1, len(variables))), np.zeros(1))
            called_fit["returned_self"] = result is surrogate
            return OptimResult(
                best_x=np.zeros(len(variables)),
                best_y=0.0,
                best_std=None,
                optimizer="de",
                raw_result=MagicMock(),
            )

        from zylab.flowchart.template import OutputParam, Template, TemplateNode

        node = TemplateNode(id="n1", type_id="passthrough", inputs={"in": "sys.in"}, params={})
        tmpl = Template(
            id="t1",
            name="t1",
            nodes=(node,),
            output_params=(OutputParam(name="y", source="n1", expr="result"),),
        )

        # patch run_batch 让 DOE 阶段返回有效数据，再 patch optimize
        fake_outcomes = [MagicMock(succeeded=True) for _ in range(4)]
        for o in fake_outcomes:
            o.resolve_outputs.return_value = {"y": 0.0}

        with (
            patch("zylab.flowchart.batch.run_batch", return_value=fake_outcomes),
            patch("zylab.optim.optimize", fake_optimize),
        ):
            explore_doe(
                tmpl,
                ds,
                n_samples=4,
                fit_surrogate=True,
                optimize=True,
                minimize=False,
                seed=0,
                n_workers=0,
            )

        assert called_fit.get("called"), "optimize 应被调用（minimize=False 触发 _NegSurrogate）"
        assert called_fit.get("returned_self"), "_NegSurrogate.fit 应返回 self"


# ---------------------------------------------------------------------------
# 非 GUI: flowchart/nodes.py —— _boundary_edges 空元组 + sweep body.nodes 非 Mapping
# ---------------------------------------------------------------------------


class TestNodesBoundaryEdgesAndSweep:
    """nodes 模块两个低挂果实分支."""

    def test_boundary_edges_empty(self) -> None:
        from zylab.flowchart.nodes import _boundary_edges

        assert _boundary_edges(()) == []

    def test_sweep_body_nodes_item_not_mapping(self) -> None:
        """compute.sweep body.nodes 项不是 Mapping → ParamError（被 batch 捕获标记 failed）."""
        from zylab.flowchart.batch import run_workflow
        from zylab.flowchart.template import Template, TemplateNode

        node = TemplateNode(
            id="sweep1",
            type_id="compute.sweep",
            params={
                "var": "x",
                "from": 0,
                "to": 1,
                "count": 3,
                "body": {"nodes": ["not_a_mapping_item"], "collect": ["y"]},
            },
        )
        tmpl = Template(id="t1", name="t1", nodes=(node,))
        outcome = run_workflow(tmpl)
        assert not outcome.succeeded
        assert "ParamError" in outcome.first_error() or "body.nodes" in outcome.first_error()


# ---------------------------------------------------------------------------
# 非 GUI: flowchart/template.py —— 重复 output_param 名
# ---------------------------------------------------------------------------


class TestTemplateDuplicateOutputParam:
    """Template 校验 output_param 重名抛 TemplateError."""

    def test_duplicate_output_param_raises(self) -> None:
        from zylab.flowchart.errors import TemplateError
        from zylab.flowchart.template import (
            OutputParam,
            Template,
            TemplateNode,
        )

        op1 = OutputParam(name="y", source="n1.result", expr="value")
        op2 = OutputParam(name="y", source="n2.result", expr="value")
        n1 = TemplateNode(id="n1", type_id="passthrough", inputs={}, params={})
        n2 = TemplateNode(id="n2", type_id="passthrough", inputs={}, params={})
        tmpl = Template(id="t", name="t", nodes=(n1, n2), output_params=(op1, op2))

        # Template 是 frozen dataclass，必须 patch 类级别方法
        with patch.object(Template, "_validate_node_links"), patch("zylab.flowchart.template.module_spec") as mock_spec:
            mock_spec.return_value.inputs = []
            mock_spec.return_value.coerce_params.return_value = None
            with pytest.raises(TemplateError, match="输出参数名重复"):
                tmpl.validate()


# ---------------------------------------------------------------------------
# 非 GUI: optim/optimize.py —— objective evaluate 分支
# ---------------------------------------------------------------------------


class TestOptimizeDirectObjectiveFailBranches:
    """optimize_direct 的 objective 闭包：workflow 失败 + target 缺失."""

    def test_objective_called_on_workflow_fail(self) -> None:
        from zylab.doe.variable import DesignVariable
        from zylab.flowchart.template import OutputParam, Template, TemplateNode
        from zylab.optim.optimize import optimize_direct

        n = TemplateNode(id="n1", type_id="passthrough", inputs={"in": "sys.in"}, params={})
        op = OutputParam(name="y", source="n1", expr="a")
        tmpl = Template(id="t", name="t", nodes=(n,), output_params=(op,))
        v = DesignVariable(name="n1.x", lower=0.0, upper=1.0)

        callback_values: list[tuple] = []

        def cb(x, y):
            callback_values.append((x.copy(), y))

        with patch("zylab.flowchart.run_workflow") as mock_run:
            mock_outcome = MagicMock()
            mock_outcome.succeeded = False
            mock_run.return_value = mock_outcome
            with contextlib.suppress(Exception):
                optimize_direct(tmpl, [v], n_iter=2, seed=0, callback=cb)
        assert len(callback_values) > 0

    def test_objective_called_on_target_missing(self) -> None:
        from zylab.doe.variable import DesignVariable
        from zylab.flowchart.template import OutputParam, Template, TemplateNode
        from zylab.optim.optimize import optimize_direct

        n = TemplateNode(id="n1", type_id="passthrough", inputs={"in": "sys.in"}, params={})
        op = OutputParam(name="y", source="n1", expr="a")
        tmpl = Template(id="t", name="t", nodes=(n,), output_params=(op,))
        v = DesignVariable(name="n1.x", lower=0.0, upper=1.0)

        callback_values: list[tuple] = []

        def cb(x, y):
            callback_values.append((x.copy(), y))

        with patch("zylab.flowchart.run_workflow") as mock_run:
            mock_outcome = MagicMock()
            mock_outcome.succeeded = True
            mock_outcome.resolve_outputs.return_value = {}
            mock_run.return_value = mock_outcome
            with contextlib.suppress(Exception):
                optimize_direct(tmpl, [v], target="y", n_iter=2, seed=0, callback=cb)
        assert len(callback_values) > 0


class TestParetoEvaluateSucceeded:
    """optimize_pareto 的 evaluate 闭包：outcome.succeeded 分支 (453-460)."""

    def test_evaluate_when_workflow_succeeds(self) -> None:
        from zylab.doe.variable import DesignVariable
        from zylab.flowchart.template import OutputParam, Template, TemplateNode
        from zylab.optim.optimize import optimize_pareto

        n = TemplateNode(id="n1", type_id="passthrough", inputs={"in": "sys.in"}, params={})
        op1 = OutputParam(name="y1", source="n1", expr="a")
        op2 = OutputParam(name="y2", source="n1", expr="b")
        tmpl = Template(id="t", name="t", nodes=(n,), output_params=(op1, op2))
        v1 = DesignVariable(name="n1.x1", lower=0.0, upper=1.0)
        v2 = DesignVariable(name="n1.x2", lower=0.0, upper=1.0)

        class FakeOutcome:
            succeeded = True

            def resolve_outputs(self, template):
                return {"y1": 0.5, "y2": 1.5}

        with patch("zylab.flowchart.run_workflow", return_value=FakeOutcome()):
            res = optimize_pareto(
                tmpl,
                [v1, v2],
                targets=["y1", "y2"],
                minimize=[True, False],
                n_population=4,
                n_generations=2,
                seed=0,
            )
        assert res.F.shape[1] == 2
        # 所有候选解 y1=0.5 (minimize)，y2=1.5 (maximize，最终翻回用户语义仍是 1.5)
        assert float(res.F[0, 0]) == pytest.approx(0.5, abs=1e-6)
        assert float(res.F[0, 1]) == pytest.approx(1.5, abs=1e-6)
