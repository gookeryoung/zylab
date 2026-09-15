"""最终冲刺：non-GUI 模块剩余分支覆盖."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


class TestConfigWinAndVersionBranch:
    """core/config.py 的 win32 分支."""

    def test_default_data_dir_win32_with_appdata(self, monkeypatch) -> None:
        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setenv("APPDATA", r"C:\Users\x\AppData\Roaming")
        from zylab.core.config import default_data_dir

        result = default_data_dir()
        assert "zylab" in str(result)

    def test_default_data_dir_win32_without_appdata(self, monkeypatch) -> None:
        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.delenv("APPDATA", raising=False)
        from zylab.core.config import default_data_dir

        result = default_data_dir()
        assert "zylab" in str(result)
        assert "AppData" in str(result)


class TestExecutorCrashBranches:
    """core/executor.py 的进程崩溃 / pid None 分支."""

    def test_cancel_when_process_pid_is_none(self, monkeypatch) -> None:
        from zylab.core.executor import ProcessExecutor, TaskHandle

        killed = []
        monkeypatch.setattr("zylab.core.executor._kill_process_tree", killed.append)

        executor = ProcessExecutor()
        handle = TaskHandle("t1")
        handle._cancel()

        fake_process = SimpleNamespace(pid=None)
        with executor._lock:
            executor._tasks["t1"] = (handle, fake_process)

        executor.cancel(handle)
        assert killed == []

    def test_shutdown_cancel_pid_none(self, monkeypatch) -> None:
        from zylab.core.executor import ProcessExecutor, TaskHandle

        killed = []
        monkeypatch.setattr("zylab.core.executor._kill_process_tree", killed.append)

        executor = ProcessExecutor()
        handle = TaskHandle("t2")
        fake_process = SimpleNamespace(pid=None, join=lambda timeout: None)
        with executor._lock:
            executor._tasks["t2"] = (handle, fake_process)

        executor.shutdown(cancel_running=True)
        assert killed == []

    def test_watch_running_status_to_crashed(self) -> None:
        import queue

        from zylab.core.executor import ProcessExecutor, TaskHandle, TaskStatus

        handle = TaskHandle("t3")
        handle._status = TaskStatus.RUNNING

        fake_process = SimpleNamespace(
            pid=12345,
            is_alive=lambda: False,
            join=lambda timeout: None,
            exitcode=-9,
        )
        fake_queue = SimpleNamespace(
            get=lambda timeout=None: (_ for _ in ()).throw(queue.Empty),
            empty=lambda: True,
            close=lambda: None,
        )

        executor = ProcessExecutor()
        with executor._lock:
            executor._tasks["t3"] = (handle, fake_process)

        executor._watch(handle, fake_process, fake_queue)
        assert handle.status is TaskStatus.CRASHED
        assert handle.done


class TestOptimizeEdgeBranches:
    """optim/optimize.py 的 else 分支 + callback with penalty 分支."""

    def test_optimize_unknown_optimizer_raises(self, monkeypatch) -> None:
        from zylab.doe import DesignVariable
        from zylab.optim.errors import OptimError
        from zylab.optim.optimize import optimize

        # 伪造 Surrogate（跳过抽象基类实例化限制）
        s = object.__new__(type("FakeS", (), {"predict": lambda self, X: np.zeros(len(X)), "_fit": True}))
        variables = [DesignVariable("x", 0.0, 1.0)]
        with pytest.raises(OptimError):
            optimize(s, variables, optimizer="non_existent_optimizer_xyz", n_iter=1)

    def test_optimize_direct_workflow_failure_calls_callback(self, monkeypatch) -> None:
        from zylab.doe import DesignVariable
        from zylab.optim.optimize import optimize_direct

        class FakeOutcome:
            succeeded = False

        monkeypatch.setattr("zylab.flowchart.run_workflow", lambda *a, **k: FakeOutcome())

        template = SimpleNamespace(output_params=[SimpleNamespace(name="f")])
        variables = [DesignVariable("n1.x", 0.0, 1.0)]
        called = []
        optimize_direct(template, variables, n_iter=1, callback=lambda x, y: called.append((x.copy(), y)))
        assert called

    def test_optimize_direct_missing_target_calls_callback(self, monkeypatch) -> None:
        from zylab.doe import DesignVariable
        from zylab.optim.optimize import optimize_direct

        class FakeOutcome:
            succeeded = True

            def resolve_outputs(self, template):
                return {"other": 1.0}

        monkeypatch.setattr("zylab.flowchart.run_workflow", lambda *a, **k: FakeOutcome())

        template = SimpleNamespace(output_params=[SimpleNamespace(name="f")])
        variables = [DesignVariable("n1.x", 0.0, 1.0)]
        called = []
        optimize_direct(template, variables, n_iter=1, callback=lambda x, y: called.append((x.copy(), y)))
        assert called

    def test_optimize_direct_unknown_optimizer_raises(self) -> None:
        from zylab.doe import DesignVariable
        from zylab.optim.errors import OptimError
        from zylab.optim.optimize import optimize_direct

        template = SimpleNamespace(output_params=[SimpleNamespace(name="f")])
        variables = [DesignVariable("n1.x", 0.0, 1.0)]
        with pytest.raises(OptimError):
            optimize_direct(template, variables, optimizer="bad_opt_xyz", n_iter=1)

    def test_optimize_pareto_batch_evaluate_maximization(self, monkeypatch) -> None:
        from zylab.doe import DesignVariable
        from zylab.optim.optimize import optimize_pareto

        class FakeOutcome:
            def __init__(self, *_a, **_k):
                self.succeeded = True

            def resolve_outputs(self, template):
                return {"E": 200.0, "d": 3.0}

        monkeypatch.setattr(
            "zylab.flowchart.run_batch",
            lambda template, rows, n_workers=None, cache=None: [FakeOutcome() for _ in rows],
        )

        template = SimpleNamespace(output_params=[SimpleNamespace(name="E"), SimpleNamespace(name="d")])
        variables = [DesignVariable("n1.x", 0.0, 1.0)]
        # minimize=[False, True] → maximize E, minimize d
        result = optimize_pareto(
            template,
            variables,
            targets=["E", "d"],
            n_population=4,
            n_generations=1,
            minimize=[False, True],
            n_workers=2,
        )
        assert result.n_generations == 1


class TestEventsExitBranch:
    """core/events.py 的 exit 分支."""

    def test_events_on_exit(self) -> None:
        from zylab.core.events import EventBus

        bus = EventBus()
        received = []
        bus.subscribe("exit", received.append)
        bus.publish("exit", "bye")
        assert received == ["bye"]
