"""后台任务调度模块测试（TaskWorker + WorkerController + WorkerThreadPool）."""

from __future__ import annotations

import time
from typing import Any

import pytest

from zylab.gui.workers import (
    TaskWorker,
    WorkerController,
    WorkerTask,
    WorkerThreadPool,
    worker_pool,
)

# --------------------------------------------------------------------------- TaskWorker


class _SimpleWorker(TaskWorker):
    """测试用 Worker：累加 progress 直到 100，检查 is_cancelled 并提前退出."""

    def __init__(self, total: int = 5) -> None:
        super().__init__()
        self._total = total

    def run(self) -> None:
        for i in range(1, self._total + 1):
            if self.is_cancelled:
                self.message.emit("已取消")
                self.finished_ok.emit(None)
                return
            self.progress.emit(int(i * 100 / self._total))
            self.message.emit(f"步骤 {i}/{self._total}")
            time.sleep(0.01)
        self.finished_ok.emit(f"完成 {self._total} 步")


class TestTaskWorkerApi:
    """TaskWorker 基类 API（不启动 QThread，仅测试方法调用）."""

    def test_request_cancel(self) -> None:
        """request_cancel 设置取消标志."""
        worker = _SimpleWorker()
        assert not worker.is_cancelled
        worker.request_cancel()
        assert worker.is_cancelled

    def test_run_raises_not_implemented(self) -> None:
        """TaskWorker.run 未被子类实现时 raise NotImplementedError."""
        base = TaskWorker()
        with pytest.raises(NotImplementedError):
            base.run()


@pytest.mark.gui
@pytest.mark.skip(reason="PySide2 5.15.2.1 + pytest-qt 下 QThread 段错误，待 PySide2 5.15.3+ 修复后启用")
class TestTaskWorkerQt:
    """TaskWorker QThread 运行时测试（需要 Qt 事件循环）."""

    def test_progress_signal(self, qtbot) -> None:
        """Worker 运行时连续发 progress 信号."""
        worker = _SimpleWorker(total=5)
        controller = WorkerController(worker)

        progress_values: list[int] = []
        worker.progress.connect(progress_values.append)

        with qtbot.wait_signal(worker.finished_ok, timeout=2000):
            controller.start()

        assert len(progress_values) == 5
        assert progress_values[-1] == 100


class TestWorkerControllerApi:
    """WorkerController 非 QThread 启动时的安全行为."""

    def test_worker_property(self) -> None:
        """worker 属性返回正确的 TaskWorker."""
        worker = _SimpleWorker()
        controller = WorkerController(worker)
        assert controller.worker is worker

    def test_stop_when_not_running(self) -> None:
        """stop() 在线程未启动时安全 no-op."""
        worker = _SimpleWorker()
        controller = WorkerController(worker)
        controller.stop(wait_ms=100)  # 线程未 start，应立即返回

    def test_stop_async_when_not_running(self) -> None:
        """stop_async() 在线程未启动时安全 no-op."""
        worker = _SimpleWorker()
        controller = WorkerController(worker)
        controller.stop_async()

    def test_stop_when_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """stop() 在线程运行时发取消+quit+wait（monkeypatch QThread.isRunning）."""
        from zylab.gui.workers.task_worker import QThread as _QT

        worker = _SimpleWorker()
        controller = WorkerController(worker)

        calls: list[str] = []

        def _fake_running(self: _QT) -> bool:
            calls.append("isRunning")
            return True

        def _fake_quit(self: _QT) -> None:
            calls.append("quit")

        def _fake_wait(self: _QT, ms: int = 0) -> bool:
            calls.append(f"wait({ms})")
            return True

        monkeypatch.setattr(_QT, "isRunning", _fake_running)
        monkeypatch.setattr(_QT, "quit", _fake_quit)
        monkeypatch.setattr(_QT, "wait", _fake_wait)

        controller.stop(wait_ms=500)
        assert worker.is_cancelled
        assert calls == ["isRunning", "quit", "wait(500)"]

    def test_stop_async_when_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """stop_async() 在线程运行时发取消+quit（monkeypatch）."""
        from zylab.gui.workers.task_worker import QThread as _QT

        worker = _SimpleWorker()
        controller = WorkerController(worker)

        calls: list[str] = []

        def _fake_running(self: _QT) -> bool:
            calls.append("isRunning")
            return True

        def _fake_quit(self: _QT) -> None:
            calls.append("quit")

        monkeypatch.setattr(_QT, "isRunning", _fake_running)
        monkeypatch.setattr(_QT, "quit", _fake_quit)

        controller.stop_async()
        assert worker.is_cancelled
        assert calls == ["isRunning", "quit"]


# -------------------------------------------------------------------- WorkerTask


class TestWorkerTask:
    """QRunnable WorkerTask 基本行为."""

    def test_run_success(self) -> None:
        """简单函数执行完成，result 正确."""
        task = WorkerTask(lambda: 42)
        finished: list[Any] = []
        task.set_callbacks(on_finished=finished.append)
        task.run()
        assert task.result == 42
        assert task.error is None
        assert finished == [42]

    def test_run_failed(self) -> None:
        """函数抛异常时走 failed 回调."""

        def _boom() -> None:
            raise ValueError("爆炸")

        task = WorkerTask(_boom)
        failed: list[BaseException] = []
        task.set_callbacks(on_failed=failed.append)
        task.run()
        assert task.error is not None
        assert isinstance(task.error, ValueError)
        assert task.result is None
        assert len(failed) == 1
        assert str(failed[0]) == "爆炸"

    def test_cancel_request(self) -> None:
        """request_cancel 设置取消标志."""
        task = WorkerTask(lambda: None)
        assert not task.is_cancelled()
        task.request_cancel()
        assert task.is_cancelled()

    def test_capture_cancel(self) -> None:
        """capture_cancel=True 时 func 收到取消查询 callable."""
        seen: dict[str, bool] = {}

        def _work(cancel_requested: Any) -> int:
            seen["fn"] = cancel_requested
            return cancel_requested()

        task = WorkerTask(_work, capture_cancel=True)
        task.request_cancel()
        task.run()
        assert callable(seen["fn"])
        assert task.result is True

    def test_auto_delete_disabled(self) -> None:
        """setAutoDelete(False) 防止 QThreadPool 自动 delete."""
        task = WorkerTask(lambda: "ok")
        task.run()
        assert task.result == "ok"


# ------------------------------------------------------------- WorkerThreadPool


@pytest.mark.gui
class TestWorkerThreadPool:
    """线程池批量调度."""

    def test_submit_one_success(self, qtbot) -> None:
        """submit_one 成功回调在主线程被调用."""
        from PySide2.QtWidgets import QApplication

        pool = WorkerThreadPool()
        holder: dict[str, Any] = {}

        def _on_done(val: Any) -> None:
            holder["val"] = val

        pool.submit_one(lambda: "hello", on_finished=_on_done)

        app = QApplication.instance()
        for _ in range(500):
            app.processEvents()
            if "val" in holder:
                break
            time.sleep(0.01)

        assert holder["val"] == "hello"

    def test_submit_one_failed(self, qtbot) -> None:
        """submit_one 失败回调在主线程被调用."""
        from PySide2.QtWidgets import QApplication

        pool = WorkerThreadPool()
        holder: dict[str, BaseException] = {}

        def _on_fail(exc: BaseException) -> None:
            holder["exc"] = exc

        pool.submit_one(
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            on_failed=_on_fail,
        )

        app = QApplication.instance()
        for _ in range(500):
            app.processEvents()
            if "exc" in holder:
                break
            time.sleep(0.01)

        assert isinstance(holder["exc"], RuntimeError)
        assert str(holder["exc"]) == "boom"

    def test_submit_many(self, qtbot) -> None:
        """submit_many 批量调度，全部完成后触发 on_all_done."""
        from PySide2.QtWidgets import QApplication

        pool = WorkerThreadPool()
        each_results: list[Any] = []
        all_done_holder: dict[str, bool] = {}

        tasks = [WorkerTask(lambda i=i: i * 2) for i in range(5)]

        def _each(task: WorkerTask) -> None:
            each_results.append(task.result)

        def _all_done() -> None:
            all_done_holder["done"] = True

        pool.submit_many(tasks, on_each=_each, on_all_done=_all_done)

        # 手动事件循环（绕过 qtbot.waitUntil 可能的时序问题）
        app = QApplication.instance()
        for _ in range(500):
            app.processEvents()
            if all_done_holder.get("done"):
                break
            time.sleep(0.01)

        assert all_done_holder.get("done"), "on_all_done 未在超时内触发"
        assert sorted(each_results) == [0, 2, 4, 6, 8]

    def test_cancel_all(self) -> None:
        """cancel_all 设置所有活跃任务的取消标志."""
        pool = WorkerThreadPool()
        task = WorkerTask(lambda: time.sleep(10), capture_cancel=True)
        task.setAutoDelete(False)
        pool._active.add(task)
        pool.cancel_all()
        assert task.is_cancelled()

    def test_shared_pool(self) -> None:
        """worker_pool 是模块级共享实例."""
        assert isinstance(worker_pool, WorkerThreadPool)
        assert worker_pool.active_count == 0
