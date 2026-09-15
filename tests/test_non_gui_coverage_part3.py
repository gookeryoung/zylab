"""第二批补充测试 —— 第三部分：核心路径/异常分支覆盖.

目标文件与行号：
- src/zylab/core/executor.py
- src/zylab/fea/buckling.py
- src/zylab/fea/modal.py
- src/zylab/fea/nonlinear.py
- src/zylab/fea/thermal_transient.py
- src/zylab/fea/elements.py
- src/zylab/reliability/form.py
- src/zylab/optim/optimize.py
"""

from __future__ import annotations

import importlib
import inspect
import queue
import threading
from typing import Any

import numpy as np
import psutil
import pytest
from scipy import sparse
from scipy.linalg import LinAlgError

from zylab.core.executor import (
    EventKind,
    ProcessExecutor,
    TaskEvent,
    TaskHandle,
    TaskSpec,
    _kill_process_tree,
    _worker_main,
)

# ======================================================================
# 1. executor.py:137-138 —— inspect.signature 对 C 扩展函数失败
# ======================================================================


def test_worker_main_inspect_signature_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """inspect.signature 抛 TypeError 时应回退到 accepts_report=False."""
    import tests._targets as tgt

    def _fake_no_report(*args: Any, **kwargs: Any) -> str:
        if "report" in kwargs:
            raise TypeError("不应被注入 report")
        return "ok"

    tgt.fake_no_report = _fake_no_report

    monkeypatch.setattr(inspect, "signature", lambda _f: (_ for _ in ()).throw(TypeError("no sig")))

    # 用 threading + mock close/join_thread 的队列（避免 multiprocessing fork 开销）
    class _FakeQueue(queue.Queue):
        def close(self) -> None:  # _worker_main 最后会调
            pass

        def join_thread(self) -> None:  # _worker_main 最后会调
            pass

    eq = _FakeQueue()
    spec = TaskSpec(target="tests._targets:fake_no_report", args=(), kwargs={})
    _worker_main(spec, eq)

    events: list[TaskEvent] = []
    while not eq.empty():
        events.append(eq.get())
    kinds = [e.kind for e in events]
    assert EventKind.STARTED in kinds
    assert EventKind.RESULT in kinds
    assert events[-1].payload == "ok"


# ======================================================================
# 2. executor.py:350 —— _watch 中 terminated=True 且进程已退出
# ======================================================================


class _FakeProcess:
    """模拟 multiprocessing.Process：可控 is_alive / exitcode / pid."""

    def __init__(self, alive_rounds: list[bool] | None = None) -> None:
        self._alive_rounds = list(alive_rounds) if alive_rounds is not None else [True, False]
        self.exitcode = 0
        self.pid = 12345

    def is_alive(self) -> bool:
        if self._alive_rounds:
            return self._alive_rounds.pop(0)
        return False

    def join(self, timeout: float = 0) -> None:
        pass

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


def test_watch_break_on_terminated_and_dead_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """收到终态事件且进程已死时，_watch 应通过 line 350 break.

    关键构造：patch handle._done.set() 使其无效，这样收到 RESULT 事件后
    terminated=True 但 handle.done=False；进程在第二轮 is_alive()=False，
    命中 line 350 的 ``terminated and not process.is_alive()`` 分支.
    """

    class _FakeQueue(queue.Queue):
        def close(self) -> None:  # _watch 最后会调
            pass

        def join_thread(self) -> None:  # _watch 最后会调
            pass

    executor = ProcessExecutor()
    handle = TaskHandle("fake-task-id")
    # alive 序列：第一轮 alive=True（不触发 350），第二轮及之后 False
    proc = _FakeProcess(alive_rounds=[True, False])
    eq = _FakeQueue()
    eq.put(TaskEvent(task_id="fake-task-id", kind=EventKind.STARTED))
    eq.put(TaskEvent(task_id="fake-task-id", kind=EventKind.RESULT, payload=42))

    # patch handle._done.set 使其为空操作 → handle.done 永不置位
    monkeypatch.setattr(handle._done, "set", lambda: None)

    watcher = threading.Thread(
        target=executor._watch,
        args=(handle, proc, eq),
        name="test-watch",
        daemon=True,
    )
    watcher.start()
    watcher.join(timeout=2.0)
    assert not watcher.is_alive(), "_watch 应在 terminated+dead 后退出"


# ======================================================================
# 3. executor.py:110-111 —— _kill_process_tree 中 proc.kill() 抛 NoSuchProcess
# ======================================================================


def test_kill_process_tree_proc_already_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    """psutil.Process.kill() 抛 NoSuchProcess 应被捕获并静默返回."""

    class _FakeProc:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def children(self, recursive: bool = False) -> list[Any]:
            return []

        def kill(self) -> None:
            raise psutil.NoSuchProcess(self.pid)

    monkeypatch.setattr(psutil, "Process", _FakeProc)
    _kill_process_tree(99999)  # 不应抛异常


# ======================================================================
# 4. fea/buckling.py:122-123 —— linalg.eig 抛 LinAlgError
# ======================================================================


def test_buckling_eig_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """scipy.linalg.eig 抛 LinAlgError 应包装为 SolverError."""
    from zylab.fea import (
        Constraint,
        ElementBlock,
        ElementType,
        LinearElastic,
        Mesh,
        NodalLoad,
        Section,
        SolverError,
        StaticCase,
    )
    from zylab.fea.buckling import solve_buckling

    mesh = Mesh(
        coords=np.array([[0.0, 0.0], [1.0, 0.0]]),
        blocks=(ElementBlock(etype=ElementType.BEAM2, conn=np.array([[0, 1]]), material=0, section=0),),
    )
    material = LinearElastic(2.1e5)
    section = Section(area=0.01, inertia=1e-4)
    case = StaticCase(
        constraints=(Constraint(0, (0, 1, 2)),),
        loads=(NodalLoad(1, (0.0, -1.0, 0.0)),),
    )

    def _eig_fails(*args: Any, **kwargs: Any) -> Any:
        raise LinAlgError("奇异矩阵无法分解")

    # buckling.py 用 ``from scipy import linalg``，模块内含自己命名空间
    monkeypatch.setattr("scipy.linalg.eig", _eig_fails)
    # 同时 patch 模块自己的引用（from scipy import linalg → buckling.linalg.eig）
    from zylab.fea import buckling as buckling_mod

    monkeypatch.setattr(buckling_mod.linalg, "eig", _eig_fails)

    with pytest.raises(SolverError, match="屈曲特征值求解失败"):
        solve_buckling(mesh, [material], [section], case, n_modes=1)


# ======================================================================
# 5. fea/modal.py:116-117 —— eigsh 失败
# ======================================================================


def test_modal_eigsh_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """scipy.sparse.linalg.eigsh 抛 RuntimeError 应包装为 SolverError."""
    from zylab.fea import (
        Constraint,
        ElementBlock,
        ElementType,
        LinearElastic,
        Mesh,
        Section,
        SolverError,
        solve_modal,
    )
    from zylab.fea import modal as modal_mod

    mesh = Mesh(
        coords=np.array([[0.0, 0.0], [1.0, 0.0]]),
        blocks=(ElementBlock(etype=ElementType.TRUSS2, conn=np.array([[0, 1]]), material=0, section=0),),
    )
    material = LinearElastic(2.1e5, density=7.85e3)
    section = Section(area=0.01)
    constraints = [Constraint(0, (0, 1))]

    def _eigsh_fails(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("ARPACK converged zero eigenvalues")

    monkeypatch.setattr(modal_mod, "eigsh", _eigsh_fails)

    with pytest.raises(SolverError, match="特征值求解失败"):
        solve_modal(mesh, [material], [section], constraints, n_modes=1)


# ======================================================================
# 6. fea/nonlinear.py:254-255 —— splu 失败
# ======================================================================


def test_nonlinear_splu_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """切线刚度求解时 splu 抛 RuntimeError 应包装为 SolverError."""
    from zylab.fea import nonlinear as nl_mod
    from zylab.fea.errors import SolverError

    k = sparse.eye(3, format="csr").tocsr()
    free = np.array([0, 1, 2], dtype=np.int64)
    residual = np.array([1.0, 2.0, 3.0])

    def _splu_fails(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Singular matrix")

    monkeypatch.setattr(nl_mod, "splu", _splu_fails)

    with pytest.raises(SolverError, match="切线刚度奇异"):
        nl_mod._solve_free(k, residual, free)


# ======================================================================
# 7. fea/thermal_transient.py:137-138 —— splu 失败
# ======================================================================


def test_thermal_transient_splu_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """瞬态有效矩阵 LU 分解失败应抛 SolverError."""
    from zylab.fea import (
        ConductionMaterial,
        ElementBlock,
        ElementType,
        Mesh,
        NodalValue,
        Section,
        SolverError,
        ThermalCase,
        solve_thermal_transient,
    )
    from zylab.fea import thermal_transient as tt_mod

    # QUAD4 支持标量场传导（thermal）
    mesh = Mesh(
        coords=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        ),
        blocks=(ElementBlock(etype=ElementType.QUAD4, conn=np.array([[0, 1, 2, 3]]), material=0, section=0),),
    )
    mat = ConductionMaterial(electric_sigma=1.0, thermal_k=1.0, volumetric_heat_capacity=1.0)
    sec = Section()

    case = ThermalCase(temperatures=(NodalValue(0, 100.0),))

    def _splu_fails(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Singular matrix")

    monkeypatch.setattr(tt_mod, "splu", _splu_fails)

    with pytest.raises(SolverError, match="瞬态有效矩阵奇异"):
        solve_thermal_transient(
            mesh,
            [mat],
            [sec],
            case,
            initial=np.zeros(mesh.n_nodes),
            total_time=1.0,
            n_steps=2,
        )


# ======================================================================
# 8. fea/elements.py:27 —— NUMBA_DISABLE_JIT 分支（显式 JIT 启用路径）
# ======================================================================


def test_elements_jit_branch_with_numba(monkeypatch: pytest.MonkeyPatch) -> None:
    """numba 可用且 NUMBA_DISABLE_JIT != '1' 时，_JIT 应为真实 JIT 装饰器."""
    monkeypatch.delenv("NUMBA_DISABLE_JIT", raising=False)
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "0")

    import numba  # noqa: F401 —— 确认环境已装

    from zylab.fea import elements

    elements = importlib.reload(elements)

    assert elements._JIT is not None
    assert callable(elements._JIT)

    # 验证 numba 装饰的函数可正常调用（数值不敏感，只测能跑通）
    coords = np.array([[0.0, 0.0], [2.0, 0.0]])
    u = np.array([0.0, 0.0, 0.0, 0.0])
    f = elements.truss2_internal_force(coords, u, 1000.0, 2.0)
    assert f.shape == (4,)


# ======================================================================
# 9. reliability/form.py:448-450 —— SORM norm_g < 1e-14
# ======================================================================


def test_sorm_gradient_norm_too_small(monkeypatch: pytest.MonkeyPatch) -> None:
    """标准化空间梯度 norm 为零时应直接用 FORM 结果返回（跳过曲率计算）."""
    from zylab.reliability import Distribution, RandomVariable, sorm_analysis
    from zylab.reliability import form as form_mod

    R = RandomVariable("R", Distribution.NORMAL, {"loc": 100.0, "scale": 10.0})
    S = RandomVariable("S", Distribution.NORMAL, {"loc": 60.0, "scale": 8.0})
    vars_ = [R, S]

    # 调用计数：FORM 内部调 2 次 _grad_x_to_u，SORM 又调 1 次（line 445）
    call_count = {"n": 0}
    original = form_mod._grad_x_to_u

    def _fake_grad_x_to_u(grad_x: np.ndarray, x: np.ndarray, variables: Any) -> np.ndarray:
        call_count["n"] += 1
        # 前 2 次（FORM 迭代）返回正常值让收敛；第 3 次（SORM）返回接近零值
        if call_count["n"] <= 2:
            return original(grad_x, x, variables)
        return np.array([0.0, 1e-300])

    monkeypatch.setattr(form_mod, "_grad_x_to_u", _fake_grad_x_to_u)

    res = sorm_analysis(
        lambda x: float(x[0] - x[1]),
        vars_,
        grad=lambda _x: np.array([1.0, -1.0]),
    )
    # norm_g < 1e-14 路径：kappa 应为空，pf_breitung / pf_hohenbichler 等于 pf_form
    assert res.kappa.size == 0
    assert res.pf_breitung == pytest.approx(res.form.pf_form)
    assert res.pf_hohenbichler == pytest.approx(res.form.pf_form)


# ======================================================================
# 10. optim/optimize.py:171-173 —— surrogate.predict_std 返回非 None
# ======================================================================


def test_optimize_gpr_predict_std() -> None:
    """GprSurrogate 支持 predict_std，optimize 应把 std 写入 OptimResult."""
    from zylab.doe import DesignVariable
    from zylab.optim import GprSurrogate, Optimizer, optimize

    rng = np.random.default_rng(0)
    X = rng.uniform(-5, 5, size=(40, 1))
    y = (X**2).ravel()

    gpr = GprSurrogate().fit(X, y)
    vars_ = [DesignVariable.continuous("x", -5.0, 5.0)]

    result = optimize(gpr, vars_, optimizer=Optimizer.DIFFERENTIAL_EVOLUTION, n_iter=30, seed=0)
    assert result.best_y < 1.0  # 全局最优接近 0
    # GprSurrogate.predict_std 返回非 None 时，best_std 应被正确赋值
    assert result.best_std is not None
    assert result.best_std >= 0.0
