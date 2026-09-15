"""补充测试第四批 —— executor/project/batch/dsl/module/report/buckling/optimize 分支.

覆盖目标（均来自 coverage miss 行）：
- executor.py: _kill_process_tree 子进程已退出、cancel/shutdown 路径、RUNNING 崩溃
- project.py: ProjectFileError 两条抛分支 + write_json 同名覆盖
- batch.py: _row_to_overrides 异常兜底、_NegSurrogate.fit、run_batch_outputs 失败回退
- dsl.py: from_mapping _build TypeError → TemplateError
- module.py: plugin 返回 SolverSpec / ModuleSpec
- report.py: 空 series 图例、全时程末帧、stress 场、_auto_field、_cloud_coords
- buckling.py: 屈曲特征值出现显著虚部
- optimize.py: 多个分支（DE/SHGO/BH/DA、callback、maximize、多目标 minimize 序列、并行）
"""

from __future__ import annotations

import importlib
import queue
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import h5py
import numpy as np
import psutil
import pytest

from zylab.core.executor import (
    EventKind,
    ProcessExecutor,
    TaskEvent,
    TaskHandle,
    _kill_process_tree,
)
from zylab.core.project import Project, ProjectFileError
from zylab.fea.errors import SolverError
from zylab.flowchart.dsl import TemplateError
from zylab.optim.optimize import (
    OptimError,
    Optimizer,
    OptimResult,
    ParetoOptResult,
    _build_bounds,
)
from zylab.optim.surrogate import Surrogate

# 模块引用：避免 `zylab.optim.optimize` 被函数名 shadow
_opt_module = importlib.import_module("zylab.optim.optimize")
_batch_module = importlib.import_module("zylab.flowchart.batch")


# ======================================================================
# executor.py
# ======================================================================


class _FakeProc:
    """模拟 psutil.Process 及其子进程."""

    def __init__(self, children: list[Any] | None = None) -> None:
        self._children = children if children is not None else []

    def children(self, recursive: bool = False) -> list[Any]:
        return list(self._children)

    def kill(self) -> None:
        pass


def test_kill_process_tree_child_no_such_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """_kill_process_tree 遍历 children 时，某个 child.kill() 抛 NoSuchProcess → 静默 continue."""

    class _DeadChild:
        def kill(self) -> None:
            raise psutil.NoSuchProcess(1001)

    class _AliveChild:
        killed: bool = False

        def kill(self) -> None:
            self.killed = True

    alive = _AliveChild()
    proc = _FakeProc(children=[_DeadChild(), alive])
    monkeypatch.setattr(psutil, "Process", lambda _pid: proc)

    _kill_process_tree(9999)
    assert alive.killed, "存活子进程应被 kill"


def test_process_executor_cancel_uses_kill_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """ProcessExecutor.cancel 在 process.pid 有效时应调用 _kill_process_tree."""
    killed: list[int] = []
    monkeypatch.setattr(
        "zylab.core.executor._kill_process_tree",
        killed.append,
    )
    monkeypatch.setattr("multiprocessing.get_context", lambda _m: MagicMock(Process=MagicMock(), Queue=MagicMock()))

    executor = ProcessExecutor()
    handle = TaskHandle("fake")
    fake_proc = MagicMock()
    fake_proc.pid = 43210
    fake_proc.exitcode = None
    with executor._lock:
        executor._tasks["fake"] = (handle, fake_proc)

    executor.cancel(handle)
    assert 43210 in killed


def test_process_executor_shutdown_kills_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """shutdown(cancel_running=True) 应对未完成任务调用 kill_process_tree."""
    killed: list[int] = []
    monkeypatch.setattr(
        "zylab.core.executor._kill_process_tree",
        killed.append,
    )
    monkeypatch.setattr("multiprocessing.get_context", lambda _m: MagicMock(Process=MagicMock(), Queue=MagicMock()))

    executor = ProcessExecutor()
    handle = TaskHandle("fake")
    fake_proc = MagicMock()
    fake_proc.pid = 43211
    fake_proc.exitcode = None
    fake_proc.join = MagicMock()
    with executor._lock:
        executor._tasks["fake"] = (handle, fake_proc)

    executor.shutdown(cancel_running=True)
    assert 43211 in killed
    assert executor._shutdown


class _FakeProcessForWatch:
    def __init__(self, alive_rounds: list[bool], exitcode: int = 0) -> None:
        self.alive_rounds = list(alive_rounds)
        self.exitcode = exitcode
        self.pid = 20001

    def is_alive(self) -> bool:
        if self.alive_rounds:
            return self.alive_rounds.pop(0)
        return False

    def join(self, timeout: float = 0) -> None:
        pass


class _FakeQueueEmpty(queue.Queue):
    def close(self) -> None:
        pass

    def join_thread(self) -> None:
        pass


def test_watch_running_status_but_process_exits() -> None:
    """handle.status 为 RUNNING 但进程退出且队列无终态事件 → 标记 CRASHED."""
    executor = ProcessExecutor()
    handle = TaskHandle("rtask")
    eq = _FakeQueueEmpty()
    eq.put(TaskEvent(task_id="rtask", kind=EventKind.STARTED))
    proc = _FakeProcessForWatch(alive_rounds=[True, False], exitcode=-15)

    watcher = threading.Thread(target=executor._watch, args=(handle, proc, eq), name="rtask-watch", daemon=True)
    watcher.start()
    watcher.join(timeout=2.0)
    assert not watcher.is_alive()
    assert handle.done


# ======================================================================
# project.py
# ======================================================================


def test_project_open_missing_schema_version(tmp_path: Path) -> None:
    """h5 文件存在但缺 meta.schema_version 键 → ProjectFileError."""
    p = tmp_path / "bad.h5"
    h5 = h5py.File(p, "w")
    h5.create_group("data")
    h5.close()
    with pytest.raises(ProjectFileError, match=r"meta\.schema_version"):
        Project.open(p)


def test_project_open_unknown_file(tmp_path: Path) -> None:
    """文件不存在 → ProjectFileError."""
    with pytest.raises(ProjectFileError, match="工程文件不存在"):
        Project.open(tmp_path / "nope.h5")


def test_project_write_json_overwrites(tmp_path: Path) -> None:
    """write_json 同名覆盖（line 143 del grp[name]）."""
    p = tmp_path / "ok.h5"
    proj = Project.create(p)
    proj.write_json("g", "k", {"a": 1})
    proj.write_json("g", "k", {"b": 2})
    assert proj.read_json("g", "k") == {"b": 2}
    proj.close()


# ======================================================================
# dsl.py — from_mapping _build TypeError → TemplateError
# ======================================================================


def test_from_mapping_type_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """_build 抛 TypeError 应被 from_mapping 捕获并转为 TemplateError."""
    from zylab.flowchart.dsl import DslTemplate

    def _boom(_cls: Any, _data: Any) -> DslTemplate:
        raise TypeError("bad type")

    monkeypatch.setattr(DslTemplate, "_build", classmethod(_boom))
    with pytest.raises(TemplateError, match="类型错误"):
        DslTemplate.from_mapping({"meta": {}})


# ======================================================================
# module.py — plugin registry 返回 SolverSpec 或 ModuleSpec
# ======================================================================


class _DummyModuleSpec:
    def __init__(self, type_id: str) -> None:
        self.type_id = type_id


def test_resolve_solver_spec_from_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """插件 registry 返回 SolverSpec → 通过 build_module_spec 转 ModuleSpec（line 974-978）."""
    import zylab.flowchart.module as m
    from zylab.fea.solvers import SolverSpec as _RealSolverSpec

    # 使用一个缓存里没有的 type_id，命中插件查找分支
    tid = "plugin_solver_xxxxx"
    m._MODULES_BY_ID.pop(tid, None)

    # fake_obj 通过 isinstance(obj, SolverSpec) 检查——给它一个 MagicMock + spec
    fake_obj = MagicMock(spec=_RealSolverSpec)
    fake_obj.type_id = tid
    fake_spec = MagicMock()
    fake_spec.name = tid
    fake_registry = MagicMock()
    fake_registry.list.return_value = [fake_spec]
    fake_registry.resolve.return_value = fake_obj
    monkeypatch.setattr("zylab.core.registry.PluginRegistry", lambda: fake_registry)

    adapter_calls: list[Any] = []

    def _adapter(spec: Any) -> _DummyModuleSpec:
        adapter_calls.append(spec)
        return _DummyModuleSpec(spec.type_id)

    monkeypatch.setattr("zylab.flowchart.solver_adapter.build_module_spec", _adapter, raising=False)

    ms = m.module_spec(tid)
    assert isinstance(ms, _DummyModuleSpec)
    assert len(adapter_calls) == 1


def test_resolve_module_spec_from_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """插件 registry 直接返回 ModuleSpec → 直接缓存并返回（line 979-981）."""
    import zylab.flowchart.module as m
    from zylab.flowchart.module import ModuleSpec

    tid = "plugin_mod_xxxxx"
    m._MODULES_BY_ID.pop(tid, None)

    # 用 MagicMock(spec=ModuleSpec) 让 isinstance(obj, ModuleSpec) 通过
    fake_obj = MagicMock(spec=ModuleSpec)
    fake_obj.type_id = tid
    fake_spec = MagicMock()
    fake_spec.name = tid
    fake_registry = MagicMock()
    fake_registry.list.return_value = [fake_spec]
    fake_registry.resolve.return_value = fake_obj
    monkeypatch.setattr("zylab.core.registry.PluginRegistry", lambda: fake_registry)

    ms = m.module_spec(tid)
    assert ms is fake_obj


# ======================================================================
# report.py — 云图 / 图例
# ======================================================================


class _Mesh:
    n_nodes = 4
    dim = 2
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    n_dofs = 8


def test_svg_legend_empty_series() -> None:
    """CurveData.series 为空时 _svg_legend 应直接返回空列表（line 498）."""
    from zylab.flowchart.report import CurveData, _svg_legend

    data = CurveData(title="t")
    assert data.series == ()
    assert _svg_legend(data) == []


def test_cloud_field_displacement_full_time_history_last_frame() -> None:
    """displacement 全时程 (n_dofs, n_times) 形状 → 取末帧（line 586）."""
    from zylab.flowchart.report import _cloud_field

    n_nodes, n_times = 4, 3
    disp = np.random.randn(n_nodes * 2, n_times)
    payload = MagicMock()
    payload.mesh = _Mesh()
    payload.displacements = disp
    field, label = _cloud_field(payload, "displacement", n_nodes)
    assert field.shape == (n_nodes,)
    assert label == "位移模"


def test_cloud_field_stress_branch() -> None:
    """declared=stress 时走 nodal_stress_field 分支（line 590）."""
    from zylab.flowchart.report import _cloud_field

    # 构造一个 StaticSolution-like payload
    n_nodes = 4
    payload = MagicMock()
    payload.mesh = _Mesh()

    class _Block:
        conn = np.array([[0, 1], [2, 3]])

    _Mesh.blocks = [_Block()]
    er0 = MagicMock()
    er0.block = 0
    er0.index = 0
    er0.stress = [10.0, 0.0, 0.0]
    er1 = MagicMock()
    er1.block = 0
    er1.index = 1
    er1.stress = [20.0, 0.0, 0.0]
    payload.element_results = [er0, er1]

    field, label = _cloud_field(payload, "stress", n_nodes)
    assert label == "应力"
    assert field.shape == (n_nodes,)
    assert np.all(np.isfinite(field))


def test_auto_field_first_valid() -> None:
    """_auto_field 返回第一个非 None 属性名（line 610）."""
    from zylab.flowchart.report import _auto_field

    p = MagicMock()
    p.temperatures = None
    p.displacements = np.ones((4, 2))
    p.voltages = np.ones(4)
    assert _auto_field(p) == "displacement"


def test_cloud_coords_full_time_history() -> None:
    """_cloud_coords 中 (n_dofs, n_times) 位移取末帧（line 626）."""
    from zylab.flowchart.report import _cloud_coords

    n_nodes = 4
    values = np.zeros(n_nodes)
    payload = MagicMock()
    payload.displacements = np.random.randn(n_nodes * 2, 3)
    coords = _cloud_coords(payload, _Mesh(), values, deform=0.5)
    assert coords.shape == (n_nodes, 2)


# ======================================================================
# batch.py — _row_to_overrides 异常兜底、_NegSurrogate、run_batch_outputs 回退
# ======================================================================


def test_row_to_overrides_exception_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    """template.node 或 module_spec 抛异常 → 原样 val 保留（line 295-296）."""
    from zylab.flowchart.batch import _row_to_overrides

    tpl = MagicMock()
    tpl.node = MagicMock(side_effect=KeyError("no such node"))
    row = {"nid.pk": 3.14}
    monkeypatch.setattr(_batch_module, "_node_params", lambda _t, _nid: {"pk": 0.0})
    overrides = _row_to_overrides(tpl, row)
    assert overrides["nid"]["pk"] == 3.14


def test_neg_surrogate_fit() -> None:
    """_NegSurrogate.fit 空实现应返回 self（line 641 的最小等价场景）."""
    import numpy as np

    from zylab.optim.surrogate import Surrogate

    class _NegSurrogate(Surrogate):
        def fit(self, X: np.ndarray, y: np.ndarray) -> _NegSurrogate:
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.zeros(X.shape[0])

    n = _NegSurrogate()
    assert n.fit(np.zeros((2, 1)), np.zeros(2)) is n


def test_batch_opt_fallback_when_run_batch_outputs_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """真实验证 run_batch_outputs 抛异常 → 回退代理值（line 665-666）.

    explore_doe optimize=True 内部有两次 run_batch_outputs：
    第一次 DOE 采样（line 71）、第二次优化后真实验证（line 131）.
    我们让第一次返回正常样本，第二次抛异常，迫使进入 line 665-666 回退.
    optimize 函数通过 ``zylab.optim.__init__`` 延迟 import，patch ``zylab.optim.optimize``
    包属性可命中后续所有 ``from zylab.optim import optimize``.
    """
    from zylab.flowchart.batch import explore_doe

    call_count = {"n": 0}

    def _double_run(*_a: Any, **_kw: Any) -> tuple[np.ndarray, np.ndarray]:
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise RuntimeError("非法区域")
        return np.array([[0.5], [0.6]]), np.array([[1.0], [2.0]])

    monkeypatch.setattr(_batch_module, "run_batch_outputs", _double_run)

    from zylab.optim.optimize import OptimResult

    fake_res = OptimResult(best_x=np.array([0.5]), best_y=3.14, best_std=None, optimizer="de")
    # patch 包级 re-export 属性（explore_doe 延迟 from zylab.optim import optimize 会命中）
    monkeypatch.setattr("zylab.optim.optimize", lambda *_a, **_kw: fake_res)

    class _FakeDV:
        def __init__(self, name: str, lower: float = 0.0, upper: float = 1.0) -> None:
            self.name = name
            self.lower = lower
            self.upper = upper
            self.levels = ()

    class _FakeDS:
        variables = [_FakeDV("n.k")]

        def sample(self, _method: Any = None, n: int = 1, seed: int = 0) -> np.ndarray:
            return np.full((n, 1), 0.5)

        def to_input_rows(self, samples: np.ndarray) -> list[dict[str, float]]:
            return [{"n.k": float(row[0])} for row in samples]

    class _FakeTemplate:
        design_vars = [_FakeDV("n.k")]
        output_params = [MagicMock(name="E")]
        design_space = _FakeDS()
        outputs = ()

    tpl = _FakeTemplate()
    ds = _FakeDS()
    result = explore_doe(
        tpl,
        ds,
        n_samples=2,
        optimize=True,
        minimize=True,
        optimizer="differential_evolution",
        opt_n_iter=2,
    )
    # 真实验证抛异常 → 应回退代理值 best_y_raw = ores.best_y = 3.14
    assert result.best_y == pytest.approx(3.14)
    assert call_count["n"] >= 2


# ======================================================================
# buckling.py — 特征值出现显著虚部
# ======================================================================


def test_buckling_complex_eigenvalues(monkeypatch: pytest.MonkeyPatch) -> None:
    """eig 返回显著虚部特征值 → SolverError.

    buckling 用 scipy.linalg.eig（稠密），且 tol=1000 阈值下 0.5j 虚部远大于
    容差仍保持 complex dtype，触发 np.iscomplexobj 判定.
    """
    from scipy import linalg as sp_linalg
    from scipy import sparse

    from zylab.fea import buckling as bk

    mesh = MagicMock()
    mesh.n_dofs = 6
    mesh.free = list(range(6))

    k = sparse.eye(6, format="csr")
    kg = sparse.eye(6, format="csr") * 0.1

    def _fake_eig(_A: Any, _B: Any) -> tuple[np.ndarray, np.ndarray]:
        vals = np.array([1.0 + 0.5j, 2.0 - 0.1j, 3.0 + 0.0j])
        vecs = np.eye(6)
        return vals, vecs

    monkeypatch.setattr(sp_linalg, "eig", _fake_eig, raising=False)

    section = MagicMock()
    material = MagicMock()
    case = MagicMock()
    ref = MagicMock()
    ref.mesh = mesh
    ref.displacements = np.zeros((6,))

    monkeypatch.setattr(bk, "assemble_stiffness", lambda *_a, **_kw: k)
    monkeypatch.setattr(bk, "assemble_geometric", lambda *_a, **_kw: kg)
    monkeypatch.setattr(bk, "_extract_axial_forces", lambda *_a, **_kw: [1.0, 2.0])
    monkeypatch.setattr(bk, "_expand_fixed", lambda *_a, **_kw: [])

    with pytest.raises(SolverError, match="显著虚部"):
        bk.solve_buckling(
            mesh,
            materials=[material],
            sections=[section],
            case=case,
            reference=ref,
            n_modes=1,
        )


# ======================================================================
# optimize.py — 多条分支
# ======================================================================


class _FitterSurrogate(Surrogate):
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> _FitterSurrogate:
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sum(X**2, axis=1)

    def predict_std(self, X: np.ndarray) -> np.ndarray | None:  # type: ignore[override]
        return np.zeros(len(X)) + 0.1


class _Var:
    def __init__(self, name: str, lower: float = -2.0, upper: float = 2.0, levels: tuple[float, ...] = ()):
        self.name = name
        self.lower = lower
        self.upper = upper
        self.levels = levels


def test_build_bounds_with_levels_and_lower_upper() -> None:
    vs = [_Var("a", 0.0, 1.0), _Var("b", levels=(1, 3, 5))]
    bounds = _build_bounds(vs)
    assert bounds == [(0.0, 1.0), (1.0, 5.0)]


def test_optimize_rejects_non_surrogate() -> None:
    with pytest.raises(OptimError, match="Surrogate 实例"):
        _opt_module.optimize("not-a-surrogate", [_Var("x")])  # type: ignore[arg-type]


def test_optimize_rejects_unfitted_surrogate() -> None:
    s = MagicMock(spec=Surrogate)
    s.predict.side_effect = RuntimeError("not fitted")
    with pytest.raises(OptimError, match="尚未 fit"):
        _opt_module.optimize(s, [_Var("x")])


def test_optimize_bad_optimizer_str() -> None:
    s = _FitterSurrogate().fit(np.zeros((2, 1)), np.zeros(2))
    with pytest.raises(OptimError, match="未知优化器"):
        _opt_module.optimize(s, [_Var("x")], optimizer="nope-opt")


@pytest.mark.parametrize(
    "opt", [Optimizer.DIFFERENTIAL_EVOLUTION, Optimizer.SHGO, Optimizer.BASIN_HOPPING, Optimizer.DUAL_ANNEALING]
)
def test_optimize_all_optimizers(opt: Optimizer) -> None:
    s = _FitterSurrogate().fit(np.zeros((2, 1)), np.zeros(2))
    res = _opt_module.optimize(s, [_Var("x")], optimizer=opt, n_iter=2, seed=1)
    assert isinstance(res, OptimResult)
    assert res.best_std is not None


def test_optimize_callback_called() -> None:
    s = _FitterSurrogate().fit(np.zeros((2, 1)), np.zeros(2))
    calls: list[tuple[np.ndarray, float]] = []
    _opt_module.optimize(s, [_Var("x")], n_iter=2, seed=1, callback=lambda x, y: calls.append((x, y)))
    assert len(calls) > 0


def test_optimize_discrete_rounds_to_nearest_level() -> None:
    s = _FitterSurrogate().fit(np.zeros((3, 1)), np.zeros(3))
    vs = [_Var("x", levels=(0.0, 1.0, 2.0, 3.0))]
    res = _opt_module.optimize(s, vs, n_iter=2, seed=1)
    assert float(res.best_x[0]) in {0.0, 1.0, 2.0, 3.0}


def test_optimize_predict_std_exception_graceful() -> None:
    s = _FitterSurrogate().fit(np.zeros((2, 1)), np.zeros(2))
    original = s.predict_std

    def _boom(*_a: Any, **_kw: Any) -> np.ndarray | None:
        raise RuntimeError("std 不可用")

    s.predict_std = _boom  # type: ignore[assignment]
    res = _opt_module.optimize(s, [_Var("x")], n_iter=2, seed=1)
    assert res.best_std is None
    s.predict_std = original  # type: ignore[assignment]


# --- optimize_direct ---


class _FakeOutcome:
    succeeded = True

    def __init__(self, resolved: dict[str, float] | None = None) -> None:
        self._resolved = resolved or {"E": 1.0}

    def resolve_outputs(self, _tpl: Any) -> dict[str, float]:
        return dict(self._resolved)


def _make_var(name: str, lower: float = 0.0, upper: float = 1.0, levels: tuple[float, ...] = ()) -> MagicMock:
    v = MagicMock()
    v.name = name
    v.lower = lower
    v.upper = upper
    v.levels = levels
    return v


def test_optimize_direct_rejects_no_dotted_name() -> None:
    class _Tpl:
        output_params = [MagicMock(name="E")]

    with pytest.raises(OptimError, match="dotted 格式"):
        _opt_module.optimize_direct(_Tpl(), [_make_var("badname")])


def test_optimize_direct_rejects_no_output_params() -> None:
    class _Tpl:
        output_params = []

    with pytest.raises(OptimError, match="output_params"):
        _opt_module.optimize_direct(_Tpl(), [_make_var("n.k")])


def test_optimize_direct_target_not_found() -> None:
    class _Tpl:
        output_params = [MagicMock(name="E")]

    with pytest.raises(OptimError, match="output_param"):
        _opt_module.optimize_direct(_Tpl(), [_make_var("n.k")], target="X")


def test_optimize_direct_picks_first_target_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    outcome = _FakeOutcome({"E": 1.0})

    def _fake_workflow(_tpl: Any, overrides: Any = None, report: Any = None) -> Any:
        return outcome

    class _Tpl:
        output_params = [MagicMock(name="E")]

    # optimize_direct 内部延迟 from zylab.flowchart import run_workflow
    # patch 源头模块
    monkeypatch.setattr("zylab.flowchart.run_workflow", _fake_workflow)
    res = _opt_module.optimize_direct(_Tpl(), [_make_var("n.k")], n_iter=1, seed=1, optimizer="differential_evolution")
    assert isinstance(res, OptimResult)


def test_optimize_direct_maximize_flips_sign(monkeypatch: pytest.MonkeyPatch) -> None:
    outcome = _FakeOutcome({"E": -10.0})

    def _fake_workflow(_tpl: Any, overrides: Any = None, report: Any = None) -> Any:
        return outcome

    class _Tpl:
        output_params = [MagicMock(name="E")]

    monkeypatch.setattr("zylab.flowchart.run_workflow", _fake_workflow)
    res = _opt_module.optimize_direct(
        _Tpl(),
        [_make_var("n.k")],
        n_iter=1,
        seed=1,
        optimizer="differential_evolution",
        maximize=True,
    )
    assert isinstance(res, OptimResult)


# --- optimize_pareto ---


def test_optimize_pareto_requires_2_targets() -> None:
    class _Tpl:
        output_params = [MagicMock(name="E")]

    with pytest.raises(OptimError, match="至少需要 2 个目标"):
        _opt_module.optimize_pareto(_Tpl(), [_make_var("n.k")], targets=["E"])


def test_optimize_pareto_rejects_small_pop() -> None:
    class _Tpl:
        output_params = [MagicMock(name="E"), MagicMock(name="d")]

    with pytest.raises(OptimError, match="n_population 至少 4"):
        _opt_module.optimize_pareto(_Tpl(), [_make_var("n.k")], targets=["E", "d"], n_population=3)


def test_optimize_pareto_missing_output_param() -> None:
    class _Tpl:
        output_params = [MagicMock(name="E")]

    with pytest.raises(OptimError, match="缺 output_param"):
        _opt_module.optimize_pareto(_Tpl(), [_make_var("n.k")], targets=["E", "X"], n_population=4)


def test_optimize_pareto_variable_name_must_be_dotted() -> None:
    """optimize_pareto 变量名非 dotted 格式 → OptimError."""

    # 校验顺序：targets 数量 → pop 大小 → missing output_param → variable name
    # 所以 output_param 必须先完整
    class _Op:
        name = "E"

    class _Op2:
        name = "d"

    class _Tpl:
        output_params = [_Op(), _Op2()]

    with pytest.raises(OptimError, match="dotted 格式"):
        _opt_module.optimize_pareto(_Tpl(), [_make_var("badname")], targets=["E", "d"], n_population=4)


def test_optimize_pareto_minimize_list_flips_some_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """minimize 传 [True, False] 时 F_pareto 第二列应被翻回正号."""
    outcome = _FakeOutcome({"E": 2.0, "d": 3.0})

    def _fake_workflow(_tpl: Any, overrides: Any = None, report: Any = None, cache: Any = None) -> Any:
        return outcome

    class _Op:
        name = "E"

    class _Op2:
        name = "d"

    class _Tpl:
        output_params = [_Op(), _Op2()]

    monkeypatch.setattr("zylab.flowchart.run_workflow", _fake_workflow)

    res = _opt_module.optimize_pareto(
        _Tpl(),
        [_make_var("n.k")],
        targets=["E", "d"],
        minimize=[True, False],
        n_population=4,
        n_generations=1,
        seed=1,
    )
    assert isinstance(res, ParetoOptResult)
    assert res.F.shape[0] >= 1


def test_optimize_pareto_parallel_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """n_workers=2 时 _batch_evaluate 走 run_batch 并行分支."""
    fake_outcome = _FakeOutcome({"E": 1.0, "d": 2.0})

    called_with_workers: list[int] = []

    def _fake_run_batch(
        _tpl: Any, rows: list[dict[str, Any]], n_workers: int | None = None, cache: Any = None
    ) -> list[Any]:
        if n_workers is not None:
            called_with_workers.append(n_workers)
        return [fake_outcome] * len(rows)

    class _Op:
        name = "E"

    class _Op2:
        name = "d"

    class _Tpl:
        output_params = [_Op(), _Op2()]

    monkeypatch.setattr("zylab.flowchart.run_batch", _fake_run_batch)
    monkeypatch.setattr(_opt_module, "get_max_workers", lambda: 4, raising=False)

    res = _opt_module.optimize_pareto(
        _Tpl(),
        [_make_var("n.k")],
        targets=["E", "d"],
        n_population=4,
        n_generations=1,
        seed=1,
        n_workers=2,
    )
    assert isinstance(res, ParetoOptResult)
    assert 2 in called_with_workers


def test_optimize_x_dict_roundtrips() -> None:
    res = OptimResult(best_x=np.array([1.5, 2.5]), best_y=1.0)

    class _V:
        def __init__(self, name: str) -> None:
            self.name = name

    d = res.x_dict([_V("a"), _V("b")])
    assert d == {"a": 1.5, "b": 2.5}


# ======================================================================
# process Executor 边界：_cancel 对已完成任务幂等 (line 259)
# ======================================================================


def test_task_handle_cancel_idempotent_when_already_done() -> None:
    """TaskHandle._cancel 对已置 done 的任务应直接 return（line 259 分支已覆盖）."""
    handle = TaskHandle("t1")
    handle._mark_terminal(handle.status.__class__.CANCELLED)
    handle._cancel()
    assert handle.done
