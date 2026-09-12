"""批处理执行：参数化计算进程内拓扑序求解 + 参数覆盖/扫描 + 结果摘要.

与 :class:`~zylab.flowchart.runner.WorkflowRunner`（GUI 进程隔离编排）互补：
批处理场景（CLI、参数扫描）无交互，进程内直调节点函数省去子进程 pickle
往返，扫描多组参数时收益显著。失败策略 = 首个失败节点中止（下游依赖其
输出，不可继续），后续节点保持未执行。

批量探索层（Phase 4/5）入口：

- :func:`run_scan`：单参数逐值扫描（旧路径，简洁）；
- :func:`run_batch`：多行扁平参数批量运行——直接消费
  :meth:`~zylab.doe.design_space.DesignSpace.to_input_rows` 的输出，
  每行 ``{"node_id.param_key": value}`` 自动展开为节点分组 override。
"""

from __future__ import annotations

import importlib
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from typing_extensions import override

from zylab.fea import (
    BucklingSolution,
    ElectroThermalSolution,
    HarmonicResponse,
    ModalSolution,
    NonlinearSolution,
    StaticSolution,
    TransientSolution,
)

from .bundle import ConductionBundle, ModelBundle
from .errors import FlowchartError
from .graph import WorkflowGraph
from .param_store import ParameterStore
from .results import resolve_input
from .template import Template

# pytest-xdist worker 内检测到则禁用内部 ProcessPoolExecutor，避免 Windows
# spawn 模式下嵌套进程池极低频死锁。正常用户调用不受影响。
_IS_XDIST_WORKER = bool(os.environ.get("PYTEST_XDIST_WORKER"))

__all__ = [
    "NodeOutcome",
    "ReportFn",
    "RunOutcome",
    "resolve_target",
    "run_batch",
    "run_batch_outputs",
    "run_scan",
    "run_workflow",
    "summarize",
]

logger = logging.getLogger(__name__)

#: 进度回调签名（与节点协议 / ProcessExecutor 注入约定一致）
ReportFn = Callable[[float, str], None]


def resolve_target(target: str) -> Callable[..., Any]:
    """解析 ``"模块:函数"`` 目标字符串为可调用对象."""
    module_name, _, attr = target.partition(":")
    return getattr(importlib.import_module(module_name), attr)


@dataclass(frozen=True)
class NodeOutcome:
    """单节点执行结果.

    :param node_id: 节点 id。
    :param name: 模块显示名。
    :param result: 节点输出对象（失败/未执行为 None）。
    :param error: 错误消息（空串表示成功）。
    :param elapsed: 执行耗时（秒）。
    """

    node_id: str
    name: str
    result: Any = None
    error: str = ""
    elapsed: float = 0.0

    @property
    def ok(self) -> bool:
        """是否执行成功."""
        return not self.error


@dataclass(frozen=True)
class RunOutcome:
    """单次工作流运行结果（拓扑序全部节点）.

    :param outcomes: 节点结果表（参数化计算定义序）。
    """

    outcomes: tuple[NodeOutcome, ...]

    @property
    def succeeded(self) -> bool:
        """是否全部节点成功执行（失败中止后的未执行节点不计入）."""
        return all(o.ok and o.result is not None for o in self.outcomes)

    def outcome(self, node_id: str) -> NodeOutcome:
        """按节点 id 取执行结果；不存在抛 :class:`KeyError`."""
        for o in self.outcomes:
            if o.node_id == node_id:
                return o
        raise KeyError(f"运行结果中无节点 {node_id!r}")

    def first_error(self) -> str:
        """首个失败节点的错误消息（全部成功返回空串）."""
        for o in self.outcomes:
            if not o.ok:
                return f"{o.node_id} ({o.name}): {o.error}"
        return ""

    # ------------------------------------------------------------------ Phase 3 参数中心化集成

    def resolve_outputs(self, template: Template) -> dict[str, Any]:
        """基于参数化计算的 OutputParam 声明解析运行结果，返回输出参数名 -> 值.

        :param template: 来源参数化计算（须声明 output_params）。
        :return: 全部成功解析的输出参数值表。
        :raises FlowchartError: 运行有失败节点 / 节点未声明 output_params /
            source 解析失败 / expr 求值失败。
        """
        if not self.succeeded:
            raise FlowchartError(f"工作流存在失败节点，无法解析输出参数: {self.first_error()}")
        store = ParameterStore.from_template(template)
        if not store.outputs:
            return {}
        outputs = {o.node_id: o.result for o in self.outcomes if o.ok and o.result is not None}
        return store.resolve_all(outputs)


def run_workflow(
    template: Template,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    report: ReportFn | None = None,
    cache: dict[str, Any] | None = None,
) -> RunOutcome:
    """进程内按拓扑序执行参数化计算全部节点（失败即中止）.

    :param template: 参数化计算。
    :param overrides: 节点参数覆盖表（节点 id -> 参数表，整体替换该节点 params）。
    :param report: 进度回调（透传给节点函数，``(progress, message)``）。
    :param cache: 可选的 ``{content_hash: result}`` 字典——提供后会计算每个
        节点的内容指纹（参数 + 上游依赖哈希链），命中则跳过真实求解、复用
        上一次的结果。适合同一 template 多次运行（batch / DOE / 敏感性）
        时跨调用共享节点级缓存。

        用法::

            cache = {}
            for row in rows:
                outcome = run_workflow(template, overrides=..., cache=cache)
            # cache 现在包含所有节点的 (hash, result) 对，可继续复用

    注意：缓存是**节点级**的——上游节点哈希变化会级联触发下游节点
    重算（指纹公式天然包含上游哈希），不存在脏结果泄漏。
    """
    merged = template.with_params(dict(overrides)) if overrides else template
    graph = WorkflowGraph(merged)
    results: dict[str, Any] = {}
    outcomes: list[NodeOutcome] = []

    failed = False
    for node in graph.nodes():
        if failed:
            outcomes.append(NodeOutcome(node_id=node.id, name=node.name))
            continue
        inputs = {port: resolve_input(ref, results) for port, ref in node.inputs.items()}

        # --- 缓存检查 ---
        content_hash = graph.compute_node_hash(node.id)
        hit = False
        cached_result: Any = None
        if cache is not None and content_hash in cache:
            cached_result = cache[content_hash]
            hit = True

        fn = resolve_target(node.spec.target)
        started = time.perf_counter()
        try:
            if hit:
                results[node.id] = cached_result
                elapsed = time.perf_counter() - started
                outcomes.append(
                    NodeOutcome(
                        node_id=node.id,
                        name=node.name,
                        result=cached_result,
                        elapsed=elapsed,
                    ),
                )
                logger.debug("批处理节点命中缓存: %s (%.3fs)", node.id, elapsed)
            else:
                results[node.id] = fn(inputs, dict(node.params), report)
                elapsed = time.perf_counter() - started
                outcomes.append(
                    NodeOutcome(
                        node_id=node.id,
                        name=node.name,
                        result=results[node.id],
                        elapsed=elapsed,
                    ),
                )
                logger.debug("批处理节点完成: %s (%.3fs)", node.id, elapsed)
                if cache is not None:
                    cache[content_hash] = results[node.id]
            graph.mark_result(node.id, results[node.id], elapsed, content_hash=content_hash)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            message = f"{type(exc).__name__}: {exc}"
            outcomes.append(
                NodeOutcome(
                    node_id=node.id,
                    name=node.name,
                    error=message,
                    elapsed=elapsed,
                ),
            )
            logger.warning("批处理节点失败: %s: %s", node.id, message)
            failed = True
    return RunOutcome(tuple(outcomes))


def run_scan(
    template: Template,
    param_ref: str,
    values: tuple[float, ...],
    report: ReportFn | None = None,
) -> tuple[RunOutcome, ...]:
    """参数化扫描：对 ``"node.param"`` 参数逐值运行整个工作流.

    :param param_ref: 参数引用（``"节点id.参数键"``）。
    :param values: 扫描取值序列（原样传入，不做插值）。
    :param report: 进度回调（透传给每次运行）。
    """
    node_id, _, key = param_ref.partition(".")
    if not node_id or not key:
        raise ValueError(f"参数引用 {param_ref!r} 应为 '节点id.参数键' 格式")
    runs = []
    for value in values:
        overrides = {node_id: {**_node_params(template, node_id), key: value}}
        runs.append(run_workflow(template, overrides, report))
    return tuple(runs)


def _node_params(template: Template, node_id: str) -> dict[str, Any]:
    """取参数化计算节点原始参数表（不存在抛 :class:`ValueError`）."""
    try:
        return dict(template.node(node_id).params)
    except FlowchartError as exc:
        raise ValueError(f"参数化计算 {template.id!r} 无节点 {node_id!r}: {exc}") from exc


def _row_to_overrides(template: Template, row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """将 ``{'node.param': value, ...}`` 扁平行展开为 ``{node: {param: value}}`` override.

    每个节点的基础参数来自 ``template.node(node_id).params``，
    扁平行中指定的键覆盖同名参数。未知节点 id 或未知扁平格式
    （不含 ``.``）会静默跳过——允许用户只覆盖部分节点。

    INT 类型参数若收到 float 值（DOE 采样天然返回浮点），
    自动 round 为整数，避免下游 ``coerce`` 拒绝。
    """
    from .module import ParamType, module_spec  # 延迟 import 避免循环

    overrides: dict[str, dict[str, Any]] = {}
    for flat_key, raw_val in row.items():
        if "." not in flat_key:
            continue
        node_id, _, param_key = flat_key.partition(".")
        if node_id not in overrides:
            try:
                overrides[node_id] = _node_params(template, node_id)
            except ValueError:
                continue
        # INT param 自动 round float 值（DOE 采样天然返回浮点）
        val: Any = raw_val
        if isinstance(val, float):
            try:
                tn = template.node(node_id)
                spec = module_spec(tn.type_id)
                pspec = next((p for p in spec.params if p.key == param_key), None)
                if pspec is not None and pspec.param_type is ParamType.INT:
                    val = round(val)
            except Exception:
                pass  # 查不到 spec 就原样用，后续 coerce 会报错
        overrides[node_id][param_key] = val
    return overrides


def _batch_row_worker(args: tuple[Template, Mapping[str, Any], bool]) -> RunOutcome:
    """进程池 worker——对单行参数完整跑一次 run_workflow.

    模块级函数（Windows spawn 模式可 pickle）。每个 worker 进程内部自建
    空节点级 cache——节点内部的哈希缓存仍有效，但**跨进程不共享**
    （进程间隔离）。跨进程共享的 cache 应在主进程里做"去重采样"预处理.

    :param args: ``(template, row_dict, use_cache)`` 三元组.
    :return: 单次 run_workflow 的 RunOutcome.
    """
    template, row, use_cache = args
    overrides = _row_to_overrides(template, row)
    local_cache: dict[str, Any] | None = {} if use_cache else None
    return run_workflow(template, overrides, report=None, cache=local_cache)


def run_batch(  # noqa: PLR0913
    template: Template,
    param_rows: Sequence[Mapping[str, Any]],
    *,
    report: ReportFn | None = None,
    use_cache: bool = True,
    cache: dict[str, Any] | None = None,
    n_workers: int | None = None,
) -> list[RunOutcome]:
    """批量参数化运行（Phase 4 入口）.

    对 ``DesignSpace.to_input_rows`` 的每一行扁平参数，展开为节点分组
    override 后进程内拓扑序执行整个参数化计算。节点间无数据依赖——
    每组参数独立运行，不会修改其它组的运行状态。

    :param template: 参数化计算模板。
    :param param_rows: 每行 ``{"node_id.param_key": value}`` 的参数表。
        典型来源：:meth:`~zylab.doe.design_space.DesignSpace.to_input_rows`
        或用户手写的 What-if 表格。
    :param report: 总进度回调 ``(fraction, message)``，每次运行结束后调用。
        **并行模式下不触发**（进程池 map 无法注入回调）。
    :param use_cache: 是否开启节点级哈希缓存（默认 True）。开启后
        内部维护一个 ``{content_hash: result}`` 共享字典。
    :param cache: 外部传入的缓存字典——提供后 ``use_cache`` 自动置 True，
        本次 batch 结果会回填到外部 dict，后续 batch / Sobol / optimize
        可继续复用。典型场景是 :func:`explore_doe` 里 DOE 采样 + Sobol
        子采样 + optimize 三次独立调用，共享同一个 cache dict 避免
        重复 FE 求解。**并行模式下此参数失效**——每个 worker 进程自建
        空缓存（进程间不能共享 dict），跨进程去重需用户先预处理
        param_rows 去掉重复配置。
    :param n_workers: 进程池大小。``None`` 或 ``<=1`` 走当前进程
        串行（默认，向后兼容）；``>=2`` 时开 :class:`ProcessPoolExecutor`
        多进程并行。Windows / Linux 均支持（模板 dataclass 可 pickle，
        worker 模块级定义）。**并行收益在单次 FE > 10ms 时显著**
        （例如非线性、瞬态、热耦合问题），快 FE（<5ms）进程池启动开销
        可能抵消收益。
    :return: 与 ``param_rows`` 等长的结果列表，结果顺序与输入行对齐.
    :raises ValueError: ``param_rows`` 为空.

    使用示例::

        # 同进程内多次 batch 共享 cache：
        cache = {}
        run_batch(template, rows_a, cache=cache)
        run_batch(template, rows_b, cache=cache)  # row_b 里与 row_a 相同
                                                   # 的节点配置自动命中缓存

        # 关掉缓存（调试 / 内存敏感场景）：
        run_batch(template, rows, use_cache=False)

        # 4 进程并行跑 1000 次非线性 FE：
        run_batch(template, rows, n_workers=4)
    """
    if not param_rows:
        raise ValueError("run_batch: param_rows 不能为空")

    n = len(param_rows)

    # ---- 并行分支 ----
    if n_workers is not None and n_workers >= 2 and not _IS_XDIST_WORKER:
        # 每个 worker 内部自建节点级 cache（进程间隔离）
        # 不回填主进程的 cache dict——跨进程共享不可行
        args_iter = ((template, row, use_cache) for row in param_rows)
        actual_workers = min(n_workers, max(1, os.cpu_count() or 1))
        with ProcessPoolExecutor(max_workers=actual_workers) as pool:
            # map 保序——保证 outcomes 与 param_rows 对齐
            outcomes = list(pool.map(_batch_row_worker, args_iter))
        # 并行时 cache 共享失效，直接返回
        return outcomes

    # ---- 串行分支 ----
    effective_cache: dict[str, Any] | None
    if cache is not None:
        effective_cache = cache
    elif use_cache:
        effective_cache = {}
    else:
        effective_cache = None
    results: list[RunOutcome] = []
    for i, row in enumerate(param_rows):
        overrides = _row_to_overrides(template, row)
        outcome = run_workflow(template, overrides, report, cache=effective_cache)
        results.append(outcome)
        if report is not None:
            report((i + 1) / n, f"批量运行 {i + 1}/{n}")
    return results


def run_batch_outputs(
    template: Template,
    param_rows: Sequence[Mapping[str, Any]],
    *,
    use_cache: bool = True,
    cache: dict[str, Any] | None = None,
    n_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """批量运行 + 输出参数解析（Phase 5 探索层便捷入口）.

    内部调用 :func:`run_batch`，成功运行的每组结果通过
    :meth:`RunOutcome.resolve_outputs` 把 template 的 ``output_params``
    声明解析成 ``y`` 向量。

    :param use_cache: 透传给 :func:`run_batch` 的缓存开关。
    :param cache: 透传给 :func:`run_batch` 的外部缓存字典。
    :param n_workers: 透传给 :func:`run_batch` 的进程池大小.

    :return: ``(X, y)`` 形状均为 ``(n_success, n_vars_or_outputs)``，
        其中 ``X`` 直接来自 ``param_rows`` 按顺序堆叠、``y`` 来自
        output_params 解析。失败运行自动跳过，两端长度始终一致。
    :raises FlowchartError: template 未声明 ``output_params`` /
        所有运行均失败 / 部分运行 output_params 解析失败。

    使用示例::

        X, y = run_batch_outputs(template, rows)  # 默认自动开缓存
        surrogate.fit(X, y)
        result = optimize(surrogate, ds.variables)
    """
    outcomes = run_batch(template, param_rows, use_cache=use_cache, cache=cache, n_workers=n_workers)
    succeeded_rows: list[np.ndarray] = []
    succeeded_ys: list[np.ndarray] = []
    for outcome, row in zip(outcomes, param_rows):
        if not outcome.succeeded:
            continue
        outputs = outcome.resolve_outputs(template)
        if not outputs:
            raise FlowchartError("run_batch_outputs: template 未声明 output_params")
        succeeded_rows.append(np.array([row[k] for k in sorted(row)]))
        succeeded_ys.append(np.array([outputs[name] for name in sorted(outputs)], dtype=float))
    if not succeeded_rows:
        raise FlowchartError("run_batch_outputs: 所有运行均失败，无可用 y 值")
    X = np.vstack(succeeded_rows)
    y = np.vstack(succeeded_ys) if succeeded_ys else np.empty((0, 0))
    return X, y


def summarize(outcome: RunOutcome) -> str:
    """生成运行结果摘要（每结果节点一行关键指标，中文）."""
    lines: list[str] = []
    for o in outcome.outcomes:
        if not o.ok:
            lines.append(f"[{o.node_id}] {o.name} 失败: {o.error}")
        elif o.result is None:
            lines.append(f"[{o.node_id}] {o.name} 未执行（上游失败）")
        else:
            lines.append(f"[{o.node_id}] {o.name} {_describe(o.result)}（{o.elapsed:.3f}s）")
    return "\n".join(lines)


def _describe(result: Any) -> str:  # noqa: PLR0911  各类解各一行指标，分支语义不可合并
    """按解类型生成单行指标描述."""
    if isinstance(result, (ModelBundle, ConductionBundle)):
        mesh = result.mesh
        return f"模型: {mesh.n_nodes} 节点 / {mesh.n_elements} 单元"
    if isinstance(result, StaticSolution):
        u = result.displacements
        tip = float(np.linalg.norm(u, axis=1).max()) if u.size else 0.0
        return f"静力: 最大位移模长 {tip:.6g}，应变能 {result.strain_energy:.6g}"
    if isinstance(result, ModalSolution):
        hz = result.frequencies_hz[:3]
        shown = " / ".join(f"{f:.4g}" for f in hz)
        return f"模态: 前 {hz.size} 阶频率 {shown} Hz（共 {result.n_modes} 阶）"
    if isinstance(result, HarmonicResponse):
        amp = np.abs(result.displacements).max() if result.displacements.size else 0.0
        return f"谐响应: 峰值位移幅值 {float(amp):.6g}（{result.n_frequencies} 频率点）"
    if isinstance(result, BucklingSolution):
        factor = float(result.load_factors.min()) if result.load_factors.size else float("nan")
        return f"屈曲: 最小临界载荷因子 {factor:.6g}（共 {result.n_modes} 阶）"
    if isinstance(result, NonlinearSolution):
        u = result.displacements
        tip = float(np.linalg.norm(u, axis=1).max()) if u.size else 0.0
        state = "收敛" if result.converged else "未收敛"
        return f"非线性: {state}，最大位移模长 {tip:.6g}，Newton 迭代 {result.total_iterations} 次"
    if isinstance(result, TransientSolution):
        u = result.displacements
        peak = float(np.abs(u).max()) if u.size else 0.0
        return f"瞬态: 峰值位移 {peak:.6g}（{result.n_steps} 步，dt={result.dt:.4g}s）"
    if isinstance(result, ElectroThermalSolution):
        return f"电热: 峰值温度 {result.t_max:.6g}，总电功率 {result.total_power:.6g} W"
    return f"完成: {type(result).__name__}"


# ---------------------------------------------------------------------------
# Phase 6: DOE 探索便捷入口（DOE → batch → surrogate → sensitivity）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExploreResult:
    """单函数 DOE 探索结果."""

    X: np.ndarray  # (N, D_in) 采样点
    Y: np.ndarray  # (N, D_out) 观测值
    surrogate: object | None = None  # RBF/GPR 响应面（若拟合）
    si: np.ndarray | None = None  # 一阶 Sobol 指数（若做敏感性分解）
    sti: np.ndarray | None = None  # 总效应 Sobol 指数
    variance: float | None = None  # 输出方差
    best_x: np.ndarray | None = None  # 优化最优解 float 向量
    best_y: float | None = None  # 最优解处目标函数值（原始尺度）
    best_x_int: np.ndarray | None = None  # 最优解 round 到 int（网格参数用）
    optimizer: str | None = None  # 用了哪个优化器


def explore_doe(  # noqa: PLR0913
    template: Any,
    ds: Any,
    *,
    n_samples: int | None = None,
    method: Any = None,
    seed: int = 0,
    fit_surrogate: bool = True,
    sensitivity: bool = False,
    sobol_N: int = 512,
    optimize: bool = False,
    optimizer: Any = None,
    opt_n_iter: int = 100,
    minimize: bool = True,
    cache: dict[str, Any] | None = None,
    n_workers: int | None = None,
) -> ExploreResult:
    """一行走完 DOE 批量探索.

    流程：
        1. DesignSpace.sample(method, N, seed) 生成拉丁超立方 / 全因子 / Sobol 采样;
        2. to_input_rows 转扁平化参数字典;
        3. run_batch_outputs(template, rows) 批量求解;
        4. （可选）RBF 响应面拟合;
        5. （可选）在响应面上做 Sobol 敏感性分解;
        6. （可选）在响应面上调用 scipy 全局优化器找最优设计点.

    Parameters
    ----------
    template:
        必须声明 `output_params` 的 Template（否则无法输出 Y）.
    ds:
        :class:~zylab.doe.design_space.DesignSpace；`ds.variables` 提供
        每个设计变量的 `.lower` / `.upper` 边界给优化器.
    n_samples:
        采样点数（覆盖 ds 默认）；method 为 FULL_FACTORIAL 时忽略.
    method:
        :class:~zylab.doe.design_space.SamplingMethod，缺省 LHC.
    seed:
        采样、Sobol 子采样和优化器的随机种子.
    fit_surrogate:
        是否拟合 RBF 响应面（default True）. optimize=True 时自动启用.
    sensitivity:
        是否在响应面上做 Sobol 敏感性分解（default False）.
    sobol_N:
        Sobol 估计器的 Saltelli 采样基数.
    optimize:
        是否在响应面上追加全局优化找最优点（default False）.
    optimizer:
        :class:~zylab.optim.surrogate.Optimizer 枚举或字符串
        （`"differential_evolution"` / `"shgo"` / `"basinhopping"` /
        `"dual_annealing"`）；缺省 DE.
    opt_n_iter:
        优化迭代次数 / 函数评估次数上限.
    minimize:
        True=最小化，False=最大化；响应面自动取负.

    Returns
    -------
    ExploreResult

    """
    # 延迟 import 避免循环依赖
    from zylab.doe.design_space import SamplingMethod

    if method is None:
        method = SamplingMethod.LATIN_HYPERCUBE

    samples = ds.sample(method, n_samples, seed=seed)
    rows = ds.to_input_rows(samples)
    _shared_cache: dict[str, Any] | None = cache if cache is not None else {}
    X, Y = run_batch_outputs(template, rows, cache=_shared_cache, n_workers=n_workers)
    result = ExploreResult(X=X, Y=Y)

    if not fit_surrogate and not sensitivity:
        return result

    from zylab.optim.surrogate import RbfSurrogate

    surf = RbfSurrogate()
    surf.fit(X.astype(float), Y.ravel())
    result = replace(result, surrogate=surf)

    if sensitivity:
        from zylab.optim.sensitivity import sobol_analysis

        # RBF.predict 接受 (n, d) → (n,)；包装一个接受 (d,) 或 (n, d) 的 func
        def func(x: np.ndarray) -> np.ndarray:
            arr = np.atleast_2d(x)
            return surf.predict(arr)

        sres = sobol_analysis(func, d=X.shape[1], N=sobol_N, seed=seed)
        result = replace(result, si=sres.Si, sti=sres.STi, variance=sres.variance)

    if optimize:
        from zylab.optim import Optimizer
        from zylab.optim import optimize as _optimize

        opt_method = Optimizer(optimizer) if optimizer is not None else Optimizer.DIFFERENTIAL_EVOLUTION

        # 优化器最小化 surf.predict；若最大化则包一层取负代理
        if minimize:
            target_surf: Surrogate = surf
        else:
            from zylab.optim.surrogate import Surrogate

            class _NegSurrogate(Surrogate):  # 取负代理，委托给 surf（最大化问题）
                @override
                def fit(self, X: np.ndarray, y: np.ndarray) -> _NegSurrogate:
                    return self

                @override
                def predict(self, X: np.ndarray) -> np.ndarray:
                    return -surf.predict(X)

            target_surf = _NegSurrogate()

        ores = _optimize(
            target_surf,
            ds.variables,
            optimizer=opt_method,
            n_iter=opt_n_iter,
            seed=seed,
        )
        best_y_raw = ores.best_y if minimize else -ores.best_y
        best_x_int = np.round(ores.best_x).astype(int)

        # 真实验证：surrogate-based 优化器可能钻 RBF 插值漏洞（如负应变能），
        # 在 best_x_int 处再跑一次真实求解，用真实值覆盖 best_y.
        rows_dict = [{dv.name: float(best_x_int[i]) for i, dv in enumerate(ds.variables)}]
        try:
            _Xv, Yv = run_batch_outputs(template, rows_dict, cache=_shared_cache)
            verified_y = float(Yv[0, 0])
        except Exception:  # 真实验证失败（如代理钻到非法区域） → 回退到代理值
            verified_y = float(best_y_raw)

        result = replace(
            result,
            best_x=ores.best_x.copy(),
            best_y=verified_y,
            best_x_int=best_x_int,
            optimizer=opt_method.value if hasattr(opt_method, "value") else str(opt_method),
        )

    return result
