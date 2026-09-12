"""优化层：代理优化、直接优化、NSGA-II 多目标 Pareto.

三条路径共用 :class:`Optimizer` 枚举和 :class:`OptimResult` 容器。
多目标场景返回 :class:`ParetoOptResult`——含整条 Pareto 前沿。

三种优化入口：

1. :func:`optimize` ——  DOE 采样 + 训练 :class:`Surrogate` + scipy 全局
   优化器在代理上找最优；
2. :func:`optimize_direct` —— 跳过代理，直接把 workflow 当目标函数
   跑 scipy 全局优化器；
3. :func:`optimize_pareto` —— 手写 NSGA-II，直接在真实函数上做多目标
   进化优化，返回一整条 Pareto 前沿（复用 :mod:`pareto` 模块的
   :func:`pareto_ranks` + :func:`crowding_distance` 做非支配排序和拥挤距离）。

三种都接受 :class:`~zylab.doe.design_space.DesignSpace.variables`
作为设计空间边界，:func:`optimize_direct` / :func:`optimize_pareto`
额外接受 ``run_workflow`` 级别的 ``cache=`` 外部缓存字典。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from scipy.optimize import basinhopping, differential_evolution, dual_annealing, shgo

from .errors import OptimError
from .surrogate import Surrogate

if TYPE_CHECKING:
    from zylab.flowchart import Template

__all__ = [
    "OptimResult",
    "Optimizer",
    "ParetoOptResult",
    "optimize",
    "optimize_direct",
    "optimize_pareto",
]


class Optimizer(str, Enum):
    """scipy 全局优化器枚举."""

    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    SHGO = "shgo"  # 简化的 Lipschitz 全局优化
    BASIN_HOPPING = "basinhopping"  # 多起点 + 局部优化
    DUAL_ANNEALING = "dual_annealing"  # 双退火


_DEFAULT_SEED = 42


@dataclass
class OptimResult:
    """优化结果容器."""

    #: 最优解（实际值空间，shape (d,) 或单值）
    best_x: np.ndarray
    #: 代理预测的最优目标值
    best_y: float
    #: 最优解处代理预测的标准差（GPR 才有，RBF 为 None）
    best_std: Optional[float] = None
    #: 优化器名称
    optimizer: str = ""
    #: scipy 原始结果（调试用）
    raw_result: object = None

    def x_dict(self, variables: Sequence[object]) -> dict[str, float]:
        """将 ``best_x`` 按变量序展开为 ``{变量名: 值}``."""
        return {getattr(v, "name", str(i)): float(self.best_x[i]) for i, v in enumerate(variables)}


def _build_bounds(variables: Sequence[object]) -> list[tuple[float, float]]:
    """从 DesignVariable 序列构造 scipy 优化器 bounds.

    - 连续变量 → ``(lower, upper)`` 直接用；
    - 离散变量 → bounds 扩展到 ``(min(levels), max(levels))``，
      优化器找到实数最优后再由 DesignVariable.denormalize 取最近水平。
    """
    bounds: list[tuple[float, float]] = []
    for v in variables:
        levels = getattr(v, "levels", ())
        if levels:
            lo, hi = float(min(levels)), float(max(levels))
        else:
            lo, hi = float(getattr(v, "lower", 0.0)), float(getattr(v, "upper", 1.0))
        bounds.append((lo, hi))
    return bounds


def optimize(  # noqa: PLR0913  优化器签名必须暴露所有 scipy 全局优化器公共参数
    surrogate: Surrogate,
    variables: Sequence[object],
    *,
    optimizer: Optimizer | str = Optimizer.DIFFERENTIAL_EVOLUTION,
    n_iter: int = 100,
    seed: int = _DEFAULT_SEED,
    callback: Optional[Callable[[np.ndarray, float], None]] = None,
) -> OptimResult:
    """在给定设计空间边界上用指定优化器最小化代理.

    :param surrogate: 已 fit 的代理模型。
    :param variables: 设计变量序列（通常来自
        :attr:`~zylab.doe.design_space.DesignSpace.variables`）。
    :param optimizer: 优化器。
    :param n_iter: 最大迭代/评估次数——具体含义随优化器：
        DE = ``maxiter``，SHGO = ``n``（采样点），BH = ``niter``，
        DA = ``maxiter``。
    :param seed: 随机种子。
    :param callback: 每代回调 ``(x, f) → None``，用于记录轨迹。
    :raises OptimError: 代理未 fit / 优化器未知 / 变量边界非法。
    """
    if not isinstance(surrogate, Surrogate):
        raise OptimError(f"surrogate 须是 Surrogate 实例，得到 {type(surrogate)!r}")
    try:
        test = surrogate.predict(np.zeros((1, len(variables))))
        del test
    except Exception as exc:
        raise OptimError(f"代理模型尚未 fit: {exc}") from exc

    bounds = _build_bounds(variables)
    try:
        opt = Optimizer(optimizer)
    except ValueError as exc:
        raise OptimError(f"未知优化器 {optimizer!r}（可选 {[e.value for e in Optimizer]}）") from exc

    rng = np.random.default_rng(seed)

    def objective(x: np.ndarray) -> float:
        # 离散变量取最近水平（代理训练用的是实际值，预测也要喂实际值）
        x_fixed = np.asarray(x, dtype=float).copy()
        for i, v in enumerate(variables):
            levels = getattr(v, "levels", ())
            if levels:
                x_fixed[i] = float(levels[int(np.argmin(np.abs(np.array(levels) - x_fixed[i])))])
        y = float(surrogate.predict(x_fixed.reshape(1, -1))[0])
        if callback is not None:
            callback(x_fixed.copy(), y)
        return y

    if opt == Optimizer.DIFFERENTIAL_EVOLUTION:
        res = differential_evolution(objective, bounds=bounds, maxiter=n_iter, seed=seed, polish=True)
    elif opt == Optimizer.SHGO:
        res = shgo(objective, bounds=bounds, n=max(n_iter // 2, 1), iters=3, sampling_method="sobol")
    elif opt == Optimizer.BASIN_HOPPING:
        x0 = rng.uniform([b[0] for b in bounds], [b[1] for b in bounds])
        res = basinhopping(objective, x0=x0, niter=n_iter, minimizer_kwargs={"bounds": bounds}, seed=seed)
    elif opt == Optimizer.DUAL_ANNEALING:
        res = dual_annealing(objective, bounds=bounds, maxiter=n_iter, seed=seed)
    else:
        raise OptimError(f"优化器 {opt} 未实现")

    best_x = np.asarray(res.x, dtype=float)
    # 离散变量 round 到最近水平
    for i, v in enumerate(variables):
        levels = getattr(v, "levels", ())
        if levels:
            best_x[i] = float(levels[int(np.argmin(np.abs(np.array(levels) - best_x[i])))])
    best_y = float(res.fun)

    std = None
    try:
        pred_std = surrogate.predict_std(best_x.reshape(1, -1))
        if pred_std is not None:
            std = float(pred_std[0])
    except Exception:
        std = None

    return OptimResult(best_x=best_x, best_y=best_y, best_std=std, optimizer=opt.value, raw_result=res)


def optimize_direct(  # noqa: PLR0913, PLR0912
    template: Template,
    variables: Sequence[object],
    *,
    target: str = "",
    optimizer: Optimizer | str = Optimizer.DIFFERENTIAL_EVOLUTION,
    n_iter: int = 100,
    seed: int = _DEFAULT_SEED,
    penalty: float = 1e10,
    maximize: bool = False,
    callback: Optional[Callable[[np.ndarray, float], None]] = None,
    report: Optional[Callable[..., Any]] = None,
) -> OptimResult:
    """直接把 workflow 当目标函数，在设计空间边界内跑 scipy 全局优化器.

    与 :func:`optimize`（代理优化）对比：

    | 维度 | :func:`optimize` | :func:`optimize_direct` |
    |------|------------------|------------------------|
    | 目标函数 | Surrogate.predict | run_workflow → resolve_outputs |
    | 采样成本 | DOE 一次 | n_iter × popsize 次 FE |
    | 最优性 | 代理上最优（有幻觉风险） | 真实最优 |
    | 适用场景 | 变量多/FE 慢/探索空间 | 变量少/FE 快/验证代理 |

    :param template: 工作流模板（须在 ``output_params`` 里声明至少一个
        标量输出参数作为优化目标）。
    :param variables: 设计变量序列（通常来自
        :attr:`~zylab.doe.design_space.DesignSpace.variables`）。变量名
        必须是 ``"node_id.param_key"`` dotted 格式——优化器内部会 partition
        成 overrides 的 ``{node_id: {param_key: value}}`` 结构。
    :param target: 输出参数名——从
        :meth:`~zylab.flowchart.template.Template.resolve_outputs` 取哪个
        标量当目标。空字符串时取 template 第一个 output_param。
    :param optimizer: 优化器。
    :param n_iter: 最大迭代/评估次数——含义随优化器（见 :func:`optimize`）。
    :param penalty: workflow 运行失败时返回的惩罚值（正值，与 minimize
        语义一致；若 ``maximize=True`` 则内部翻成 ``-penalty``）。
    :param maximize: True 则把目标取负后喂给 scipy 最小化器，
        返回值会翻回原始尺度。
    :param callback: 每代回调 ``(x, f) → None``，用于记录轨迹。
    :param report: 透传给 :func:`~zylab.flowchart.run_workflow` 的进度回调
        （每单次 FE 评估调用一次）。
    :raises OptimError: template 无 output_params / 目标参数缺失 / 变量名
        不合法 / 优化器未知。
    """
    # 延迟导入——optim 是底层包，不依赖 flowchart
    from zylab.flowchart import run_workflow

    # 校验变量名
    for i, v in enumerate(variables):
        name = getattr(v, "name", str(i))
        if "." not in name:
            raise OptimError(f"variables[{i}].name={name!r} 应为 'node_id.param_key' dotted 格式")

    # 确定 target 输出参数
    out_params = getattr(template, "output_params", ())
    if not out_params:
        raise OptimError("template 未声明 output_params，无法确定优化目标")
    if target:
        if target not in {op.name for op in out_params}:
            raise OptimError(f"template 无名为 {target!r} 的 output_param")
    else:
        target = out_params[0].name

    bounds = _build_bounds(variables)
    try:
        opt = Optimizer(optimizer)
    except ValueError as exc:
        raise OptimError(f"未知优化器 {optimizer!r}（可选 {[e.value for e in Optimizer]}）") from exc

    rng = np.random.default_rng(seed)
    penalty_val = -penalty if maximize else penalty

    def _round_discrete(x_arr: np.ndarray) -> np.ndarray:
        """把优化器的实数向量 round 成变量实际取值（离散/整数边界）."""
        fixed = np.asarray(x_arr, dtype=float).copy()
        for i, v in enumerate(variables):
            levels = getattr(v, "levels", ())
            if levels:
                fixed[i] = float(levels[int(np.argmin(np.abs(np.array(levels) - fixed[i])))])
            # 连续变量也 round 到 int——变量名带 .nx/.ny 这类网格参数必是整数
            elif "." in getattr(v, "name", ""):
                # 只 round 看起来是整数型的（lower/upper 都是整数）
                lo, hi = bounds[i]
                if float(lo).is_integer() and float(hi).is_integer():
                    fixed[i] = round(float(fixed[i]))
        # 夹回边界
        for i, (lo, hi) in enumerate(bounds):
            fixed[i] = float(np.clip(fixed[i], lo, hi))
        return fixed

    def _build_overrides(x_fixed: np.ndarray) -> Mapping[str, Mapping[str, Any]]:
        overrides: dict[str, dict[str, Any]] = {}
        for v, xi in zip(variables, x_fixed):
            nid, _, k = getattr(v, "name", "").partition(".")
            val: Any = xi
            lo, hi = bounds[list(variables).index(v)]
            if float(lo).is_integer() and float(hi).is_integer():
                val = round(float(xi))
            overrides.setdefault(nid, {})[k] = val
        return overrides

    def objective(x: np.ndarray) -> float:
        x_fixed = _round_discrete(x)
        overrides = _build_overrides(x_fixed)
        outcome = run_workflow(template, overrides=overrides, report=report)
        if not outcome.succeeded:
            if callback is not None:
                callback(x_fixed.copy(), penalty_val)
            return penalty_val
        resolved = outcome.resolve_outputs(template)
        if target not in resolved:
            if callback is not None:
                callback(x_fixed.copy(), penalty_val)
            return penalty_val
        val = float(resolved[target])
        if maximize:
            val = -val
        if callback is not None:
            callback(x_fixed.copy(), val if not maximize else -val)
        return val

    if opt == Optimizer.DIFFERENTIAL_EVOLUTION:
        res = differential_evolution(objective, bounds=bounds, maxiter=n_iter, seed=seed, polish=True)
    elif opt == Optimizer.SHGO:
        res = shgo(objective, bounds=bounds, n=max(n_iter // 2, 1), iters=3, sampling_method="sobol")
    elif opt == Optimizer.BASIN_HOPPING:
        x0 = rng.uniform([b[0] for b in bounds], [b[1] for b in bounds])
        res = basinhopping(objective, x0=x0, niter=n_iter, minimizer_kwargs={"bounds": bounds}, seed=seed)
    elif opt == Optimizer.DUAL_ANNEALING:
        res = dual_annealing(objective, bounds=bounds, maxiter=n_iter, seed=seed)
    else:
        raise OptimError(f"优化器 {opt} 未实现")

    best_x = _round_discrete(res.x)
    best_y = float(res.fun)
    if maximize:
        best_y = -best_y

    return OptimResult(best_x=best_x, best_y=best_y, best_std=None, optimizer=opt.value, raw_result=res)


# ---------------------------------------------------------------------------
# 多目标 NSGA-II
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParetoOptResult:
    """多目标 Pareto 优化结果容器.

    :ivar X: ``(M, D_in)`` Pareto 前沿上的候选解（D_in 是设计变量数）。
    :ivar F: ``(M, K)`` 每个候选解在 K 个目标上的真实值（minimize 方向）。
    :ivar n_generations: NSGA-II 执行的代数。
    :ivar n_evaluations: 总 FE 评估次数。
    :ivar target_names: 用户指定的目标输出参数名列表。
    """

    X: np.ndarray
    F: np.ndarray
    n_generations: int
    n_evaluations: int
    target_names: tuple[str, ...]


def optimize_pareto(  # noqa: PLR0913, PLR0912
    template: Template,
    variables: Sequence[object],
    *,
    targets: Sequence[str],
    n_population: int = 40,
    n_generations: int = 40,
    seed: int = _DEFAULT_SEED,
    minimize: bool | Sequence[bool] = True,
    penalty: float = 1e10,
    callback: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
    report: Optional[Callable[..., Any]] = None,
    cache: dict[str, Any] | None = None,
    n_workers: int | None = None,
) -> ParetoOptResult:
    """NSGA-II 风格直接多目标优化——跳过代理，把 workflow 当目标函数跑.

    与 :func:`optimize_direct`（单目标 scipy DE）相比，本函数用手写
    NSGA-II 进化算法同时优化多个目标，返回一整条 Pareto 前沿
    （而不是单个最优点）。适合：

    - 工程上常见的「多目标权衡」问题（如刚度 vs 质量、
      应力 vs 位移），scipy 没有现成多目标优化器
    - 设计变量少（≤5）、单次 FE 快（<1 s），允许几千次 FE 评估

    复用 :func:`~zylab.optim.pareto.pareto_ranks` /
    :func:`~zylab.optim.pareto.crowding_distance` 做非支配排序
    和拥挤距离选择，SBX 交叉 + 多项式变异 + 锦标赛选择 都是
    极简实现，零依赖额外库。

    :param template: 工作流模板（须声明至少 len(targets) 个 output_params）。
    :param variables: 设计变量序列（name 必须是 ``"node_id.param_key"`` dotted 格式）。
    :param targets: 多目标 output_param 名列表（如 ``["E", "dmax"]``）。
    :param n_population: NSGA-II 每代种群大小。
    :param n_generations: 进化代数。
    :param seed: 随机种子。
    :param minimize: True 全部目标最小化；逐目标传 bool 序列可混合
        （如 ``[True, False]`` 第 2 目标最大化）。
    :param penalty: workflow 运行失败时该目标维度的惩罚值。
    :param callback: 每代结束回调 ``(X_pop, F_pop) → None``，
        用于记录进化轨迹或画 Pareto 前沿动画。
    :param report: 透传给 :func:`~zylab.flowchart.run_workflow` 的进度回调。
    :param cache: 外部传入的节点级哈希缓存——同进程内多次优化可共享。
    :raises OptimError: template 缺 targets 指定的 output_param /
        variables 名不合法 / n_population 过小。
    """
    # 延迟导入——optim 是底层包，不硬依赖 flowchart / pareto
    from zylab.flowchart import run_batch, run_workflow

    from .pareto import crowding_distance, pareto_ranks

    # --- 参数校验 ---
    K = len(targets)
    if K < 2:
        raise OptimError(f"optimize_pareto 至少需要 2 个目标（当前 {K}）；单目标请用 optimize_direct")
    if n_population < 4:
        raise OptimError(f"n_population 至少 4（NSGA-II 需要足够多 front 0 点稳定选择），实际 {n_population}")
    out_params = getattr(template, "output_params", ())
    available = {op.name for op in out_params}
    missing = [t for t in targets if t not in available]
    if missing:
        raise OptimError(f"template 缺 output_param: {missing}")

    for i, v in enumerate(variables):
        if "." not in getattr(v, "name", ""):
            raise OptimError(f"variables[{i}].name 应为 'node_id.param_key' dotted 格式")

    bounds = _build_bounds(variables)
    D = len(bounds)

    # 统一 minimize 格式
    if isinstance(minimize, bool):
        mins = [minimize] * K
    else:
        mins = list(minimize)
    maximize_mask = [not m for m in mins]

    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    eta_c = 20.0  # SBX 分布指数
    eta_m = 20.0  # 多项式变异分布指数
    p_m = 1.0 / D

    cache_local: dict[str, Any] | None = cache if cache is not None else {}

    def _round(x: np.ndarray) -> np.ndarray:
        """边界夹 + 整数型变量 round."""
        fixed = np.clip(x.copy(), lo, hi)
        for i, _v in enumerate(variables):
            if float(lo[i]).is_integer() and float(hi[i]).is_integer():
                fixed[i] = round(float(fixed[i]))
        return fixed

    def _build_overrides(x_fixed: np.ndarray) -> Mapping[str, Mapping[str, Any]]:
        overrides: dict[str, dict[str, Any]] = {}
        for v, xi in zip(variables, x_fixed):
            nid, _, k = getattr(v, "name", "").partition(".")
            val: Any = xi
            vi = list(variables).index(v)
            if float(lo[vi]).is_integer() and float(hi[vi]).is_integer():
                val = round(float(xi))
            overrides.setdefault(nid, {})[k] = val
        return overrides

    def evaluate(x_fixed: np.ndarray) -> np.ndarray:
        """单次 FE 评估，返回 K 目标值（全 minimize 方向）."""
        outcome = run_workflow(template, overrides=_build_overrides(x_fixed), report=report, cache=cache_local)
        F = np.full(K, float(penalty))
        if outcome.succeeded:
            resolved = outcome.resolve_outputs(template)
            for k, tname in enumerate(targets):
                if tname in resolved:
                    val = float(resolved[tname])
                    # 最大化目标翻成最小化（内部统一 minimize）
                    F[k] = -val if maximize_mask[k] else val
        return F

    # 根据 n_workers 决定走串行 evaluate 闭包还是 run_batch 并行
    _parallel = n_workers is not None and n_workers >= 2

    def _batch_evaluate(pop_matrix: np.ndarray) -> np.ndarray:
        """批量评估种群——n_workers>=2 时用 run_batch 并行."""
        N = len(pop_matrix)
        if not _parallel:
            return np.array([evaluate(p) for p in pop_matrix])

        vnames = [getattr(v, "name", "") for v in variables]
        rows: list[dict[str, Any]] = []
        for p in pop_matrix:
            row = {}
            for j, vn in enumerate(vnames):
                row[vn] = float(p[j])
            rows.append(row)
        outcomes = run_batch(template, rows, n_workers=n_workers, cache=cache_local)
        F = np.full((N, K), float(penalty))
        for idx, outcome in enumerate(outcomes):
            if outcome.succeeded:
                resolved = outcome.resolve_outputs(template)
                for k, tname in enumerate(targets):
                    if tname in resolved:
                        val = float(resolved[tname])
                        F[idx, k] = -val if maximize_mask[k] else val
        return F

    # --- 初始化种群 ---
    pop = rng.uniform(lo, hi, (n_population, D))
    pop = np.array([_round(p) for p in pop])
    F_pop = _batch_evaluate(pop)

    n_eval = n_population
    offspring = np.empty_like(pop)
    F_off = np.empty_like(F_pop)

    def tournament() -> np.ndarray:
        """二元锦标赛：rank 小的赢；rank 相同 cd 大的赢."""
        ranks = pareto_ranks(F_pop)
        cd = crowding_distance(F_pop)
        cand = rng.choice(n_population, 3, replace=False)
        # lexsort: 先按 ranks 升、再按 -cd 升（大的在前）
        order = np.lexsort((-cd[cand], ranks[cand]))
        return pop[cand[order[0]]].copy()

    for _gen in range(n_generations):
        # --- 生成子代 SBX + 多项式变异 ---
        for i in range(0, n_population, 2):
            p1 = tournament()
            p2 = tournament()
            u = rng.uniform(size=D)
            beta_q = np.where(
                u <= 0.5,
                (2 * u) ** (1.0 / (eta_c + 1)),
                (0.5 / np.maximum(u, 1e-12)) ** (1.0 / (eta_c + 1)),
            )
            c1 = 0.5 * ((1 + beta_q) * p1 + (1 - beta_q) * p2)
            c2 = 0.5 * ((1 - beta_q) * p1 + (1 + beta_q) * p2)
            for idx, child in enumerate([c1, c2]):
                mask = rng.uniform(size=D) < p_m
                for j in range(D):
                    if mask[j]:
                        span = hi[j] - lo[j]
                        if span < 1e-12:
                            continue
                        d1 = child[j] - lo[j]
                        d2 = hi[j] - child[j]
                        u2 = rng.uniform()
                        if u2 < 0.5:
                            delta = (2 * u2 + (1 - 2 * u2) * (1 - d1 / span) ** (eta_m + 1)) ** (1.0 / (eta_m + 1)) - 1
                        else:
                            delta = 1 - (2 * (1 - u2) + 2 * (u2 - 0.5) * (1 - d2 / span) ** (eta_m + 1)) ** (
                                1.0 / (eta_m + 1)
                            )
                        child[j] += delta * span
                child[:] = _round(child)
                if i + idx < n_population:
                    offspring[i + idx] = child

        F_off = _batch_evaluate(offspring)
        n_eval += n_population

        # --- 环境选择：父代 + 子代 合并保留 n_population ---
        combined_pop = np.vstack([pop, offspring])
        combined_F = np.vstack([F_pop, F_off])
        ranks_all = pareto_ranks(combined_F)
        cd_all = crowding_distance(combined_F)

        keep: list[int] = []
        front = 0
        while len(keep) < n_population and front <= int(ranks_all.max()):
            idx_f = np.where(ranks_all == front)[0]
            remaining = n_population - len(keep)
            if len(idx_f) <= remaining:
                keep.extend(int(i) for i in idx_f)
            else:
                order = np.argsort(-cd_all[idx_f])
                keep.extend(int(idx_f[i]) for i in order[:remaining])
                break
            front += 1

        keep_arr = np.array(keep, dtype=int)
        pop = combined_pop[keep_arr]
        F_pop = combined_F[keep_arr]

        if callback is not None:
            callback(pop.copy(), F_pop.copy())

    # --- 最后提取 Pareto front 0 ---
    ranks_final = pareto_ranks(F_pop)
    front0 = np.where(ranks_final == 0)[0]
    X_pareto = pop[front0]
    F_pareto = F_pop[front0]

    # 内部翻成 minimize 的目标翻回用户语义
    for k, flip in enumerate(maximize_mask):
        if flip:
            F_pareto[:, k] = -F_pareto[:, k]

    return ParetoOptResult(
        X=X_pareto,
        F=F_pareto,
        n_generations=n_generations,
        n_evaluations=n_eval,
        target_names=tuple(targets),
    )
