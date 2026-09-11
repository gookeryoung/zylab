"""代理上的优化：在 DesignSpace 边界内用 scipy 全局优化器找代理最优.

最小链路：

1. DOE 采样 → 训练样本 ``(X_train, y_train)``
2. :class:`~zylab.optim.surrogate.Surrogate.fit(X_train, y_train)`
3. :func:`optimize` 把代理目标函数交给 scipy 全局优化器，
   边界自动从 :class:`~zylab.doe.design_space.DesignSpace` 变量
   的 ``lower/upper`` 取（离散变量按 levels 映射）
4. 返回 :class:`OptimResult`——含最优实际值、代理预测、迭代轨迹等
5. 可选：用真函数在最优解处重跑，对比代理值与真实值

这里的优化问题是 **minimization**——若用户有最大化目标，
请把 ``y_train`` 先乘 ``-1``，拿到结果再翻回去。

:func:`optimize_direct` 跳过代理训练，直接把 workflow 当目标函数
跑 scipy 全局优化器。适合：
- 设计变量少（≤5）、单次 FE 快（<1 s）
- 追求真实最优、拒绝 surrogate 插值幻觉
- 做参数扫描前的快速缩圈

两种调用方式共用 :class:`Optimizer` 枚举和 :class:`OptimResult` 容器。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import basinhopping, differential_evolution, dual_annealing, shgo

from .errors import OptimError
from .surrogate import Surrogate

if TYPE_CHECKING:
    from zylab.flowchart import Template

__all__ = ["OptimResult", "Optimizer", "optimize", "optimize_direct"]


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
