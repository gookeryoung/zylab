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
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.optimize import basinhopping, differential_evolution, dual_annealing, shgo

from .errors import OptimError
from .surrogate import Surrogate

__all__ = ["OptimResult", "Optimizer", "optimize"]


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
