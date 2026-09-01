"""感度试验数据分析与试验总装：MLE / Dixon-Mood / Karber 估计.

估计方法按 GJB/Z 377A 各方法的配套分析规则分派：

- 序贯法（兰利/OSTR/D-优化）与方法201：极大似然估计（数值优化 +
  数值 Hessian 标准误，scipy 实现）；
- 方法103 升降法：Dixon-Mood 公式（Bruceton 经典分析）；
- 方法202 完全步进法：Spearman-Kärber 非参数估计。

:func:`run_sensitivity_test` 总装「设计 → 蒙特卡洛模拟 → 分析」全流程，
给定真值参数 ``(μ, σ)`` 模拟感度试验并给出统计估计，供 DSL 模板以
固定种子做示例验证。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .errors import ReliabilityError
from .methods import (
    METHOD_LABELS,
    METHOD_NAMES,
    doptimal_next,
    langlie_next,
    neyer_next,
    ostr_next,
    probit_levels,
    stepstress_levels,
    updown_next,
)
from .model import MODEL_NAMES, _info_matrix, neg_log_likelihood, response_prob

__all__ = [
    "SensitivityEstimate",
    "SensitivityTestResult",
    "dixon_mood",
    "karber",
    "mle_estimate",
    "run_sensitivity_test",
]

#: 拟合响应曲线取样点数
_CURVE_POINTS = 200


@dataclass(frozen=True)
class SensitivityEstimate:
    """感度参数估计结果.

    :param mu: 50% 响应点估计。
    :param sigma: 感度标准差估计。
    :param se_mu: μ 的近似标准误（数值 Hessian 逆对角元）。
    :param se_sigma: σ 的近似标准误。
    :param estimator: 估计方法名（MLE/Dixon-Mood/Kärber）。
    :param converged: 估计是否收敛（数据完全分离时 MLE 不存在）。
    """

    mu: float
    sigma: float
    se_mu: float = float("nan")
    se_sigma: float = float("nan")
    estimator: str = "MLE"
    converged: bool = True


@dataclass(frozen=True)
class SensitivityTestResult:
    """感度试验模拟与统计结果（DSL 结果引用的数据载荷）.

    :param method: 方法名（``langlie``/``ostr``/...）。
    :param method_label: 方法中文名（含标准编号）。
    :param model: 响应模型名（``logistic``/``normal``）。
    :param levels: 试验刺激量序列。
    :param responses: 响应指示序列（0/1，与 levels 等长）。
    :param estimate: 统计估计结果。
    :param curve_x: 拟合响应曲线取样刺激量。
    :param curve_p: 拟合响应概率（与 curve_x 等长）。
    """

    method: str
    method_label: str
    model: str
    levels: np.ndarray
    responses: np.ndarray
    estimate: SensitivityEstimate
    curve_x: np.ndarray
    curve_p: np.ndarray

    @property
    def mu_hat(self) -> float:
        """μ 估计值（DSL text 引用便捷属性）."""
        return self.estimate.mu

    @property
    def sigma_hat(self) -> float:
        """σ 估计值（DSL text 引用便捷属性）."""
        return self.estimate.sigma


def _is_separated(x: np.ndarray, y: np.ndarray) -> bool:
    """完全分离检测：按刺激量排序后响应序列单调（0 全在 1 一侧或镜像）.

    同一水平既有响应又有不响应时似然存在内点最优，MLE 必存在。
    """
    if np.all(y == 0) or np.all(y == 1):
        return True
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    diffs = np.diff(ys)
    # 同水平混合响应（相邻等值刺激量标签相异）→ 非分离
    if bool(np.any((xs[1:] == xs[:-1]) & (diffs != 0))):
        return False
    return bool(np.all(diffs >= 0) or np.all(diffs <= 0))


def mle_estimate(model: str, x: np.ndarray, y: np.ndarray) -> SensitivityEstimate:
    """极大似然估计 ``(μ, σ)``（Nelder-Mead 优化 + 数值 Hessian 标准误）.

    数据完全分离（全响应/全不响应或存在分离阈值）时 MLE 不存在，
    返回 ``converged=False`` 与区间中点近似。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if x.size < 2:
        raise ReliabilityError("MLE 至少需要 2 个试验点")
    if model == "weibull" and np.any(x <= 0.0):
        raise ReliabilityError("weibull 模型刺激量须全为正")
    if _is_separated(x, y):
        return SensitivityEstimate(
            mu=float(0.5 * (x.min() + x.max())),
            sigma=float(max((x.max() - x.min()) / 4.0, 1.0e-9)),
            estimator="MLE",
            converged=False,
        )
    responded, silent = x[y > 0], x[y == 0]
    mu0 = 0.5 * (float(responded.mean()) + float(silent.mean()))
    sigma0 = max(float(x.std()) * 0.5, float(x.max() - x.min()) / 20.0, 1.0e-6)
    if model == "weibull":
        # Weibull 参数化 (ln η, ln k)：尺度初值取响应/不响应重心均值，形状初值 1.2
        theta0 = np.array([np.log(max(mu0, 1.0e-6)), np.log(1.2)])
    else:
        theta0 = np.array([mu0, np.log(sigma0)])
    result = minimize(
        neg_log_likelihood,
        theta0,
        args=(x, y, model),
        method="Nelder-Mead",
        options={"xatol": 1.0e-10, "fatol": 1.0e-12, "maxiter": 2000},
    )
    mu_hat, sigma_hat = (
        (float(np.exp(result.x[0])), float(np.exp(result.x[1])))
        if model == "weibull"
        else (float(result.x[0]), float(np.exp(result.x[1])))
    )
    hessian = _numerical_hessian(result.x, x, y, model)
    covariance = np.linalg.inv(hessian) if np.all(np.isfinite(hessian)) else np.full((2, 2), np.nan)
    # 参数化 (μ, ln σ) 或 (ln η, ln k)：δ法传导至原参数尺度
    se_mu = float(np.sqrt(covariance[0, 0])) if covariance[0, 0] > 0.0 else float("nan")
    se_log_sigma = float(np.sqrt(covariance[1, 1])) if covariance[1, 1] > 0.0 else float("nan")
    return SensitivityEstimate(
        mu=mu_hat,
        sigma=sigma_hat,
        se_mu=mu_hat * se_mu if model == "weibull" else se_mu,
        se_sigma=abs(sigma_hat * se_log_sigma),
        estimator="MLE",
        converged=bool(result.success),
    )


def _numerical_hessian(
    theta: np.ndarray, x: np.ndarray, y: np.ndarray, model: str, delta: float = 1.0e-4
) -> np.ndarray:
    """负对数似然的中心差分 Hessian（步长相对参数量级自适应）."""
    steps = np.maximum(np.abs(theta) * delta, delta)
    hessian = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ei, ej = np.zeros(2), np.zeros(2)
            ei[i], ej[j] = steps[i], steps[j]
            hessian[i, j] = (
                neg_log_likelihood(theta + ei + ej, x, y, model)
                - neg_log_likelihood(theta + ei - ej, x, y, model)
                - neg_log_likelihood(theta - ei + ej, x, y, model)
                + neg_log_likelihood(theta - ei - ej, x, y, model)
            ) / (4.0 * steps[i] * steps[j])
    return hessian


def dixon_mood(x: np.ndarray, y: np.ndarray, step: float) -> SensitivityEstimate:
    """升降法 Dixon-Mood 分析（以频数较少的响应类别计数）.

    设 ``n_i`` 为较少类别在第 ``i`` 级（自最低试验水平起算）的计数，
    ``A = Σ i·n_i``、``B = Σ i²·n_i``，则::

        μ̂ = x_min + d·(A/n ± 0.5)    （不响应计数取 +0.5，响应计数取 -0.5）
        σ̂ = 1.620·d·sqrt((n·B − A²)/n² + 0.029)

    协方差修正：标准误按 Dixon-Mood 隐含的正态假设，取试验点上期望
    Fisher 信息矩阵（正态模型）逆的对角元（大样本近似）。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if step <= 0.0:
        raise ReliabilityError(f"升降法步长须为正，得到 {step!r}")
    n_response = int(np.sum(y > 0))
    use_response = n_response <= (y.size - n_response)
    counts = y > 0 if use_response else y == 0
    if not counts.any():
        raise ReliabilityError("升降法数据无混合响应，Dixon-Mood 分析不可用")
    indices = np.round((x - x.min()) / step).astype(int)
    order = np.argsort(indices)
    group_indices = indices[order][counts[order]]
    n_used = int(counts.sum())
    a_value = float(group_indices.sum())
    b_value = float((group_indices**2).sum())
    mean_shift = a_value / n_used - (0.5 if use_response else -0.5)
    mu_hat = float(x.min() + step * mean_shift)
    variance = max((n_used * b_value - a_value**2) / n_used**2 + 0.029, 0.0)
    sigma_hat = float(1.620 * step * np.sqrt(variance))
    sigma_final = max(sigma_hat, step * 0.1)
    se_mu, se_sigma = _fisher_standard_errors(x, "normal", mu_hat, sigma_final)
    return SensitivityEstimate(
        mu=mu_hat,
        sigma=sigma_final,
        se_mu=se_mu,
        se_sigma=se_sigma,
        estimator="Dixon-Mood",
    )


def _fisher_standard_errors(x: np.ndarray, model: str, mu: float, sigma: float) -> tuple[float, float]:
    """期望 Fisher 信息矩阵逆的对角元平方根（协方差修正，大样本近似）.

    信息矩阵奇异（试验点信息不足）时返回 NaN。
    """
    try:
        covariance = np.linalg.inv(_info_matrix(x, model, mu, sigma))
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    se_mu = float(np.sqrt(covariance[0, 0])) if covariance[0, 0] > 0.0 else float("nan")
    se_sigma = float(np.sqrt(covariance[1, 1])) if covariance[1, 1] > 0.0 else float("nan")
    return se_mu, se_sigma


def karber(x_levels: np.ndarray, hits: np.ndarray, n_per_level: int) -> SensitivityEstimate:
    """完全步进法 Spearman-Kärber 非参数估计.

    频率序列 ``p_i`` 视为 CDF 在边界 ``[a, b]``（下/上各补一格，p=0/1）
    上的取值，梯形积分结合分部积分得::

        μ̂ = b − Σ ½(p_i + p_{i+1})·(x_{i+1} − x_i)
        σ̂² = b² − Σ ½(p_i + p_{i+1})·(x_{i+1}² − x_i²) − μ̂²
    """
    x = np.asarray(x_levels, dtype=float)
    hits = np.asarray(hits, dtype=int)
    if x.size < 2 or n_per_level < 1:
        raise ReliabilityError("Kärber 分析至少需要 2 个水平且每水平发数 ≥ 1")
    spacing = float(np.diff(np.unique(x)).mean())
    edges = np.concatenate(([x[0] - spacing], x, [x[-1] + spacing]))
    probs = np.concatenate(([0.0], hits / n_per_level, [1.0]))
    upper = float(edges[-1])
    mu_hat = upper - float(np.sum(0.5 * (probs[:-1] + probs[1:]) * np.diff(edges)))
    second = float(np.sum(0.5 * (probs[:-1] + probs[1:]) * np.diff(edges**2)))
    sigma_hat = float(np.sqrt(max(upper * upper - second - mu_hat**2, 0.0)))
    return SensitivityEstimate(mu=mu_hat, sigma=max(sigma_hat, spacing * 0.5), estimator="Kärber")


def run_sensitivity_test(  # noqa: PLR0913  六方法统一入口，参数即标准配置项
    *,
    method: str,
    model: str = "logistic",
    mu: float = 10.0,
    sigma: float = 1.0,
    n_total: int = 30,
    x_low: float = 6.0,
    x_high: float = 14.0,
    step: float = 1.0,
    n_per_level: int = 10,
    n_levels: int = 7,
    seed: int = 7,
) -> SensitivityTestResult:
    """总装感度试验：设计生成 → 蒙特卡洛模拟 → 统计分析.

    :param method: 试验方法名（:data:`~zylab.reliability.methods.METHOD_NAMES`）。
    :param model: 响应模型名（``logistic``/``normal``/``gumbel``/``weibull``，
        Weibull 参数化为 ``μ=尺度 η``、``σ=形状 k``）。
    :param mu: 真值 50% 响应点（``weibull`` 为尺度 η，模拟用）。
    :param sigma: 真值感度标准差（``weibull`` 为形状 k，模拟用）。
    :param n_total: 序贯法总发数。
    :param x_low: 初始区间下界（全不响应估计界）。
    :param x_high: 初始区间上界（全响应估计界）。
    :param step: 升降法/步进法固定步长。
    :param n_per_level: 概率单位法/步进法每水平发数。
    :param n_levels: 概率单位法水平数。
    :param seed: 随机种子（固定种子保证可重复验证）。
    :raises ReliabilityError: 方法/模型名非法或参数配置不合法。
    """
    if method not in METHOD_NAMES:
        raise ReliabilityError(f"感度试验方法 {method!r} 不受支持（可选 {METHOD_NAMES}）")
    if model not in MODEL_NAMES:
        raise ReliabilityError(f"响应模型 {model!r} 不受支持（可选 {MODEL_NAMES}）")
    if sigma <= 0.0 or n_total < 4:
        raise ReliabilityError("真值 σ 须为正且序贯发数 ≥ 4")
    rng = np.random.default_rng(seed)
    if method in ("langlie", "ostr", "updown", "doptimal", "neyer"):
        levels, responses = _run_sequential(method, model, mu, sigma, n_total, x_low, x_high, step, rng)
        estimate = dixon_mood(levels, responses, step) if method == "updown" else mle_estimate(model, levels, responses)
    elif method == "probit":
        levels, responses = _run_probit(model, mu, sigma, x_low, x_high, n_levels, n_per_level, rng)
        estimate = mle_estimate(model, levels, responses)
    else:
        levels, responses = _run_stepstress(model, mu, sigma, x_low, x_high, step, n_per_level, rng)
        unique_levels = np.unique(levels)
        estimate = karber(
            unique_levels, np.array([int(responses[levels == lv].sum()) for lv in unique_levels]), n_per_level
        )
    curve_x = np.linspace(float(levels.min()), float(levels.max()), _CURVE_POINTS)
    curve_p = np.asarray(response_prob(model, curve_x, estimate.mu, estimate.sigma), dtype=float)
    return SensitivityTestResult(
        method=method,
        method_label=METHOD_LABELS[method],
        model=model,
        levels=np.asarray(levels, dtype=float),
        responses=np.asarray(responses, dtype=int),
        estimate=estimate,
        curve_x=curve_x,
        curve_p=curve_p,
    )


def _run_sequential(  # noqa: PLR0913, PLR0917  序贯法共享逐发循环，参数即标准配置项
    method: str,
    model: str,
    mu: float,
    sigma: float,
    n_total: int,
    x_low: float,
    x_high: float,
    step: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """序贯法（101-104）模拟：逐发设计 → 伯努利响应."""
    levels: list[float] = []
    responses: list[int] = []
    mu_guess, sigma_guess = 0.5 * (x_low + x_high), (x_high - x_low) / 6.0
    for _ in range(int(n_total)):
        history_x = np.asarray(levels)
        history_y = np.asarray(responses, dtype=int)
        if method == "langlie":
            x_next = langlie_next(history_x, history_y, x_low, x_high)
        elif method == "updown":
            x_next = updown_next(history_x, history_y, x_low, x_high, step)
        else:
            estimate = mle_estimate(model, history_x, history_y) if history_x.size >= 4 else None
            usable = estimate if estimate is not None and estimate.converged else None
            mu_use = usable.mu if usable is not None else mu_guess
            sigma_use = usable.sigma if usable is not None else sigma_guess
            if method == "ostr":
                x_next = ostr_next(history_x, history_y, x_low, x_high, model, mu_use, sigma_use)
            elif method == "neyer":
                x_next = neyer_next(history_x, history_y, x_low, x_high, model, mu_use, sigma_use)
            else:
                x_next = doptimal_next(history_x, history_y, x_low, x_high, model, mu_use, sigma_use)
        probability = float(response_prob(model, x_next, mu, sigma))
        levels.append(float(x_next))
        responses.append(1 if rng.random() < probability else 0)
    return np.asarray(levels), np.asarray(responses, dtype=int)


def _run_probit(  # noqa: PLR0913, PLR0917  参数即标准配置项
    model: str,
    mu: float,
    sigma: float,
    x_low: float,
    x_high: float,
    n_levels: int,
    n_per_level: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """概率单位法（201）模拟：固定水平表每水平多发."""
    levels: list[float] = []
    responses: list[int] = []
    for level in probit_levels(x_low, x_high, n_levels):
        probability = float(response_prob(model, level, mu, sigma))
        hits = int(rng.binomial(int(n_per_level), probability))
        levels.extend([float(level)] * int(n_per_level))
        responses.extend([1] * hits + [0] * (int(n_per_level) - hits))
    return np.asarray(levels), np.asarray(responses, dtype=int)


def _run_stepstress(  # noqa: PLR0913, PLR0917  参数即标准配置项
    model: str,
    mu: float,
    sigma: float,
    x_low: float,
    x_high: float,
    step: float,
    n_per_level: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """完全步进法（202）模拟：逐级上升每水平多发，全响应后补一级收尾."""
    binomial_cache: dict[float, int] = {}

    def counts(level: float) -> int:
        if level not in binomial_cache:
            probability = float(response_prob(model, level, mu, sigma))
            binomial_cache[level] = int(rng.binomial(int(n_per_level), probability))
        return binomial_cache[level]

    level_table, _hits = stepstress_levels(x_low, x_high, step, n_per_level, counts)
    levels: list[float] = []
    responses: list[int] = []
    for level in level_table:
        hits = counts(float(level))
        levels.extend([float(level)] * int(n_per_level))
        responses.extend([1] * hits + [0] * (int(n_per_level) - hits))
    return np.asarray(levels), np.asarray(responses, dtype=int)
