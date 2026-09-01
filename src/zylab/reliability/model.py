"""感度试验响应模型：Logistic/正态/Gumbel/Weibull 响应概率与伯努利似然.

感度试验（GJB/Z 377A《感度试验用数理统计方法》框架）以临界刺激量
为随机变量，刺激量 ``x`` 下试样响应概率 ``p(x) = F(x; μ, σ)``。
支持四种响应模型：

- ``logistic``：Logistic 分布 CDF（兰利/OSTR/升降/D-优化序贯法惯用）；
- ``normal``：正态分布 CDF（概率单位法 Probit 模型）；
- ``gumbel``：最小极值分布 CDF（位置-尺度族；Weibull 临界刺激量取
  对数后的分布），感度低尾建模常用；
- ``weibull``：双参数 Weibull CDF，``μ`` 为尺度参数 η、``σ`` 为形状
  参数 k（刺激量须为正），感度分布非对称建模。

前三种为位置-尺度参数化 ``F((x-μ)/σ)``；Weibull 为 ``(η, k)``
参数化。概率/分位/似然计算复用 scipy 专用函数保证数值稳定。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit, log_ndtr, ndtr
from scipy.stats import gumbel_l, norm, weibull_min

from .errors import ReliabilityError

__all__ = ["MODEL_NAMES", "inverse_prob", "neg_log_likelihood", "response_prob"]

#: 支持的响应模型名（Logistic / 正态 / Gumbel / Weibull）
MODEL_NAMES = ("logistic", "normal", "gumbel", "weibull")


def response_prob(model: str, x: Any, mu: float, sigma: float) -> Any:
    """响应概率 ``p(x) = F(x; μ, σ)``.

    :param model: 模型名（:data:`MODEL_NAMES`）。
    :param x: 刺激量（标量或数组；``weibull`` 须全为正）。
    :param mu: 位置参数（``weibull`` 为尺度参数 η，须为正）。
    :param sigma: 尺度参数（``weibull`` 为形状参数 k）。
    :raises ReliabilityError: 模型名非法、参数非正或刺激量非法。
    """
    if model not in MODEL_NAMES:
        raise ReliabilityError(f"响应模型 {model!r} 不受支持（可选 {MODEL_NAMES}）")
    if sigma <= 0.0:
        raise ReliabilityError(f"感度标准差/形状参数须为正，得到 {sigma!r}")
    z = np.asarray(x, dtype=float)
    if model == "logistic":
        return expit((z - mu) / sigma)
    if model == "normal":
        return ndtr((z - mu) / sigma)
    if model == "gumbel":
        return gumbel_l.cdf(z, loc=mu, scale=sigma)
    if mu <= 0.0:
        raise ReliabilityError(f"weibull 尺度参数 μ(η) 须为正，得到 {mu!r}")
    if np.any(z <= 0.0):
        raise ReliabilityError("weibull 模型刺激量须全为正")
    return weibull_min.cdf(z, sigma, scale=mu)


def neg_log_likelihood(theta: np.ndarray, x: np.ndarray, y: np.ndarray, model: str) -> float:
    """伯努利试验负对数似然.

    位置-尺度模型参数化 ``theta = (μ, ln σ)`` 保证 σ > 0；Weibull
    参数化 ``theta = (ln η, ln k)`` 保证 η、k > 0。

    :param theta: 参数向量（含义随模型）。
    :param x: 刺激量序列。
    :param y: 响应指示序列（0/1）。
    :param model: 响应模型名。
    """
    # 优化器（Nelder-Mead 扩张/轮廓搜索）可能探测极端 ln σ/ln η/ln k，
    # 直接 exp 会溢出 float64；clip 到 ±500 后 exp 仍达 1e±217，
    # 足以让 NLL 数值稳定地排斥该区域而不产生 RuntimeWarning。
    t0, t1 = float(np.clip(theta[0], -500.0, 500.0)), float(np.clip(theta[1], -500.0, 500.0))
    if model == "weibull":
        eta, shape = float(np.exp(t0)), float(np.exp(t1))
        # 伯努利响应概率：P(y=1) = F(x)（CDF），P(y=0) = exp(−(x/η)^k)
        log_p = weibull_min.logcdf(x, shape, scale=eta)
        log_q = -((x / eta) ** shape)
    else:
        mu, sigma = t0, float(np.exp(t1))
        z = (x - mu) / sigma
        if model == "logistic":
            log_p = -np.logaddexp(0.0, -z)
            log_q = -np.logaddexp(0.0, z)
        elif model == "normal":
            log_p = log_ndtr(z)
            log_q = log_ndtr(-z)
        else:
            log_p = gumbel_l.logcdf(z)
            log_q = gumbel_l.logsf(z)
    # 极端参数下 log 似然项可达 -1e300 量级，直接求和会溢出 float64；
    # 逐项封底到 -1e300 再求和（项数远小于 1e8 时和必有界），
    # 封顶保持"排斥该区域"的语义且无 RuntimeWarning。
    terms = np.clip(np.where(y > 0, log_p, log_q), -1.0e300, 0.0)
    nll = float(-np.sum(terms))
    return nll if np.isfinite(nll) else 1.0e300


def inverse_prob(model: str, p: float, mu: float, sigma: float) -> float:
    """响应概率 ``p`` 对应的分位刺激量 ``F⁻¹(p)``（D-最优候选点构造）."""
    if model == "logistic":
        return mu + sigma * float(np.log(p / (1.0 - p)))
    if model == "normal":
        return mu + sigma * float(norm.ppf(p))
    if model == "gumbel":
        return float(gumbel_l.ppf(p, loc=mu, scale=sigma))
    return float(weibull_min.ppf(p, sigma, scale=mu))


def _info_matrix(xs: Any, model: str, mu: float, sigma: float) -> np.ndarray:
    """期望 Fisher 信息矩阵（2×2，参数 ``(μ, σ)`` 语义随模型）.

    单点信息 ``g gᵀ / (p(1−p))``，其中 ``g = ∂p/∂θ`` 取中心差分数值
    微分——位置-尺度与 Weibull ``(η, k)`` 参数化统一处理，供 D-最优
    设计器与协方差估计复用。
    """
    values = np.asarray(xs, dtype=float)
    theta = np.array([float(mu), float(sigma)])
    gradients = np.zeros((values.size, 2))
    for i in range(2):
        step = max(abs(theta[i]) * 1.0e-5, 1.0e-7)
        hi, lo = theta.copy(), theta.copy()
        hi[i] += step
        lo[i] -= step
        p_hi = np.asarray(response_prob(model, values, float(hi[0]), float(hi[1])), dtype=float)
        p_lo = np.asarray(response_prob(model, values, float(lo[0]), float(lo[1])), dtype=float)
        gradients[:, i] = (p_hi - p_lo) / (2.0 * step)
    p = np.asarray(response_prob(model, values, mu, sigma), dtype=float)
    weight = np.where((p > 0.0) & (p < 1.0), 1.0 / np.maximum(p * (1.0 - p), 1.0e-300), 0.0)
    g0 = gradients[:, 0] * np.sqrt(weight)
    g1 = gradients[:, 1] * np.sqrt(weight)
    return np.array(
        [
            [float(g0 @ g0), float(g0 @ g1)],
            [float(g0 @ g1), float(g1 @ g1)],
        ]
    )
