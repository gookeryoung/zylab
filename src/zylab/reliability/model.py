"""感度试验响应模型：Logistic/正态响应概率与伯努利似然.

感度试验（GJB/Z 377A《感度试验用数理统计方法》框架）以临界刺激量
为随机变量，刺激量 ``x`` 下试样响应概率 ``p(x) = F((x-μ)/σ)``，
其中 ``μ`` 为 50% 响应点（位置参数）、``σ`` 为感度散布（尺度参数）。
标准支持两种响应模型：

- ``logistic``：Logistic 分布 CDF（兰利/OSTR/升降/D-优化序贯法惯用）；
- ``normal``：正态分布 CDF（概率单位法 Probit 模型）。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit, log_ndtr, ndtr
from scipy.stats import norm

from .errors import ReliabilityError

__all__ = ["MODEL_NAMES", "inverse_prob", "neg_log_likelihood", "response_prob"]

#: 支持的响应模型名（Logistic / 正态）
MODEL_NAMES = ("logistic", "normal")


def response_prob(model: str, x: Any, mu: float, sigma: float) -> Any:
    """响应概率 ``p(x) = F((x-μ)/σ)``.

    :param model: 模型名（``logistic``/``normal``）。
    :param x: 刺激量（标量或数组）。
    :param mu: 50% 响应点。
    :param sigma: 感度标准差（须为正）。
    :raises ReliabilityError: 模型名非法或 sigma 非正。
    """
    if model not in MODEL_NAMES:
        raise ReliabilityError(f"响应模型 {model!r} 不受支持（可选 {MODEL_NAMES}）")
    if sigma <= 0.0:
        raise ReliabilityError(f"感度标准差须为正，得到 {sigma!r}")
    z = (np.asarray(x, dtype=float) - mu) / sigma
    if model == "logistic":
        return expit(z)
    return ndtr(z)


def neg_log_likelihood(theta: np.ndarray, x: np.ndarray, y: np.ndarray, model: str) -> float:
    """伯努利试验负对数似然（参数化 ``theta = (μ, ln σ)`` 保证 σ > 0）.

    :param theta: ``(μ, ln σ)`` 参数向量。
    :param x: 刺激量序列。
    :param y: 响应指示序列（0/1）。
    :param model: 响应模型名。
    """
    mu, log_sigma = float(theta[0]), float(theta[1])
    sigma = np.exp(log_sigma)
    z = (x - mu) / sigma
    if model == "logistic":
        log_p = -np.logaddexp(0.0, -z)
        log_q = -np.logaddexp(0.0, z)
    else:
        log_p = log_ndtr(z)
        log_q = log_ndtr(-z)
    return float(-np.sum(np.where(y > 0, log_p, log_q)))


def inverse_prob(model: str, p: float, mu: float, sigma: float) -> float:
    """响应概率 ``p`` 对应的分位刺激量 ``F⁻¹(p)``（D-最优候选点构造）."""
    if model == "logistic":
        return mu + sigma * float(np.log(p / (1.0 - p)))
    return mu + sigma * float(norm.ppf(p))
