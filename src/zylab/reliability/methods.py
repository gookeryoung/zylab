"""感度试验设计器：GJB/Z 377A 六种方法的下一刺激量规则.

序贯法（101-104）提供 ``next_level`` 规则——给定已试水平与响应序列
返回下一试验刺激量：

- **方法101 兰利法**：下一水平取「最大全不响应水平」与「最小全响应
  水平」的中点，初始取区间中点，保证试验水平始终夹在响应/不响应之间；
- **方法102 OSTR 法**（最优序贯规则，Wu 1985）：在兰利混合区间内
  以 D-最优准则（Fisher 信息行列式最大化）选点，MLE 不可估时回退
  兰利中点规则；
- **方法103 升降法**（Bruceton）：固定步长，上发响应则降一级、
  不响应则升一级；
- **方法104 D-优化法**：候选点覆盖全区间 ``F⁻¹(p)`` 网格，以当前
  MLE（不可估时用初始猜测）计算期望信息行列式取最大。

非序贯法（201-202）提供固定/步进设计表：

- **方法201 概率单位法**：等间隔水平表，每水平 ``n_per_level`` 发；
- **方法202 完全步进法**：自下界按固定步长逐级上升，每水平
  ``n_per_level`` 发，升至全响应水平后补一级收尾（Kärber 分析要求
  响应频率覆盖 0% 至 100%）。
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .errors import ReliabilityError
from .model import inverse_prob, response_prob

__all__ = [
    "METHOD_LABELS",
    "METHOD_NAMES",
    "doptimal_next",
    "langlie_next",
    "ostr_next",
    "probit_levels",
    "stepstress_levels",
    "updown_next",
]

#: 支持的感度试验方法名（GJB/Z 377A 方法编号）
METHOD_NAMES = ("langlie", "ostr", "updown", "doptimal", "probit", "stepstress")

#: 方法编号与中文名（报告/界面展示用）
METHOD_LABELS = {
    "langlie": "方法101 兰利法",
    "ostr": "方法102 OSTR法",
    "updown": "方法103 升降法",
    "doptimal": "方法104 D优化法",
    "probit": "方法201 概率单位法",
    "stepstress": "方法202 完全步进法",
}

#: D-最优候选点响应概率网格（覆盖信息量最大的中部与尾部区域）
_D_P_GRID = np.linspace(0.05, 0.95, 19)


def _mixed_interval(levels: np.ndarray, responses: np.ndarray, x_low: float, x_high: float) -> tuple[float, float]:
    """兰利混合区间 ``(D0, D1)``：最大全不响应水平与最小全响应水平.

    全不响应水平 ``x``：所有 ≤ x 的试验均未响应；全响应水平 ``x``：
    所有 ≥ x 的试验均响应。无记录时用初始区间界兜底。
    """
    lower: list[float] = [x_low]
    upper: list[float] = [x_high]
    for level in levels:
        if np.all(responses[levels <= level] == 0):
            lower.append(float(level))
        if np.all(responses[levels >= level] == 1):
            upper.append(float(level))
    return max(lower), min(upper)


def langlie_next(levels: np.ndarray, responses: np.ndarray, x_low: float, x_high: float) -> float:
    """兰利法下一水平：混合区间中点（初始为 ``[x_low, x_high]`` 中点）."""
    d0, d1 = _mixed_interval(levels, responses, x_low, x_high)
    return 0.5 * (d0 + d1)


def ostr_next(  # noqa: PLR0913, PLR0917  设计器签名与标准方法规则一一对应
    levels: np.ndarray,
    responses: np.ndarray,
    x_low: float,
    x_high: float,
    model: str,
    mu: float,
    sigma: float,
) -> float:
    """OSTR 法下一水平：混合区间内 D-最优候选点（MLE 不可估回退兰利中点）.

    :param mu: 当前参数估计 μ（MLE 或初始猜测）。
    :param sigma: 当前参数估计 σ。
    """
    d0, d1 = _mixed_interval(levels, responses, x_low, x_high)
    if d1 - d0 <= max(abs(d0), abs(d1)) * 1.0e-12:
        return 0.5 * (d0 + d1)
    candidates = np.unique(np.clip(np.array([inverse_prob(model, p, mu, sigma) for p in _D_P_GRID]), d0, d1))
    return _argmax_d_criterion(candidates, levels, model, mu, sigma)


def doptimal_next(  # noqa: PLR0913, PLR0917  设计器签名与标准方法规则一一对应
    levels: np.ndarray,
    responses: np.ndarray,
    x_low: float,
    x_high: float,
    model: str,
    mu: float,
    sigma: float,
) -> float:
    """D-优化法下一水平：全区间 ``F⁻¹(p)`` 网格中 D-最优候选点."""
    del responses  # 全区间设计，不依赖响应历史（保持设计器签名对称）
    candidates = np.unique(np.clip(np.array([inverse_prob(model, p, mu, sigma) for p in _D_P_GRID]), x_low, x_high))
    return _argmax_d_criterion(candidates, levels, model, mu, sigma)


def _argmax_d_criterion(
    candidates: np.ndarray,
    levels: np.ndarray,
    model: str,
    mu: float,
    sigma: float,
) -> float:
    """在候选点中选使加入该点后期望 Fisher 信息行列式最大者.

    期望信息按伯努利响应计算：单点信息 ``I(x) = g gᵀ / (p(1-p))``，
    其中 ``g = ∂p/∂θ = [f(z)/σ, z·f(z)/σ]ᵀ``（位置-尺度参数化）。
    """
    best_x, best_det = float(candidates[0]), -np.inf
    for x in candidates:
        info = _info_matrix(np.append(levels, x), model, mu, sigma)
        determinant = float(np.linalg.det(info))
        if determinant > best_det:
            best_x, best_det = float(x), determinant
    return best_x


def _info_matrix(xs: np.ndarray, model: str, mu: float, sigma: float) -> np.ndarray:
    """位置-尺度参数的期望 Fisher 信息矩阵（2×2）."""
    z = (np.asarray(xs, dtype=float) - mu) / sigma
    p = np.asarray(response_prob(model, z, 0.0, 1.0), dtype=float)
    density = p * (1.0 - p) if model == "logistic" else np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    weight = np.where((p > 0.0) & (p < 1.0), density / np.maximum(p * (1.0 - p), 1.0e-300), 0.0) / sigma
    g0 = weight
    g1 = weight * z
    return np.array(
        [
            [float(np.sum(g0 * g0)), float(np.sum(g0 * g1))],
            [float(np.sum(g0 * g1)), float(np.sum(g1 * g1))],
        ]
    )


def updown_next(levels: np.ndarray, responses: np.ndarray, x_low: float, x_high: float, step: float) -> float:
    """升降法下一水平：上一发响应降一级、不响应升一级（初始取区间中点）.

    :param step: 固定步长（须为正）。
    """
    if step <= 0.0:
        raise ReliabilityError(f"升降法步长须为正，得到 {step!r}")
    if levels.size == 0:
        return 0.5 * (x_low + x_high)
    previous = float(levels[-1])
    return previous - step if responses[-1] > 0 else previous + step


def probit_levels(x_low: float, x_high: float, n_levels: int) -> np.ndarray:
    """概率单位法固定水平表：区间内 ``n_levels`` 个等间隔水平."""
    if n_levels < 2:
        raise ReliabilityError(f"概率单位法水平数须 ≥ 2，得到 {n_levels}")
    return np.linspace(x_low, x_high, int(n_levels))


def stepstress_levels(
    x_low: float,
    x_high: float,
    step: float,
    n_per_level: int,
    counts: Callable[[float], int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """完全步进法水平表：自下界逐级上升，每水平 ``n_per_level`` 发.

    停止规则：升至首个全响应水平（计数 = ``n_per_level``）后再补一级
    确认收尾；始终无全响应则试至上界。返回 ``(水平表, 每水平响应计数)``。

    :param counts: 水平 -> 响应计数回调（模拟器注入；缺省全 0 仅生成
        到上界的完整表，用于设计预览）。
    """
    if step <= 0.0:
        raise ReliabilityError(f"步进法步长须为正，得到 {step!r}")
    if n_per_level < 1:
        raise ReliabilityError(f"每水平发数须 ≥ 1，得到 {n_per_level}")
    levels: list[float] = []
    hits: list[int] = []
    remaining_after_full = -1
    x = float(x_low)
    while x <= x_high + step * 0.5:
        hit = int(counts(x)) if counts is not None else 0
        levels.append(x)
        hits.append(hit)
        if hit >= n_per_level and remaining_after_full < 0:
            remaining_after_full = 1
        elif remaining_after_full > 0:
            remaining_after_full -= 1
            if remaining_after_full == 0:
                break
        x += step
    return np.asarray(levels), np.asarray(hits, dtype=int)
