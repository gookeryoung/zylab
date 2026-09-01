"""升降法 Dixon-Mood 核心估计、中间参数与 G/H 查表系数的数值标定（GJB/Z 377A 方法103）.

Dixon-Mood 分析以频数较少的响应类别计数：设其在第 ``i`` 级（自最低试验
水平起算）计数 ``n_i``，``n = Σn_i``、``A = Σi·n_i``、``B = Σi²·n_i``、
``M = (nB − A²)/n²``，则（GB/T 24176 / Collins《Mechanical Design of
Machine Elements》形式）::

    μ̂ = x_min + d·(A/n ∓ 0.5)     （响应计数取 −0.5，不响应计数取 +0.5）
    σ̂ = 1.620·d·(M + 0.029)       （M ≥ 0.3；M < 0.3 时 σ̂ = 0.53·d）
    ρ = σ̂/d = 1.620·(M + 0.029)

标准误系数 G/H（``se(μ̂) = G·σ̂/√n``、``se(σ̂) = H·σ̂/√n``）标准中按 ρ
查表；本模块以蒙特卡洛标定（正态真值、初始水平相位均匀化、每组 1600
发 × 40000 次重复的大样本渐近）将查表值固化为 ρ 网格上的系数表，线性
插值取值，免去人工查表。

标定复核（对照 ``ref/升降法试验数据_20250330.xlsx`` 查表值）：ρ=1.52 时
标定 G≈0.960 与表值 0.961 一致；表值 H=1.735 为原 Dixon-Mood 表口径
（开方式 σ̂ 的 σ̂² 标准误系数），与本模块线性式自洽标定值（H≈1.58，
σ̂ 标准误系数）不同——线性式下沿用原表 H 会低估 σ̂ 的不确定度，故以
自洽标定口径为准。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import ReliabilityError
from .model import response_prob

__all__ = [
    "DixonMoodDetail",
    "dixon_mood_core",
    "gh_factors",
    "simulate_updown",
]

#: ρ = σ̂/d 网格（0.5~3.0 步长 0.1；0.5 < d/σ < 2 为升降法适用域）
_RHO_GRID = np.array(
    [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
        1.10,
        1.20,
        1.30,
        1.40,
        1.50,
        1.60,
        1.70,
        1.80,
        1.90,
        2.00,
        2.10,
        2.20,
        2.30,
        2.40,
        2.50,
        2.60,
        2.70,
        2.80,
        2.90,
        3.00,
    ]
)

#: se(μ̂) = G·σ̂/√n 的 G 系数（按 _RHO_GRID 蒙特卡洛标定）
_G_TABLE = np.array(
    [
        1.153,
        1.098,
        1.064,
        1.039,
        1.018,
        1.000,
        0.995,
        0.984,
        0.968,
        0.967,
        0.960,
        0.955,
        0.950,
        0.945,
        0.946,
        0.940,
        0.940,
        0.939,
        0.923,
        0.929,
        0.927,
        0.931,
        0.927,
        0.926,
        0.921,
        0.919,
    ]
)

#: se(σ̂) = H·σ̂/√n 的 H 系数（按 _RHO_GRID 蒙特卡洛标定；ρ=0.5 处因
#: σ̂ 触及 0.53d 退化分支而显著偏小，为标定的真实形态）
_H_TABLE = np.array(
    [
        0.355,
        1.222,
        1.253,
        1.291,
        1.336,
        1.372,
        1.419,
        1.456,
        1.500,
        1.532,
        1.575,
        1.610,
        1.646,
        1.695,
        1.727,
        1.761,
        1.785,
        1.832,
        1.871,
        1.898,
        1.940,
        1.952,
        1.999,
        2.035,
        2.058,
        2.083,
    ]
)

#: M < 0.3 时 σ̂ 的退化分支系数（σ̂ = 0.53·d）
_SIGMA_DEGENERATE = 0.53

#: 线性式 σ̂ 的斜率与修正常数（σ̂ = 1.620·d·(M + 0.029)）
_SIGMA_SLOPE = 1.620
_SIGMA_BIAS = 0.029

#: 线性式 M 的适用下界
_M_MIN = 0.3


@dataclass(frozen=True)
class DixonMoodDetail:
    """Dixon-Mood 中间参数与标准误系数（GJB/Z 377A 方法103 查表参数）.

    :param n_used: 较少响应类别的计数 n（点估计与标准误的有效样本量）。
    :param a_value: 加权和 A = Σ i·n_i。
    :param b_value: 加权和 B = Σ i²·n_i。
    :param m_value: 方差形式统计量 M = (nB − A²)/n²。
    :param rho: 查表参数 ρ = σ̂/d = 1.620·(M + 0.029)。
    :param g_factor: se(μ̂) = G·σ̂/√n 的系数 G。
    :param h_factor: se(σ̂) = H·σ̂/√n 的系数 H。
    """

    n_used: int
    a_value: float
    b_value: float
    m_value: float
    rho: float
    g_factor: float
    h_factor: float

    def standard_errors(self, sigma_hat: float) -> tuple[float, float]:
        """按 G/H 系数计算 ``(se(μ̂), se(σ̂))``（大样本近似）."""
        root_n = math.sqrt(self.n_used)
        return self.g_factor * sigma_hat / root_n, self.h_factor * sigma_hat / root_n


def gh_factors(rho: float) -> tuple[float, float]:
    """按 ρ 线性插值取 ``(G, H)`` 标准误系数（网格外取端点值）."""
    g_factor = float(np.interp(rho, _RHO_GRID, _G_TABLE))
    h_factor = float(np.interp(rho, _RHO_GRID, _H_TABLE))
    return g_factor, h_factor


def dixon_mood_core(x: np.ndarray, y: np.ndarray, step: float) -> tuple[float, float, DixonMoodDetail]:
    """Dixon-Mood 点估计核心与中间参数（无 bootstrap/标准误，供批量复用）.

    以频数较少的响应类别在各水平（自最低试验水平起算）的计数计算
    ``(μ̂, σ̂, 中间参数)``；数据无混合响应时抛 :class:`ReliabilityError`。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
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
    m_value = (n_used * b_value - a_value**2) / n_used**2
    mean_shift = a_value / n_used - (0.5 if use_response else -0.5)
    mu_hat = float(x.min() + step * mean_shift)
    if m_value >= _M_MIN:
        rho = _SIGMA_SLOPE * (m_value + _SIGMA_BIAS)
    else:
        rho = _SIGMA_DEGENERATE
    sigma_hat = max(rho * step, step * 0.1)
    g_factor, h_factor = gh_factors(rho)
    return (
        mu_hat,
        sigma_hat,
        DixonMoodDetail(
            n_used=n_used,
            a_value=a_value,
            b_value=b_value,
            m_value=m_value,
            rho=rho,
            g_factor=g_factor,
            h_factor=h_factor,
        ),
    )


def simulate_updown(  # noqa: PLR0913  模型两参数 + 试验三要素 + 随机源，语义不可合并
    mu: float, sigma: float, n_total: int, step: float, *, x_start: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """按正态模型模拟一组升降试验（固定步长，供参数 bootstrap）."""
    levels: list[float] = []
    responses: list[int] = []
    x_current = float(x_start)
    for _ in range(n_total):
        probability = float(response_prob("normal", x_current, mu, sigma))
        hit = 1 if rng.random() < probability else 0
        levels.append(x_current)
        responses.append(hit)
        x_current = x_current - step if hit else x_current + step
    return np.asarray(levels), np.asarray(responses, dtype=int)
