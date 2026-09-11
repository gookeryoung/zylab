"""Pareto 多目标分析：支配排序、前沿提取、拥挤距离、标量化.

针对 DOE / batch 探索产出的多输出样本集 ``Y (N, K)``，提供：

- :func:`pareto_ranks` —— 快速非支配排序（NSGA-II 同款分层算法）
- :func:`pareto_front` —— 提取指定 Pareto 前沿的样本索引
- :func:`crowding_distance` —— 拥挤距离（NSGA-II 多样性度量）
- :func:`scalarize` —— 加权和 / epsilon-constraint 标量化
- :func:`pareto_summary` —— 一键统计所有前沿规模 + 边界框

不依赖 FE / flowchart 层，纯 numpy 实现；
与 :func:`zylab.flowchart.batch.run_batch_outputs`
返回的 ``Y (N, n_outputs)`` 天然对接。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "ParetoSummary",
    "crowding_distance",
    "pareto_front",
    "pareto_ranks",
    "pareto_summary",
    "scalarize",
]


# ---------------------------------------------------------------------------
# 非支配排序
# ---------------------------------------------------------------------------


def pareto_ranks(
    Y: np.ndarray,
    *,
    minimize: bool | Sequence[bool] = True,
) -> np.ndarray:
    """NSGA-II 风格快速非支配排序.

    :param Y: ``(N, K)`` 目标矩阵，N 样本、K 目标。
    :param minimize: True 表示所有目标都是**越小越好**；
        传长度 K 的布尔序列可逐目标指定
        （如 ``[True, False]`` 表示第 2 个目标要最大化）。
    :return: ``(N,)`` 每个点的支配等级——0 = Pareto 最优前沿，
        1 = 被 front 0 支配但不互相支配，以此类推。

    复杂度 O(N²K)——对 DOE 探索常见规模（N ≤ 500）秒级。
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    N, K = Y.shape

    # 统一转成最小化问题
    if isinstance(minimize, bool):
        mins = [minimize] * K
    else:
        mins = list(minimize)
    Y_eff = Y.copy()
    for k, is_min in enumerate(mins):
        if not is_min:
            Y_eff[:, k] = -Y_eff[:, k]

    # 预计算支配关系
    # dominates[i, j] = True  表示 i 严格支配 j（所有目标 ≤，且至少一个 <）
    dominates = np.zeros((N, N), dtype=bool)
    for i in range(N):
        diff = Y_eff[i, :] - Y_eff[np.arange(N), :]  # (N, K) = Y[i] - Y[j]
        # 所有维度 ≤ 0 且至少一个 < 0
        all_le = (diff <= 0).all(axis=1)
        has_lt = (diff < 0).any(axis=1)
        dominates[i] = all_le & has_lt
        dominates[i, i] = False  # 自己不支配自己

    dominated_count = dominates.sum(axis=0)  # 每个点被多少其他点支配

    ranks = np.full(N, -1, dtype=int)
    current_rank = 0
    # 动态剥离前沿：与 NSGA-II 原算法一致
    while True:
        front_idx = np.where((dominated_count == 0) & (ranks == -1))[0]
        if len(front_idx) == 0:
            break
        ranks[front_idx] = current_rank
        # 把这些点支配的那些点的计数减 1
        dominated_count -= dominates[front_idx].sum(axis=0)
        current_rank += 1

    return ranks


def pareto_front(
    Y: np.ndarray,
    *,
    rank: int = 0,
    minimize: bool | Sequence[bool] = True,
) -> np.ndarray:
    """提取指定 Pareto 前沿的样本索引.

    :param Y: ``(N, K)`` 目标矩阵。
    :param rank: 要提取的前沿编号，0=Pareto 最优（默认）。
    :param minimize: 见 :func:`pareto_ranks`。
    :return: ``(M,)`` 索引数组，M 是 rank 前沿的样本数。
        若 rank 超过最大前沿，返回空数组。
    """
    ranks = pareto_ranks(Y, minimize=minimize)
    return np.where(ranks == rank)[0]


# ---------------------------------------------------------------------------
# 拥挤距离（多样性）
# ---------------------------------------------------------------------------


def crowding_distance(Y: np.ndarray, *, minimize: bool | Sequence[bool] = True) -> np.ndarray:
    """NSGA-II 拥挤距离——衡量解在目标空间中周围有多密集.

    拥挤距离大 → 该解周围样本少，多样性好。front 边界点距离 = inf。

    :param Y: ``(M, K)`` 目标矩阵，通常只传一个前沿的点
        （如果传所有点，**必须**先按 pareto 前沿分组再各算各的——
        本函数不负责分组，用户自己确保传入同一 rank 的点）。
    :return: ``(M,)`` 每个点的拥挤距离。
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    M, K = Y.shape
    if M <= 2:
        return np.full(M, np.inf)  # 边界点都是 inf

    if isinstance(minimize, bool):
        mins = [minimize] * K
    else:
        mins = list(minimize)
    Y_eff = Y.copy()
    for k, is_min in enumerate(mins):
        if not is_min:
            Y_eff[:, k] = -Y_eff[:, k]

    dist = np.zeros(M)
    for k in range(K):
        order = np.argsort(Y_eff[:, k])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        lo, hi = Y_eff[order[0], k], Y_eff[order[-1], k]
        span = hi - lo
        if span < 1e-12:
            continue
        y_k = Y_eff[order, k]
        # 中间点：(y[k+1] - y[k-1]) / span
        dist[order[1:-1]] += (y_k[2:] - y_k[:-2]) / span
    return dist


# ---------------------------------------------------------------------------
# 标量化（多目标 → 单目标辅助）
# ---------------------------------------------------------------------------


def scalarize(
    Y: np.ndarray,
    *,
    weights: np.ndarray | Sequence[float] | None = None,
    method: str = "weighted_sum",
    ref_point: np.ndarray | None = None,
) -> np.ndarray:
    """多目标标量化——把 (N, K) 压成 (N,).

    :param Y: ``(N, K)`` 目标矩阵（已经是 minimize 方向的值）。
    :param weights: ``(K,)`` 权重；None 时均匀权重。
    :param method:

        - ``"weighted_sum"`` —— ``sum(w_k * y_k)``（凸前沿上有效）
        - ``"chebyshev"`` —— ``max(w_k * |y_k - z_k|)`` 切比雪夫范数，
          对非凸前沿能找到 Pareto 解
        - ``"epsilon"`` —— 对 ref_point 做硬约束：
          ``sum(w_k * y_k)`` 但违反约束 ``y_k ≤ ref_point_k`` 时返回 inf

    :param ref_point: Chebyshev / epsilon 需要的参考点 ``(K,)``。
    :return: ``(N,)`` 标量目标函数值。
    """
    Y = np.asarray(Y, dtype=float)
    N, K = Y.shape
    if weights is None:
        weights = np.ones(K) / K
    w = np.asarray(weights, dtype=float)
    assert w.shape == (K,), f"weights 形状应为 ({K},)，实际 {w.shape}"

    if method == "weighted_sum":
        return (Y * w).sum(axis=1)
    if method == "chebyshev":
        if ref_point is None:
            ref_point = np.zeros(K)
        return np.max(np.abs(Y - ref_point) * w, axis=1)
    if method == "epsilon":
        if ref_point is None:
            raise ValueError("epsilon 方法必须提供 ref_point")
        penalty = np.full(N, np.inf)
        ok = (ref_point >= Y).all(axis=1)
        penalty[ok] = (Y[ok] * w).sum(axis=1)
        return penalty
    raise ValueError(f"未知标量化方法 {method!r}（可选 weighted_sum / chebyshev / epsilon）")


# ---------------------------------------------------------------------------
# 汇总容器
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParetoSummary:
    """Pareto 分析汇总."""

    ranks: np.ndarray  # (N,) 每个样本的支配等级
    n_fronts: int  # 前沿总数
    front_sizes: np.ndarray  # (n_fronts,) 每个前沿的样本数
    best_idx: int  # front 0 中拥挤距离最大的样本索引（代表性 Pareto 解）


def pareto_summary(
    Y: np.ndarray,
    *,
    minimize: bool | Sequence[bool] = True,
) -> ParetoSummary:
    """一键统计 Pareto 前沿规模 + 挑一个代表性最优解.

    典型用法：DOE 跑完 ``X, Y`` 后::

        ps = pareto_summary(Y, minimize=[True, True])
        best_x = X[ps.best_idx]  # Pareto front 0 中最孤立的点（多样性好）
    """
    ranks = pareto_ranks(Y, minimize=minimize)
    n_fronts = ranks.max() + 1
    front_sizes = np.array([int((ranks == f).sum()) for f in range(n_fronts)])

    # front 0 中拥挤距离最大的点
    f0_idx = pareto_front(Y, rank=0, minimize=minimize)
    if len(f0_idx) == 0:
        best_idx = -1
    elif len(f0_idx) <= 2:
        best_idx = int(f0_idx[0])
    else:
        cd = crowding_distance(Y[f0_idx], minimize=minimize)
        best_idx = int(f0_idx[np.argmax(cd)])

    return ParetoSummary(
        ranks=ranks,
        n_fronts=n_fronts,
        front_sizes=front_sizes,
        best_idx=best_idx,
    )
