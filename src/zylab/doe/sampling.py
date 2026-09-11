"""采样方法：全因子 / 拉丁超立方 / Sobol / 纯随机.

统一入口 :func:`sample` 在单位超立方体 ``[0, 1]^d`` 上按选定
采样器生成样本，再由 :class:`~zylab.doe.variable.DesignVariable`
做连续/离散映射还原到实际值。两种输出形态：

- **实际值矩阵** ``X``：shape ``(n_samples, n_vars)``，供 workflow 直接消费；
- **单位坐标矩阵** ``U``：shape ``(n_samples, n_vars)``，供均匀性评估或
  后续变换（如可靠性的 ``inverse_prob`` 采样）。
"""

from __future__ import annotations

from enum import Enum
from typing import Sequence

import numpy as np
from scipy.stats import qmc

from .errors import DoeError
from .variable import DesignVariable

__all__ = ["SamplingMethod", "sample", "sample_unit"]


class SamplingMethod(str, Enum):
    """支持的采样方法枚举."""

    FULL_FACTORIAL = "full_factorial"  # 全因子（笛卡尔积）
    LATIN_HYPERCUBE = "latin_hypercube"  # 拉丁超立方
    SOBOL = "sobol"  # Sobol 低差异序列
    RANDOM = "random"  # 纯随机均匀分布


#: 默认随机种子（保证可复现）
_DEFAULT_SEED = 42


# ------------------------------------------------------------------ 单位超立方体采样


def _sample_factorial_unit(d: int, n_per_dim: int = 5) -> np.ndarray:
    """全因子单位采样：``n_per_dim^d`` 个点，d 较大时样本数爆炸.

    :raises DoeError: ``n_per_dim^d > 1_000_000``（安全上限，防内存溢出）。
    """
    total = n_per_dim**d
    if total > 1_000_000:
        raise DoeError(
            f"全因子在 d={d}, n_per_dim={n_per_dim} 下共 {total} 点，超过安全上限 1e6。"
            " 请改用 LatinHypercube/Sobol，或减少维度/每维水平数。"
        )
    # 每维均匀网格 [0.5/n, 1.5/n, ..., (n-0.5)/n]（中心点法，避免边界退化）
    grids = [np.linspace(0.5 / n_per_dim, 1.0 - 0.5 / n_per_dim, n_per_dim) for _ in range(d)]
    mesh = np.meshgrid(*grids, indexing="ij")
    return np.column_stack([m.ravel() for m in mesh])


def _sample_lhc_unit(d: int, n: int, seed: int) -> np.ndarray:
    """拉丁超立方单位采样（scipy qmc）."""
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    return sampler.random(n=n)


def _sample_sobol_unit(d: int, n: int, seed: int) -> np.ndarray:
    """Sobol 低差异序列单位采样（scipy qmc）."""
    # QMC 通常要求 n 为 2^k，但 qmc.Sobol 对任意 n 也能跑（自动截断）
    sampler = qmc.Sobol(d=d, seed=seed, scramble=True)
    return sampler.random(n=n)


def _sample_random_unit(d: int, n: int, seed: int) -> np.ndarray:
    """纯随机均匀单位采样."""
    rng = np.random.default_rng(seed)
    return rng.random((n, d))


# ------------------------------------------------------------------ 公开入口


def sample_unit(
    variables: Sequence[DesignVariable],
    method: SamplingMethod | str,
    n_samples: int | None = None,
    *,
    seed: int = _DEFAULT_SEED,
    n_per_dim: int = 5,
) -> np.ndarray:
    """在单位超立方体上按指定方法采样（shape ``(n, d)``）.

    :param variables: 设计变量列表（决定维度 ``d``）。
    :param method: 采样方法。
    :param n_samples: 目标样本数（``FULL_FACTORIAL`` 时忽略，由
        ``n_per_dim ** d`` 自动决定）。
    :param seed: 随机种子（保证可复现）。
    :param n_per_dim: 全因子法每维水平数（仅 ``FULL_FACTORIAL`` 使用）。
    :raises DoeError: 样本数非法 / 维度为 0 / 全因子点数爆炸。
    """
    if not variables:
        raise DoeError("至少需 1 个设计变量才能采样")
    for v in variables:
        v.validate()

    d = len(variables)
    try:
        m = SamplingMethod(method)
    except ValueError as exc:
        raise DoeError(f"未知采样方法 {method!r}（可选 {[e.value for e in SamplingMethod]}）") from exc

    if m == SamplingMethod.FULL_FACTORIAL:
        return _sample_factorial_unit(d, n_per_dim)
    if n_samples is None or n_samples < 1:
        raise DoeError(f"{m.value!r} 采样需显式提供 n_samples ≥ 1，得到 {n_samples!r}")
    if m == SamplingMethod.LATIN_HYPERCUBE:
        return _sample_lhc_unit(d, n_samples, seed)
    if m == SamplingMethod.SOBOL:
        return _sample_sobol_unit(d, n_samples, seed)
    if m == SamplingMethod.RANDOM:
        return _sample_random_unit(d, n_samples, seed)
    raise DoeError(f"未知采样方法 {m.value!r}")


def sample(
    variables: Sequence[DesignVariable],
    method: SamplingMethod | str,
    n_samples: int | None = None,
    *,
    seed: int = _DEFAULT_SEED,
    n_per_dim: int = 5,
) -> np.ndarray:
    """生成实际值采样矩阵（shape ``(n_samples, n_vars)``）.

    先在单位超立方体采样，再用各 :class:`DesignVariable.denormalize`
    映射：连续变量线性拉伸到 ``[lower, upper]``，离散变量取最近水平。

    :return: ``ndarray[float64]``，每行一个设计点，每列对应一个变量（
        列顺序与 ``variables`` 相同）。离散变量值保留原始 ``levels`` 原值。
    """
    unit = sample_unit(variables, method, n_samples, seed=seed, n_per_dim=n_per_dim)
    X = np.empty_like(unit)
    for i, v in enumerate(variables):
        col = v.denormalize(unit[:, i])
        X[:, i] = np.asarray(col, dtype=float)
    return X
