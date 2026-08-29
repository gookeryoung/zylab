"""线性求解器：约束消元 + 稀疏 LU 直接求解.

约束处理采用自由/约束自由度划块法：

    K_ff u_f = f_f - K_fc u_c

其中 u_c 为指定位移（默认 0）。求解后回代得到约束自由度反力。
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

from .errors import SolverError

__all__ = ["solve_system"]


def solve_system(
    k_global: csr_matrix,
    force: np.ndarray,
    fixed_dofs: np.ndarray,
    fixed_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """求解带位移约束的线性方程组 K u = f.

    Args:
        k_global: 全局刚度矩阵（CSR，方阵）。
        force: 全局载荷向量。
        fixed_dofs: 约束自由度索引数组（0 基，升序与否不限）。
        fixed_values: 约束位移值数组（与 fixed_dofs 等长）。

    Returns:
        (u, reactions)：u 为完整位移向量；reactions 为约束自由度上的反力
        （与 fixed_dofs 顺序对应，即 K u - f 在约束自由度处的分量）。

    Raises:
        SolverError: 刚度矩阵奇异（约束不足或机构）时抛出。
    """
    n = force.size
    if k_global.shape != (n, n):
        raise SolverError(f"刚度矩阵形状 {k_global.shape} 与载荷维度 {n} 不匹配")
    fixed = np.asarray(fixed_dofs, dtype=np.intp)
    values = np.asarray(fixed_values, dtype=float)
    if fixed.size != values.size:
        raise SolverError(f"约束自由度数 {fixed.size} 与约束值数 {values.size} 不一致")
    if np.any(fixed >= n) or np.any(fixed < 0):
        raise SolverError("约束自由度索引越界")

    mask = np.zeros(n, dtype=bool)
    mask[fixed] = True
    free = np.flatnonzero(~mask)
    u = np.zeros(n)
    u[fixed] = values
    if free.size == 0:  # pragma: no cover（全约束病态输入，防御分支）
        return u, np.zeros(0)

    # scipy 稀疏矩阵的花式切片在运行时可用，pyrefly 的 scipy 存根未覆盖
    k_ff = k_global[free][:, free]  # type: ignore[bad-index]
    rhs = force[free].copy()
    if fixed.size:
        rhs -= k_global[free][:, fixed] @ values  # type: ignore[bad-index]
    try:
        u[free] = splu(k_ff.tocsc()).solve(rhs)  # type: ignore[union-attr]
    except RuntimeError as exc:
        raise SolverError("刚度矩阵奇异：约束不足或存在机构，无法求解") from exc
    reactions = np.asarray(k_global @ u - force)[fixed]
    return u, reactions
