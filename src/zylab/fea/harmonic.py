"""谐响应分析编排：装配 K/M/C -> 约束划块 -> 频率扫描直接求解.

对外主入口 :func:`solve_harmonic`，返回 :class:`HarmonicResponse`
（各激励频率下的复位移幅值场）。阻尼采用 Rayleigh 模型 ``C = αM + βK``，
每个频率点求解 ``(K + iωC - ω²M) u = f``（划块消元，约束值须为 0）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

from .assemble import assemble_loads, assemble_mass, assemble_stiffness
from .boundary import StaticCase
from .errors import SolverError
from .material import LinearElastic, Section
from .mesh import Mesh
from .modal import _validate_constraints

__all__ = ["HarmonicResponse", "solve_harmonic"]


def _no_report(_progress: float, _message: str) -> None:
    """空进度回调（未注入 report 时使用）."""


@dataclass(frozen=True)
class HarmonicResponse:
    """谐响应分析结果.

    Attributes:
        mesh: 参与求解的网格。
        frequencies: 激励圆频率序列 ``(n_freq,)``（rad/s，与输入同序）。
        displacements: 复位移幅值场 ``(n_dofs, n_freq)``，约束自由度分量为 0，
            实部为与载荷同相分量、虚部为正交（滞后 90°）分量。
    """

    mesh: Mesh
    frequencies: np.ndarray
    displacements: np.ndarray

    @property
    def n_frequencies(self) -> int:
        """频率点数."""
        return int(self.frequencies.size)

    def amplitude(self, index: int) -> np.ndarray:
        """取指定频率点的全场位移幅值 ``(n_dofs,)``."""
        return np.abs(self.displacements[:, index])

    def phase(self, index: int) -> np.ndarray:
        """取指定频率点的全场相位（rad，相对载荷相位）."""
        return np.angle(self.displacements[:, index])

    def node_response(self, node: int, index: int) -> np.ndarray:
        """取指定节点在指定频率点的复位移 ``(dofs_per_node,)``."""
        base = node * self.mesh.dofs_per_node
        return self.displacements[base : base + self.mesh.dofs_per_node, index].copy()


def rayleigh_damping(
    alpha: float,
    beta: float,
    mass: csr_matrix,
    stiffness: csr_matrix,
) -> csr_matrix:
    """组合 Rayleigh 阻尼矩阵 ``C = αM + βK``.

    Args:
        alpha: 质量比例系数（低频阻尼主导项）。
        beta: 刚度比例系数（高频阻尼主导项）。
        mass: 全局质量矩阵。
        stiffness: 全局刚度矩阵。

    Returns:
        阻尼矩阵（与输入同形 CSR）。
    """
    if alpha < 0.0 or beta < 0.0:
        raise SolverError(f"Rayleigh 阻尼系数须非负，实际 alpha={alpha}, beta={beta}")
    return (alpha * mass + beta * stiffness).tocsr()


def solve_harmonic(  # noqa: PLR0913  与静力/模态共用网格四要素 + 工况/频率/阻尼，语义不可合并
    mesh: Mesh,
    materials: Sequence[LinearElastic],
    sections: Sequence[Section],
    case: StaticCase,
    frequencies: Sequence[float],
    *,
    alpha: float = 0.0,
    beta: float = 0.0,
    report: Callable[[float, str], None] | None = None,
) -> HarmonicResponse:
    """线性谐响应分析（稳态简谐响应，频率扫描直接法）.

    对每个激励圆频率 ω 求解 ``(K + iωC - ω²M) u = f``，其中
    ``C = αM + βK``（Rayleigh 阻尼），``f`` 由工况载荷装配。
    约束按划块消元处理（约束值须为 0，非零基础激励不在 v1 范围）。

    Args:
        mesh: 网格。
        materials: 材料表（须配置正的质量密度）。
        sections: 截面表（ElementBlock.section 索引引用）。
        case: 载荷工况（约束参与划块；边压力/体力同静力装配）。
        frequencies: 激励圆频率序列（rad/s，须非负且非空）。
        alpha: Rayleigh 质量比例阻尼系数。
        beta: Rayleigh 刚度比例阻尼系数。
        report: 进度回调 ``(progress, message)``；进程执行器会自动注入。

    Returns:
        :class:`HarmonicResponse`，含各频率点复位移幅值场。

    Raises:
        SolverError: 频率序列空/含负值、约束缺失或非零、阻尼系数为负、
            或动刚度矩阵奇异（共振点无阻尼）时抛出。
    """
    progress = _no_report if report is None else report

    freqs = np.asarray(frequencies, dtype=float)
    if freqs.size == 0:
        raise SolverError("谐响应分析频率序列为空")
    if np.any(freqs < 0.0):
        raise SolverError("激励圆频率须非负")
    if alpha < 0.0 or beta < 0.0:
        raise SolverError(f"Rayleigh 阻尼系数须非负，实际 alpha={alpha}, beta={beta}")

    progress(0.1, "校验约束与载荷")
    fixed_dofs = _validate_constraints(mesh, case.constraints)
    force = assemble_loads(mesh, case, sections)

    progress(0.3, "装配刚度/质量/阻尼矩阵")
    k_global = assemble_stiffness(mesh, materials, sections)
    m_global = assemble_mass(mesh, materials, sections)

    mask = np.zeros(mesh.n_dofs, dtype=bool)
    mask[fixed_dofs] = True
    free = np.flatnonzero(~mask)
    # scipy 稀疏矩阵的花式切片在运行时可用，pyrefly 的 scipy 存根未覆盖
    k_ff = k_global[free][:, free]  # type: ignore[bad-index]
    m_ff = m_global[free][:, free]  # type: ignore[bad-index]
    c_ff = rayleigh_damping(alpha, beta, m_ff, k_ff)  # type: ignore[arg-type]
    f_free = force[free]

    displacements = np.zeros((mesh.n_dofs, freqs.size), dtype=complex)
    for i, omega in enumerate(freqs):
        progress(0.3 + 0.65 * (i + 1) / freqs.size, f"求解频率点 {i + 1}/{freqs.size}（ω={omega:.4g} rad/s）")
        dynamic = (k_ff + 1j * omega * c_ff - omega * omega * m_ff).tocsc()  # type: ignore[operator]
        try:
            lu = splu(dynamic)  # type: ignore[arg-type]
            displacements[free, i] = lu.solve(f_free)
        except (RuntimeError, ValueError) as exc:
            raise SolverError(f"频率点 ω={omega:.6g} rad/s 动刚度矩阵奇异：无阻尼共振或约束不足（{exc}）") from exc

    progress(1.0, "谐响应求解完成")
    return HarmonicResponse(mesh=mesh, frequencies=freqs, displacements=displacements)
