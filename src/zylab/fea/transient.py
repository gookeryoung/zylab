"""瞬态动力分析编排：装配 K/M/C -> 约束划块 -> Newmark 直接时间积分.

对外主入口 :func:`solve_transient`，返回 :class:`TransientSolution`
（全时程位移/速度/加速度场）。积分采用 Newmark 平均加速度法
（``β=1/4, γ=1/2``，无条件稳定、无数值阻尼），阻尼为 Rayleigh 模型
``C = αM + βK``。载荷时程由 ``load_fn(t)`` 对工况装配的空间载荷向量
整体缩放实现（分布不变、幅值随时间变化）。
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
from .harmonic import rayleigh_damping
from .material import LinearElastic, Section
from .mesh import Mesh
from .modal import _validate_constraints

__all__ = ["TransientSolution", "solve_transient"]

# Newmark 平均加速度法参数（无条件稳定、无数值阻尼）
_BETA = 0.25
_GAMMA = 0.5


def _no_report(_progress: float, _message: str) -> None:
    """空进度回调（未注入 report 时使用）."""


@dataclass(frozen=True)
class TransientSolution:
    """瞬态动力分析结果.

    Attributes:
        mesh: 参与求解的网格。
        times: 时间站点序列 ``(n_times,)``（含初始时刻 t=0，严格递增）。
        displacements: 全时程位移场 ``(n_dofs, n_times)``，约束自由度分量为 0。
        velocities: 全时程速度场 ``(n_dofs, n_times)``。
        accelerations: 全时程加速度场 ``(n_dofs, n_times)``。
    """

    mesh: Mesh
    times: np.ndarray
    displacements: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray

    @property
    def n_steps(self) -> int:
        """积分步数（时间站点数减一）."""
        return int(self.times.size - 1)

    @property
    def dt(self) -> float:
        """时间步长（均匀步长；非均匀时程取首步）."""
        return float(self.times[1] - self.times[0])

    def node_history(self, node: int, comp: int) -> np.ndarray:
        """取指定节点指定分量随时间的位移序列 ``(n_times,)``.

        Args:
            node: 节点索引（0 基）。
            comp: 节点内局部自由度索引（0 基）。

        Raises:
            SolverError: 节点或自由度编号越界时抛出。
        """
        if not 0 <= node < self.mesh.n_nodes:
            raise SolverError(f"节点索引 {node} 越界（共 {self.mesh.n_nodes} 节点）")
        if not 0 <= comp < self.mesh.dofs_per_node:
            raise SolverError(f"自由度 {comp} 超出 [0, {self.mesh.dofs_per_node})")
        base = node * self.mesh.dofs_per_node
        return self.displacements[base + comp, :].copy()


def solve_transient(  # noqa: PLR0913  与静力/模态共用四要素 + 时程/载荷/阻尼/初值，语义不可合并
    mesh: Mesh,
    materials: Sequence[LinearElastic],
    sections: Sequence[Section],
    case: StaticCase,
    *,
    duration: float,
    n_steps: int,
    load_fn: Callable[[float], float] | None = None,
    alpha: float = 0.0,
    beta: float = 0.0,
    initial_velocity: Sequence[float] | None = None,
    report: Callable[[float, str], None] | None = None,
) -> TransientSolution:
    """线性瞬态动力分析（Newmark 平均加速度法直接积分）.

    求解 ``M ü + C u̇ + K u = λ(t) f``，其中 ``f`` 为工况装配的载荷向量、
    ``λ(t)`` 为 ``load_fn`` 给定的时程因子（缺省恒为 1，即阶跃施加载荷），
    ``C = αM + βK``。初始位移取零，初始速度由 ``initial_velocity``
    指定（缺省零）。约束按划块消元处理（约束值须为 0）。

    Args:
        mesh: 网格。
        materials: 材料表（须配置正的质量密度）。
        sections: 截面表（ElementBlock.section 索引引用）。
        case: 载荷工况（约束参与划块；边压力/体力同静力装配）。
        duration: 总时长（> 0）。
        n_steps: 积分步数（>= 1，步长 = duration / n_steps）。
        load_fn: 载荷时程因子函数 ``λ(t)``；缺省恒 1。
        alpha: Rayleigh 质量比例阻尼系数。
        beta: Rayleigh 刚度比例阻尼系数。
        initial_velocity: 初始速度全场向量 ``(n_dofs,)``；缺省零。
        report: 进度回调 ``(progress, message)``；进程执行器会自动注入。

    Returns:
        :class:`TransientSolution`，含全时程位移/速度/加速度场
        （时间站点含 t=0，共 ``n_steps + 1`` 个）。

    Raises:
        SolverError: 时长/步数/阻尼/初值参数非法、约束缺失或非零、
            或有效刚度矩阵奇异（约束不足）时抛出。
    """
    progress = _no_report if report is None else report

    if duration <= 0.0:
        raise SolverError(f"总时长须为正，实际 {duration}")
    if n_steps < 1:
        raise SolverError(f"积分步数须至少为 1，实际 {n_steps}")
    if alpha < 0.0 or beta < 0.0:
        raise SolverError(f"Rayleigh 阻尼系数须非负，实际 alpha={alpha}, beta={beta}")

    v0 = np.zeros(mesh.n_dofs) if initial_velocity is None else np.asarray(initial_velocity, dtype=float)
    if v0.ndim != 1 or v0.size != mesh.n_dofs:
        raise SolverError(f"初始速度须为一维数组且长度为总自由度数 {mesh.n_dofs}，实际形状 {v0.shape}")

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
    c_ff: csr_matrix = rayleigh_damping(alpha, beta, m_ff, k_ff)  # type: ignore[arg-type]

    dt = duration / n_steps
    times = np.linspace(0.0, duration, n_steps + 1)

    # Newmark 积分常数（平均加速度法）
    a0 = 1.0 / (_BETA * dt * dt)
    a1 = _GAMMA / (_BETA * dt)
    a2 = 1.0 / (_BETA * dt)
    a3 = 1.0 / (2.0 * _BETA) - 1.0
    a4 = _GAMMA / _BETA - 1.0
    a5 = 0.5 * dt * (_GAMMA / _BETA - 2.0)

    # 有效刚度 K_eff = K + a0 M + a1 C（均匀步长下全程不变，分解一次）
    k_eff = (k_ff + a0 * m_ff + a1 * c_ff).tocsc()  # type: ignore[operator]
    progress(0.45, "分解有效刚度矩阵")
    try:
        lu = splu(k_eff)  # type: ignore[arg-type]
    except (RuntimeError, ValueError) as exc:
        raise SolverError(f"有效刚度矩阵奇异：约束不足或存在机构（{exc}）") from exc

    u = np.zeros(free.size)
    v = v0[free].copy()
    # 初始加速度 = M⁻¹(f(0) - C v0)（初始位移为零）：须对质量矩阵求解，
    # 不能复用 K_eff 的分解（K_eff = K + a0 M + a1 C ≠ M）
    time_factor = 1.0 if load_fn is None else float(load_fn(0.0))
    try:
        m_lu = splu(m_ff.tocsc())  # type: ignore[arg-type, missing-attribute]
    except (RuntimeError, ValueError) as exc:
        raise SolverError(f"质量矩阵奇异：质量密度缺失或约束异常（{exc}）") from exc
    acc = m_lu.solve(force[free] * time_factor - c_ff @ v)
    displacements = np.zeros((mesh.n_dofs, n_steps + 1))
    velocities = np.zeros((mesh.n_dofs, n_steps + 1))
    accelerations = np.zeros((mesh.n_dofs, n_steps + 1))
    displacements[free, 0], velocities[free, 0], accelerations[free, 0] = u, v, acc

    for step in range(1, n_steps + 1):
        progress(0.45 + 0.5 * step / n_steps, f"积分时间步 {step}/{n_steps}（t={times[step]:.4g}）")
        # 有效载荷 f_eff = f(t+dt) + M(a0 u + a2 v + a3 a) + C(a1 u + a4 v + a5 a)
        time_factor = 1.0 if load_fn is None else float(load_fn(times[step]))
        f_eff = force[free] * time_factor + m_ff @ (a0 * u + a2 * v + a3 * acc) + c_ff @ (a1 * u + a4 * v + a5 * acc)
        u_new = lu.solve(f_eff)
        acc_new = a0 * (u_new - u) - a2 * v - a3 * acc
        v_new = v + dt * ((1.0 - _GAMMA) * acc + _GAMMA * acc_new)
        u, v, acc = u_new, v_new, acc_new
        displacements[free, step], velocities[free, step], accelerations[free, step] = u, v, acc

    progress(1.0, "瞬态求解完成")
    return TransientSolution(
        mesh=mesh,
        times=times,
        displacements=displacements,
        velocities=velocities,
        accelerations=accelerations,
    )
