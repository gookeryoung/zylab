"""线性屈曲分析编排：参考态静力解 -> 单元轴力 -> 几何刚度装配 -> 特征值屈曲.

对外主入口 :func:`solve_buckling`，返回 :class:`BucklingSolution`
（临界载荷因子与屈曲振型）。特征值问题 ``(K + λ·K_G)φ = 0`` 中 λ 为
参考载荷的放大倍数；因 K_G 一般不定（拉压混合），采用稠密广义特征值
求解并取最小正因子（v1 规模，稀疏迭代推迟）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import linalg

from .assemble import assemble_geometric, assemble_stiffness
from .boundary import Constraint, StaticCase
from .errors import SolverError
from .material import LinearElastic, Section
from .mesh import ElementType, Mesh
from .static import StaticSolution, solve_static

__all__ = ["BucklingSolution", "solve_buckling"]


def _no_report(_progress: float, _message: str) -> None:
    """空进度回调（未注入 report 时使用）."""


@dataclass(frozen=True)
class BucklingSolution:
    """线性屈曲分析求解结果.

    Attributes:
        mesh: 参与求解的网格。
        load_factors: 临界载荷因子 ``(n_modes,)``（升序，均大于 0）；
            临界载荷 = 因子 × 参考工况载荷。
        mode_shapes: 屈曲振型 ``(n_dofs, n_modes)``（最大分量绝对值归一），
            约束自由度分量为 0。
        reference: 参考态线性静力解（轴力即源于此）。
    """

    mesh: Mesh
    load_factors: np.ndarray
    mode_shapes: np.ndarray
    reference: StaticSolution

    @property
    def n_modes(self) -> int:
        """屈曲阶数."""
        return int(self.load_factors.size)

    def mode_shape(self, index: int) -> np.ndarray:
        """取指定阶屈曲振型并整形为 ``(n_nodes, dofs_per_node)``."""
        return self.mode_shapes[:, index].reshape(self.mesh.n_nodes, self.mesh.dofs_per_node).copy()


def solve_buckling(  # noqa: PLR0913  静力四要素 + 阶数与回调，语义不可合并
    mesh: Mesh,
    materials: Sequence[LinearElastic],
    sections: Sequence[Section],
    case: StaticCase,
    *,
    n_modes: int = 5,
    report: Callable[[float, str], None] | None = None,
) -> BucklingSolution:
    """线性特征值屈曲分析（杆/梁结构）.

    流程：参考态静力求解 -> 由单元轴向应力提取轴力 -> 装配几何刚度
    ``K_G`` -> 求解 ``(K + λ·K_G)φ = 0`` 的最小正特征值（临界载荷因子）。

    Args:
        mesh: 网格（v1 须为杆/梁单元；连续体单元无几何刚度贡献）。
        materials: 材料表。
        sections: 截面表（ElementBlock.section 索引引用）。
        case: 参考载荷工况（约束 + 载荷；临界载荷 = 因子 × 工况载荷）。
        n_modes: 提取屈曲阶数。
        report: 进度回调 ``(progress, message)``；进程执行器自动注入。

    Returns:
        :class:`BucklingSolution`，含升序临界载荷因子与屈曲振型。

    Raises:
        SolverError: 约束不足、无正特征值（参考态不受压）、
            阶数越界或特征值求解失败时抛出。
    """
    progress = _no_report if report is None else report

    progress(0.2, "参考态线性静力求解")
    reference = solve_static(mesh, materials, sections, case)
    progress(0.5, "提取单元轴力并装配几何刚度")
    axial_forces = _extract_axial_forces(mesh, sections, reference)
    k_global = assemble_stiffness(mesh, materials, sections)
    kg_global = assemble_geometric(mesh, axial_forces)

    fixed_dofs = _expand_fixed(mesh, case.constraints)
    mask = np.zeros(mesh.n_dofs, dtype=bool)
    mask[fixed_dofs] = True
    free = np.flatnonzero(~mask)
    if n_modes >= free.size:
        raise SolverError(f"屈曲阶数 {n_modes} 须小于自由自由度数 {free.size}")
    if n_modes < 1:
        raise SolverError(f"屈曲阶数须至少为 1，实际 {n_modes}")

    progress(0.8, "求解特征值屈曲问题")
    k_ff = k_global[free][:, free].toarray()  # type: ignore[bad-index]
    kg_ff = kg_global[free][:, free].toarray()  # type: ignore[bad-index]
    try:
        # K_G 不定（拉压混合），Cholesky 路径不可用，用稠密 QZ 广义特征值
        raw_values, raw_vectors = linalg.eig(k_ff, -kg_ff)
    except (linalg.LinAlgError, ValueError) as exc:
        raise SolverError(f"屈曲特征值求解失败：{exc}") from exc
    # scipy 存根将 eig 返回标注为 tuple，asarray 收敛为 ndarray
    eigenvalues = np.asarray(raw_values)
    eigenvectors = np.asarray(raw_vectors)

    # 取正实有限特征值：-K_G 奇异（轴向零空间）时 QZ 会返回 inf，
    # 须排除；数值噪声取实部
    real = np.real_if_close(eigenvalues, tol=1000)
    if np.iscomplexobj(real):
        raise SolverError("屈曲特征值出现显著虚部，参考态或约束异常")
    finite = np.isfinite(real)
    positive = np.flatnonzero(finite & (real > _FACTOR_TOL))
    if positive.size == 0:
        raise SolverError("参考态无压缩轴力，不存在屈曲（载荷因子全为负或零）")
    order = positive[np.argsort(real[positive])][:n_modes]
    factors = np.array(real[order], dtype=float)

    shapes = np.zeros((mesh.n_dofs, order.size))
    for column, source in enumerate(order):
        vector = np.real(eigenvectors[:, source])
        vector /= np.max(np.abs(vector))  # 最大分量绝对值归一
        shapes[free, column] = vector
    progress(1.0, "屈曲求解完成")
    return BucklingSolution(mesh=mesh, load_factors=factors, mode_shapes=shapes, reference=reference)


_FACTOR_TOL = 1.0e-12


def _extract_axial_forces(
    mesh: Mesh,
    sections: Sequence[Section],
    reference: StaticSolution,
) -> list[float]:
    """从参考静力解提取每单元轴力（展平序，拉伸为正）.

    杆/梁取 ``σ_axial × A``；连续体单元位置填 0（无几何刚度贡献）。
    """
    # (block_index, elem_index) -> 轴向应力，按 etype 分别查询参考解
    stress_map: dict[tuple[int, int], float] = {}
    for etype in (ElementType.TRUSS2, ElementType.BEAM2):
        for result in reference.element_stresses(etype):
            stress_map[(result.block, result.index)] = float(result.stress[0])
    forces: list[float] = []
    for block_index, block in enumerate(mesh.blocks):
        area = sections[block.section].area
        for elem_index in range(block.conn.shape[0]):
            stress = stress_map.get((block_index, elem_index))
            forces.append(stress * area if stress is not None else 0.0)
    return forces


def _expand_fixed(mesh: Mesh, constraints: Sequence[Constraint]) -> np.ndarray:
    """展开约束为全局自由度索引（屈曲约束值须为 0，与模态同理）."""
    width = mesh.dofs_per_node
    seen: dict[int, float] = {}
    for constraint in constraints:
        if not 0 <= constraint.node < mesh.n_nodes:
            raise SolverError(f"约束引用节点 {constraint.node} 越界（共 {mesh.n_nodes} 节点）")
        for dof in constraint.dofs:
            if not 0 <= dof < width:
                raise SolverError(f"约束节点 {constraint.node} 的自由度 {dof} 超出 [0, {width})")
            seen.setdefault(constraint.node * width + dof, constraint.value)
    if not seen:
        raise SolverError("屈曲分析缺少位移约束")
    nonzero = {dof: value for dof, value in seen.items() if value != 0.0}
    if nonzero:
        raise SolverError("屈曲分析约束值须为 0（参考态位移由静力解承担）")
    return np.fromiter(seen.keys(), dtype=np.intp)
