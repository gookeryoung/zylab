"""静力分析编排：装配 -> 求解 -> 结果后处理.

对外主入口 :func:`solve_static`，返回 :class:`StaticSolution`
（节点位移、约束反力、单元应力）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .assemble import assemble_loads, assemble_stiffness, element_dofs
from .boundary import Constraint, StaticCase
from .elements import element_stress
from .errors import SolverError
from .material import LinearElastic, Section
from .mesh import ElementType, Mesh
from .solve import solve_system

__all__ = ["ElementResult", "StaticSolution", "solve_static"]


@dataclass(frozen=True)
class ElementResult:
    """单元结果（常应变单元直接给出，等参单元取高斯点平均）.

    Attributes:
        block: 单元块索引。
        index: 块内单元索引。
        stress: 应力向量（杆为 (1,)，平面为 (3,)，空间为 (6,)）。
    """

    block: int
    index: int
    stress: np.ndarray


@dataclass(frozen=True)
class StaticSolution:
    """静力求解结果.

    Attributes:
        mesh: 参与求解的网格。
        displacements: 节点位移 ``(n_nodes, dim)``。
        reactions: 约束反力，键为约束自由度全局索引，值为反力。
        element_results: 单元应力结果列表（与网格块顺序对应展开）。
        strain_energy: 系统应变能 0.5 u^T K u。
    """

    mesh: Mesh
    displacements: np.ndarray
    reactions: dict[int, float]
    element_results: tuple[ElementResult, ...] = field(default=())
    strain_energy: float = 0.0

    def node_displacement(self, node: int) -> np.ndarray:
        """取指定节点位移向量 (dim,)."""
        return self.displacements[node]

    def element_stresses(self, etype: ElementType) -> list[ElementResult]:
        """按单元类型过滤应力结果."""
        return [r for r in self.element_results if self.mesh.blocks[r.block].etype is etype]


def solve_static(
    mesh: Mesh,
    materials: Sequence[LinearElastic],
    sections: Sequence[Section],
    case: StaticCase,
) -> StaticSolution:
    """线性静力分析主流程（小位移、线弹性）.

    Args:
        mesh: 网格。
        materials: 材料表（ElementBlock.material 索引引用）。
        sections: 截面表（ElementBlock.section 索引引用）。
        case: 静力工况（约束 + 节点载荷）。

    Returns:
        :class:`StaticSolution`，含位移/反力/单元应力/应变能。
    """
    case.validate(mesh)
    k_global = assemble_stiffness(mesh, materials, sections)
    force = assemble_loads(mesh, case)
    fixed_dofs, fixed_values = _expand_constraints(mesh, case.constraints)
    u, reactions = solve_system(k_global, force, fixed_dofs, fixed_values)

    reaction_map = {int(dof): float(val) for dof, val in zip(fixed_dofs, reactions)}
    results = _recover_stresses(mesh, materials, u)
    energy = 0.5 * float(u @ (k_global @ u))
    return StaticSolution(
        mesh=mesh,
        displacements=u.reshape(mesh.n_nodes, mesh.dim).copy(),
        reactions=reaction_map,
        element_results=results,
        strain_energy=energy,
    )


def _expand_constraints(mesh: Mesh, constraints: Sequence[Constraint]) -> tuple[np.ndarray, np.ndarray]:
    """节点局部约束展开为全局自由度索引与约束值（同一自由度取首个约束）."""
    dim = mesh.dim
    seen: dict[int, float] = {}
    for constraint in constraints:
        base = constraint.node * dim
        for dof in constraint.dofs:
            seen.setdefault(base + dof, constraint.value)
    if not seen:
        raise SolverError("静力工况缺少位移约束，方程组欠定")
    fixed = np.fromiter(seen.keys(), dtype=np.intp)
    values = np.fromiter(seen.values(), dtype=float)
    return fixed, values


def _recover_stresses(
    mesh: Mesh,
    materials: Sequence[LinearElastic],
    u: np.ndarray,
) -> tuple[ElementResult, ...]:
    """逐单元恢复应力（杆为轴向应力，连续体为应力向量）."""
    results: list[ElementResult] = []
    for block_index, block in enumerate(mesh.blocks):
        material = materials[block.material]
        for elem_index, conn in enumerate(block.conn):
            dofs = element_dofs(mesh, conn)
            stress = element_stress(block.etype, mesh.coords[conn], material, u[dofs])
            results.append(ElementResult(block=block_index, index=elem_index, stress=stress))
    return tuple(results)
