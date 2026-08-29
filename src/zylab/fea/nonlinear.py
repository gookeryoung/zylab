"""几何非线性静力分析（总拉格朗日 + 载荷增量 Newton-Raphson 迭代）.

对外主入口 :func:`solve_nonlinear_static`。v2 范围：TRUSS2 大位移/大转动
（Green-Lagrange 应变，几何精确）；外载荷限节点集中力（分布载荷随
构型变化推迟）；约束值须为 0。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

from .assemble import assemble_loads, element_dofs
from .boundary import Constraint, NodalLoad, StaticCase
from .elements import truss2_internal_force, truss2_tangent_stiffness
from .errors import SolverError
from .material import LinearElastic, Section
from .mesh import ElementType, Mesh

__all__ = ["NonlinearSolution", "solve_nonlinear_static"]


def _no_report(_progress: float, _message: str) -> None:
    """空进度回调（未注入 report 时使用）."""


@dataclass(frozen=True)
class NonlinearSolution:
    """几何非线性静力求解结果.

    Attributes:
        mesh: 参与求解的网格。
        displacements: 收敛位移 ``(n_nodes, dofs_per_node)``。
        load_factor: 收敛态载荷因子（正常收敛 = 1.0）。
        iterations: 每增量步的 Newton 迭代次数（长度 = 增量步数）。
        residual_norm: 收敛态自由自由度残差范数。
        converged: 是否全部增量步收敛。
    """

    mesh: Mesh
    displacements: np.ndarray
    load_factor: float
    iterations: tuple[int, ...]
    residual_norm: float
    converged: bool
    #: 每增量步收敛后的载荷因子序列（含 0 起始，长度 = 步数 + 1）
    history_factors: np.ndarray = field(default_factory=lambda: np.zeros(1))
    #: 每增量步收敛后的位移快照 (步数 + 1, n_nodes, dofs_per_node)，首帧为零位移
    history_displacements: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 1)))

    @property
    def total_iterations(self) -> int:
        """全部增量步 Newton 迭代次数合计."""
        return sum(self.iterations)

    def history_dof(self, node: int, component: int = 0) -> np.ndarray:
        """指定节点分量随增量步的位移序列（载荷-位移曲线数据源）.

        Args:
            node: 节点序号。
            component: 节点内自由度分量（0=x，1=y）。

        Returns:
            长度 = 增量步数 + 1 的一维数组（含零位移起始）。
        """
        return self.history_displacements[:, node, component]


def solve_nonlinear_static(  # noqa: PLR0913  求解四要素 + 步进控制参数，语义不可合并
    mesh: Mesh,
    materials: Sequence[LinearElastic],
    sections: Sequence[Section],
    case: StaticCase,
    *,
    n_increments: int = 10,
    tolerance: float = 1.0e-8,
    max_iterations: int = 30,
    report: Callable[[float, str], None] | None = None,
) -> NonlinearSolution:
    """几何非线性静力分析（载荷增量 + Newton-Raphson 平衡迭代）.

    每增量步施加载荷 ``(i/n_increments)·f_ext``，Newton 迭代求解
    ``R = f_ext − f_int(u) = 0``：切线刚度划块消元解修正量 Δu，
    直至相对残差小于 tolerance。TRUSS2 单元几何精确
    （刚体转动零应变），小位移极限退化为线性解。

    Args:
        mesh: 网格（v2 须全为 TRUSS2 单元）。
        materials: 材料表。
        sections: 截面表。
        case: 载荷工况（约束值须为 0；载荷限节点集中力）。
        n_increments: 载荷增量步数（>= 1）。
        tolerance: 残差收敛容差（相对自由自由度外载荷范数）。
        max_iterations: 每增量步最大 Newton 迭代次数。
        report: 进度回调 ``(progress, message)``；进程执行器自动注入。

    Returns:
        :class:`NonlinearSolution`，含收敛位移与迭代历程。

    Raises:
        SolverError: 单元类型不支持、约束非法、载荷类型不支持、
            或 Newton 迭代不收敛时抛出。
    """
    progress = _no_report if report is None else report
    _validate_model(mesh, case)
    if n_increments < 1:
        raise SolverError(f"载荷增量步数须至少为 1，实际 {n_increments}")
    if tolerance <= 0.0:
        raise SolverError(f"收敛容差须为正，实际 {tolerance}")

    blocks = mesh.blocks
    # 单元遍历预备：每单元 (coords, e_modulus, area, dofs)
    element_cache: list[tuple[np.ndarray, float, float, np.ndarray]] = []
    for block in blocks:
        material = materials[block.material]
        area = sections[block.section].area
        for conn in block.conn:
            dofs = element_dofs(mesh, conn)
            element_cache.append((mesh.coords[conn], material.e_modulus, area, dofs))

    f_ext = np.asarray(assemble_loads(mesh, case, sections), dtype=float)
    free = _free_dofs(mesh, case.constraints)
    f_ref = np.linalg.norm(f_ext[free])
    if f_ref <= 0.0:
        raise SolverError("非线性分析缺少非零外载荷（自由自由度上外力范数为零）")

    u = np.zeros(mesh.n_dofs)
    iterations: list[int] = []
    converged = True
    # 全过程追踪：载荷因子 + 每步收敛位移快照（载荷-位移曲线数据源）
    factors: list[float] = [0.0]
    snapshots: list[np.ndarray] = [np.zeros((mesh.n_nodes, mesh.dofs_per_node))]
    for step in range(1, n_increments + 1):
        factor = step / n_increments
        target = f_ext * factor
        for _iteration in range(1, max_iterations + 1):
            f_int = _assemble_internal_forces(element_cache, u)
            residual = (target - f_int)[free]
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm <= tolerance * max(f_ref * factor, 1.0e-30):
                break
            k_tangent = _assemble_tangent(element_cache, u)
            delta = _solve_free(k_tangent, residual, free)
            u[free] += delta
        else:
            converged = False
            raise SolverError(
                f"第 {step}/{n_increments} 步 Newton 迭代 {max_iterations} 次未收敛，可增大增量步数或放宽容差"
            )
        iterations.append(_iteration)
        factors.append(factor)
        snapshots.append(u.reshape(mesh.n_nodes, mesh.dofs_per_node).copy())
        progress(step / n_increments, f"增量步 {step}/{n_increments} 完成（{_iteration} 次迭代）")

    displacements = u.reshape(mesh.n_nodes, mesh.dofs_per_node)
    return NonlinearSolution(
        mesh=mesh,
        displacements=displacements,
        load_factor=1.0,
        iterations=tuple(iterations),
        residual_norm=residual_norm,
        converged=converged,
        history_factors=np.asarray(factors),
        history_displacements=np.stack(snapshots),
    )


def _validate_model(mesh: Mesh, case: StaticCase) -> None:
    """校验模型范围：全 TRUSS2、节点载荷、零值约束."""
    for block in mesh.blocks:
        if block.etype is not ElementType.TRUSS2:
            raise SolverError(f"几何非线性 v2 仅支持 TRUSS2 单元，遇到 {block.etype}")
    for load in case.loads:
        if not isinstance(load, NodalLoad):
            raise SolverError(f"几何非线性 v2 载荷限节点集中力，遇到 {type(load).__name__}")
    width = mesh.dofs_per_node
    seen: set[int] = set()
    for constraint in case.constraints:
        if not 0 <= constraint.node < mesh.n_nodes:
            raise SolverError(f"约束引用节点 {constraint.node} 越界（共 {mesh.n_nodes} 节点）")
        for dof in constraint.dofs:
            if not 0 <= dof < width:
                raise SolverError(f"约束节点 {constraint.node} 的自由度 {dof} 超出 [0, {width})")
            if constraint.value != 0.0:
                raise SolverError("几何非线性 v2 约束值须为 0（非零给定位移推迟）")
            seen.add(constraint.node * width + dof)
    if not seen:
        raise SolverError("非线性分析缺少位移约束")


def _free_dofs(mesh: Mesh, constraints: Sequence[Constraint]) -> np.ndarray:
    """展开约束并返回自由自由度索引（升序）."""
    width = mesh.dofs_per_node
    fixed = {c.node * width + dof for c in constraints for dof in c.dofs}
    mask = np.ones(mesh.n_dofs, dtype=bool)
    for dof in fixed:
        mask[dof] = False
    return np.flatnonzero(mask)


def _assemble_internal_forces(
    element_cache: Sequence[tuple[np.ndarray, float, float, np.ndarray]],
    u: np.ndarray,
) -> np.ndarray:
    """装配当前位移态的全局内力向量."""
    f_int = np.zeros(u.size)
    for coords, e_modulus, area, dofs in element_cache:
        f_int[dofs] += truss2_internal_force(coords, u[dofs], e_modulus, area)
    return f_int


def _assemble_tangent(
    element_cache: Sequence[tuple[np.ndarray, float, float, np.ndarray]],
    u: np.ndarray,
) -> csr_matrix:
    """装配当前位移态的全局切线刚度（CSR）."""
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for coords, e_modulus, area, dofs in element_cache:
        kt = truss2_tangent_stiffness(coords, u[dofs], e_modulus, area)
        n_dof_elem = dofs.size
        rows.append(np.repeat(dofs, n_dof_elem))
        cols.append(np.tile(dofs, n_dof_elem))
        values.append(kt.ravel())
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    value = np.concatenate(values)
    return csr_matrix((value, (row, col)), shape=(u.size, u.size))


def _solve_free(k_tangent: csr_matrix, residual: np.ndarray, free: np.ndarray) -> np.ndarray:
    """自由自由度划块消元求解切线方程 K_ff Δu = R."""
    k_ff = k_tangent[free][:, free]  # type: ignore[bad-index]
    try:
        return np.asarray(splu(k_ff.tocsc()).solve(residual))  # type: ignore[union-attr]
    except RuntimeError as exc:
        raise SolverError("切线刚度奇异：结构失稳或存在机构") from exc
