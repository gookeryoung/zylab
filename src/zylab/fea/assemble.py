"""全局装配：单元刚度散布到 CSR 稀疏矩阵、载荷向量组装.

自由度编号规则：全局 DOF = node * dofs_per_node + 局部分量（0 基，逐节点连续）。
每节点自由度数由网格内最宽单元族决定（梁 3、余者 = 网格维数），连续体单元
只占用前 dim 个分量，梁单元附加转角分量。
装配采用 COO 三元组收集后转 CSR，避免 Python 端逐项修改稀疏矩阵。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix

from .boundary import BodyForce, EdgePressure, StaticCase
from .elements import element_geometric_stiffness, element_mass, element_measure, element_stiffness
from .errors import MeshError
from .material import LinearElastic, Section
from .mesh import ElementBlock, ElementType, Mesh

__all__ = [
    "assemble_geometric",
    "assemble_loads",
    "assemble_mass",
    "assemble_stiffness",
    "element_dofs",
]

# 连续体单元族（可施加体力；杆/梁的自重等分布载荷 v1 暂不涉及）
_CONTINUUM_TYPES = frozenset({ElementType.TRIA3, ElementType.QUAD4, ElementType.TET4, ElementType.HEX8})

# 支持几何刚度的单元族（v1 限杆/梁）
_GEOMETRIC_TYPES = frozenset({ElementType.TRUSS2, ElementType.BEAM2})


def element_dofs(mesh: Mesh, conn: np.ndarray) -> np.ndarray:
    """单元连接表展开为全局自由度编号（长度 = n_node_per_elem * dofs_per_node）."""
    width = mesh.dofs_per_node
    return (np.asarray(conn, dtype=np.intp)[:, None] * width + np.arange(width)[None, :]).ravel()


def assemble_stiffness(
    mesh: Mesh,
    materials: Sequence[LinearElastic],
    sections: Sequence[Section],
) -> csr_matrix:
    """装配全局刚度矩阵（CSR）.

    Args:
        mesh: 网格。
        materials: 材料表（ElementBlock.material 索引引用）。
        sections: 截面表（ElementBlock.section 索引引用）。

    Returns:
        全局刚度矩阵 ``(n_dofs, n_dofs)`` CSR 稀疏矩阵。

    Raises:
        MeshError: 材料/截面索引越界时抛出。
    """
    _check_tables(mesh, materials, sections)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for block in mesh.blocks:
        material = materials[block.material]
        section = sections[block.section]
        for conn in block.conn:
            dofs = element_dofs(mesh, conn)
            ke = element_stiffness(block.etype, mesh.coords[conn], material, section)
            n_dof_elem = dofs.size
            # ke.ravel() 行主序：(i,j) 元素对应 row=dofs[i]、col=dofs[j]，
            # 行索引重复 n 次、列索引平铺 n 次即可对齐
            rows.append(np.repeat(dofs, n_dof_elem))
            cols.append(np.tile(dofs, n_dof_elem))
            values.append(ke.ravel())
    if not rows:  # pragma: no cover（Mesh 校验已保证块非空，但 blocks 可为空元组）
        raise MeshError("网格无单元，无法装配刚度矩阵")
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    value = np.concatenate(values)
    return csr_matrix((value, (row, col)), shape=(mesh.n_dofs, mesh.n_dofs))


def assemble_mass(
    mesh: Mesh,
    materials: Sequence[LinearElastic],
    sections: Sequence[Section],
) -> csr_matrix:
    """装配全局一致质量矩阵（CSR）.

    Args:
        mesh: 网格。
        materials: 材料表（须配置正的质量密度）。
        sections: 截面表（ElementBlock.section 索引引用）。

    Returns:
        全局质量矩阵 ``(n_dofs, n_dofs)`` CSR 稀疏矩阵。

    Raises:
        MeshError: 材料/截面索引越界或密度非正时抛出。
    """
    _check_tables(mesh, materials, sections)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for block in mesh.blocks:
        material = materials[block.material]
        section = sections[block.section]
        for conn in block.conn:
            dofs = element_dofs(mesh, conn)
            me = element_mass(block.etype, mesh.coords[conn], material, section)
            n_dof_elem = dofs.size
            rows.append(np.repeat(dofs, n_dof_elem))
            cols.append(np.tile(dofs, n_dof_elem))
            values.append(me.ravel())
    if not rows:  # pragma: no cover（Mesh 校验已保证块非空，但 blocks 可为空元组）
        raise MeshError("网格无单元，无法装配质量矩阵")
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    value = np.concatenate(values)
    return csr_matrix((value, (row, col)), shape=(mesh.n_dofs, mesh.n_dofs))


def assemble_geometric(
    mesh: Mesh,
    axial_forces: Sequence[float],
) -> csr_matrix:
    """装配全局几何刚度矩阵（初应力刚度，CSR）.

    仅杆/梁单元贡献几何刚度；其余单元族位置跳过（贡献 0）。
    ``axial_forces`` 与网格单元展平序（块序 + 块内序）一一对应，
    连续体单元位置的数值被忽略。

    Args:
        mesh: 网格。
        axial_forces: 每单元参考态轴力（拉伸为正），长度 = 总单元数。

    Returns:
        全局几何刚度矩阵 ``(n_dofs, n_dofs)`` CSR 稀疏矩阵。

    Raises:
        MeshError: 轴力数组长度与单元数不符时抛出。
        ElementError: 单元几何退化时抛出（经 :func:`element_geometric_stiffness`）。
    """
    total = sum(block.conn.shape[0] for block in mesh.blocks)
    if len(axial_forces) != total:
        raise MeshError(f"轴力数组长度 {len(axial_forces)} 与单元总数 {total} 不符")
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []
    cursor = 0
    for block in mesh.blocks:
        for conn in block.conn:
            force = axial_forces[cursor]
            cursor += 1
            if block.etype in _GEOMETRIC_TYPES and force != 0.0:
                dofs = element_dofs(mesh, conn)
                kg = element_geometric_stiffness(block.etype, mesh.coords[conn], force)
                n_dof_elem = dofs.size
                rows.append(np.repeat(dofs, n_dof_elem))
                cols.append(np.tile(dofs, n_dof_elem))
                values.append(kg.ravel())
    shape = (mesh.n_dofs, mesh.n_dofs)
    if not rows:
        return csr_matrix(shape)
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    value = np.concatenate(values)
    return csr_matrix((value, (row, col)), shape=shape)


def assemble_loads(
    mesh: Mesh,
    case: StaticCase,
    sections: Sequence[Section] = (),
) -> np.ndarray:
    """组装全局载荷向量（节点集中力 + 边压力 + 体力）.

    Args:
        mesh: 网格。
        case: 静力工况（节点约束不参与，仅载荷）。
        sections: 截面表（体力按 2D 单元厚度换算体积时引用；无体力可省略）。

    Returns:
        全局载荷向量 ``(n_dofs,)``。

    Raises:
        MeshError: 体力工况缺少截面表或截面索引越界时抛出。
    """
    case.validate(mesh)
    width = mesh.dofs_per_node
    force = np.zeros(mesh.n_dofs)
    for load in case.loads:
        base = load.node * width
        force[base : base + width] += np.asarray(load.forces, dtype=float)
    for pressure in case.edge_pressures:
        _apply_edge_pressure(mesh, pressure, force)
    if case.body_forces:
        _apply_body_forces(mesh, case.body_forces, sections, force)
    return force


def _apply_edge_pressure(mesh: Mesh, pressure: EdgePressure, force: np.ndarray) -> None:
    """边压力折线逐段化为一致节点力（线性单元 pL/2 分配两端）.

    节点序使材料位于行进方向左侧，行进方向左法向指向材料内部；
    正压力（压缩）沿该法向，负值为外向拉力。
    """
    coords = mesh.coords
    for here, there in zip(pressure.nodes[:-1], pressure.nodes[1:]):
        segment = coords[there] - coords[here]
        length = float(np.linalg.norm(segment))
        if length <= 0.0:
            raise MeshError("边压力折线出现重复相邻节点，段长为零")
        # 行进方向左法向（指向材料内部）
        normal = np.array([-segment[1], segment[0]]) / length
        nodal = 0.5 * pressure.pressure * length * normal
        for node in (here, there):
            base = node * mesh.dofs_per_node
            force[base : base + 2] += nodal


def _apply_body_forces(
    mesh: Mesh,
    bodies: Sequence[BodyForce],
    sections: Sequence[Section],
    force: np.ndarray,
) -> None:
    """均匀体力化为连续体单元一致节点力（线性单元度量均分）.

    每节点分得 measure/n_node * b（T3/T4/直边 Q4/平行六面体精确，
    一般六面体为等分近似）。2D 按截面厚度换算体积。
    """
    dim = mesh.dim
    resultant = np.zeros(dim)
    for body in bodies:
        resultant += (body.fx, body.fy) if dim == 2 else (body.fx, body.fy, body.fz)
    for block in mesh.blocks:
        if block.etype not in _CONTINUUM_TYPES:
            continue
        if dim == 2:
            # 2D 体力按截面厚度换算体积，须提供截面表
            if not 0 <= block.section < len(sections):
                raise MeshError(
                    f"体力装配须提供截面表，单元块 {block.etype.value} 截面索引 {block.section} 越界（共 {len(sections)} 项）"
                )
            thickness = sections[block.section].thickness
        else:
            thickness = 1.0
        for conn in block.conn:
            measure = element_measure(block.etype, mesh.coords[conn])
            total = measure * thickness
            share = total / conn.size
            for node in conn:
                base = node * mesh.dofs_per_node
                force[base : base + dim] += share * resultant


def _check_tables(
    mesh: Mesh,
    materials: Sequence[LinearElastic],
    sections: Sequence[Section],
) -> None:
    """校验各单元块引用的材料/截面索引不越界."""
    for block in mesh.blocks:
        if not 0 <= block.material < len(materials):
            raise MeshError(f"单元块 {block.etype.value} 材料索引 {block.material} 越界（共 {len(materials)} 项）")
        if not 0 <= block.section < len(sections):
            raise MeshError(f"单元块 {block.etype.value} 截面索引 {block.section} 越界（共 {len(sections)} 项）")


def block_at(mesh: Mesh, index: int) -> ElementBlock:
    """按索引取单元块（供测试与结果后处理定位）."""
    if not 0 <= index < len(mesh.blocks):
        raise MeshError(f"单元块索引 {index} 越界（共 {len(mesh.blocks)} 块）")
    return mesh.blocks[index]
