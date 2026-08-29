"""全局装配：单元刚度散布到 CSR 稀疏矩阵、载荷向量组装.

自由度编号规则：全局 DOF = node * dim + 局部分量（0 基，逐节点连续）。
装配采用 COO 三元组收集后转 CSR，避免 Python 端逐项修改稀疏矩阵。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix

from .boundary import StaticCase
from .elements import element_stiffness
from .errors import MeshError
from .material import LinearElastic, Section
from .mesh import ElementBlock, Mesh

__all__ = ["assemble_loads", "assemble_stiffness", "element_dofs"]


def element_dofs(mesh: Mesh, conn: np.ndarray) -> np.ndarray:
    """单元连接表展开为全局自由度编号（长度 = n_node_per_elem * dim）."""
    dim = mesh.dim
    return (np.asarray(conn, dtype=np.intp)[:, None] * dim + np.arange(dim)[None, :]).ravel()


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


def assemble_loads(mesh: Mesh, case: StaticCase) -> np.ndarray:
    """组装全局载荷向量.

    Args:
        mesh: 网格。
        case: 静力工况（节点约束不参与，仅载荷）。

    Returns:
        全局载荷向量 ``(n_dofs,)``。
    """
    case.validate(mesh)
    dim = mesh.dim
    force = np.zeros(mesh.n_dofs)
    for load in case.loads:
        base = load.node * dim
        force[base : base + dim] += np.asarray(load.forces, dtype=float)
    return force


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
