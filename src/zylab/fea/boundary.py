"""边界条件：节点约束（Dirichlet）与节点集中载荷.

约束/载荷均以「节点 + 节点内局部自由度」描述，装配阶段再映射到全局编号，
避免调用方直接操作全局自由度索引。
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import MeshError
from .mesh import Mesh

__all__ = ["Constraint", "NodalLoad", "StaticCase"]


@dataclass(frozen=True)
class Constraint:
    """节点位移约束（同一节点同一自由度重复约束时取首个值）.

    Attributes:
        node: 节点索引（0 基）。
        dofs: 节点内局部自由度索引元组（0 基；2D 为 0/1，3D 为 0/1/2）。
        value: 约定位移值（全部自由度共用；非零约束时按此值施加）。
    """

    node: int
    dofs: tuple[int, ...]
    value: float = 0.0


@dataclass(frozen=True)
class NodalLoad:
    """节点集中载荷.

    Attributes:
        node: 节点索引（0 基）。
        forces: 各自由度方向的力分量元组（长度须等于网格维数）。
    """

    node: int
    forces: tuple[float, ...]


@dataclass(frozen=True)
class StaticCase:
    """静力分析工况：一组约束 + 一组节点载荷."""

    constraints: tuple[Constraint, ...] = ()
    loads: tuple[NodalLoad, ...] = ()

    def validate(self, mesh: Mesh) -> None:
        """按网格校验节点索引与自由度编号范围."""
        dim = mesh.dim
        for constraint in self.constraints:
            _check_node(mesh, constraint.node, "约束")
            for dof in constraint.dofs:
                if not 0 <= dof < dim:
                    raise MeshError(f"约束节点 {constraint.node} 的自由度 {dof} 超出 [0, {dim})")
        for load in self.loads:
            _check_node(mesh, load.node, "载荷")
            if len(load.forces) != dim:
                raise MeshError(f"载荷节点 {load.node} 力分量数 {len(load.forces)} != 网格维数 {dim}")


def _check_node(mesh: Mesh, node: int, kind: str) -> None:
    """校验节点索引不越界."""
    if not 0 <= node < mesh.n_nodes:
        raise MeshError(f"{kind}引用节点 {node} 越界（共 {mesh.n_nodes} 节点）")
